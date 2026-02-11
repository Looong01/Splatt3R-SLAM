"""
Splatt3R utilities for SLAM
Adapted from mast3r_utils.py to use Splatt3R models and Gaussian splatting.
"""

import PIL
import numpy as np
import torch
import einops
import sys
import os

# Add splatt3r_core to path (also done via __init__.py/_setup_paths.py)
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_splatt3r_core_dir = os.path.join(_root_dir, "splatt3r_core")
if _splatt3r_core_dir not in sys.path:
    sys.path.insert(0, _splatt3r_core_dir)

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from splatt3r_core.main import MAST3RGaussians
from splatt3r_slam.retrieval_database import RetrievalDatabase
from splatt3r_slam.config import config
import splatt3r_slam.matching as matching

# Gaussian Splatting utilities (resolved via splatt3r_core on sys.path)
from utils.geometry import build_covariance
from utils.sh_utils import RGB2SH


def load_splatt3r(path=None, device="cuda"):
    """
    Load Splatt3R model (with Gaussian splatting capabilities).

    Args:
        path: Path to checkpoint. If None, downloads from HuggingFace.
        device: Device to load model on.

    Returns:
        Splatt3R model (MAST3RGaussians)
    """
    if path is None:
        from huggingface_hub import hf_hub_download

        model_name = "brandonsmart/splatt3r_v1.0"
        filename = "epoch=19-step=1200.ckpt"
        print(f"Downloading Splatt3R checkpoint from {model_name}")
        weights_path = hf_hub_download(repo_id=model_name, filename=filename)
    else:
        weights_path = path

    print(f"Loading Splatt3R model from {weights_path}")
    model = MAST3RGaussians.load_from_checkpoint(weights_path, device)
    model.eval()
    return model


def load_retriever(splatt3r_model, retriever_path=None, device="cuda"):
    """
    Load retrieval database. Uses the encoder from Splatt3R model.

    Args:
        splatt3r_model: Splatt3R model with encoder
        retriever_path: Path to retriever checkpoint
        device: Device to load on

    Returns:
        RetrievalDatabase instance
    """
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(
        retriever_path, backbone=splatt3r_model.encoder, device=device
    )
    return retriever


@torch.inference_mode()
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    """Decode features using Splatt3R decoder"""
    dec1, dec2 = model.encoder._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    """Downsample predictions according to config"""
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


# =============================================================================
# Gaussian Splatting rendering integration
# =============================================================================


def _extract_gaussian_params(res):
    """Extract Gaussian-specific parameters from decoder result dict.

    Returns a dict with cloned tensors so the decoder intermediates can be freed.
    Keys: means (B,H,W,3), scales (B,H,W,3), rotations (B,H,W,4),
          sh (B,H,W,3,sh_degree), opacities (B,H,W,1)
    """
    return {
        "means": res["means"].clone(),
        "scales": res["scales"].clone(),
        "rotations": res["rotations"].clone(),
        "sh": res["sh"].clone(),
        "opacities": res["opacities"].clone(),
    }


def _get_original_img_hwc(frame_img):
    """Convert normalised frame.img tensor to (B, H, W, 3) in [0, 1] range.

    ImgNorm uses mean=0.5, std=0.5 → normalised = 2*img - 1.
    """
    img = frame_img
    if img.dim() == 3:  # (C, H, W) – from SharedKeyframes
        img = img.unsqueeze(0)
    img = img * 0.5 + 0.5
    img = img.clamp(0, 1)
    return einops.rearrange(img, "b c h w -> b h w c")


def _sim3_to_4x4(T_sim3):
    """Convert lietorch.Sim3 to a 4x4 matrix [sR | t ; 0 0 0 1]."""
    import lietorch as _lt

    data = T_sim3.data.detach()
    if data.dim() == 1:
        data = data.unsqueeze(0)
    t, q, s = data.split([3, 4, 1], dim=-1)
    se3 = _lt.SE3(torch.cat([t, q], dim=-1))
    mat = se3.matrix()  # (..., 4, 4)
    # Factor scale into the rotation block: Sim3 acts as x' = sRx + t
    mat[..., :3, :3] = mat[..., :3, :3] * s.unsqueeze(-1)
    return mat.to(device=T_sim3.data.device, dtype=torch.float32)


def _estimate_default_intrinsics(h, w, device="cuda"):
    """Rough intrinsics when calibration is unavailable."""
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0
    return torch.tensor(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]],
        device=device,
        dtype=torch.float32,
    )


@torch.inference_mode()
def gaussians_to_world(frame, include_cross=True, spatial_stride=1):
    """Convert camera-local Gaussian predictions to world coordinates.

    Args:
        frame:  Frame with gaussian_pred (and optionally gaussian_pred_cross) set.
        include_cross: Also convert cross-predictions and concatenate.
        spatial_stride: Subsample Gaussians spatially (stride in H and W dims).
                        stride=4 reduces per-frame Gaussians by 16×.

    Returns:
        (means_world, cov_triu, colors, opacities) ready for SharedGaussians.append().
        means_world: (G, 3)
        cov_triu:    (G, 6)  upper-triangle of world-space 3×3 covariance
        colors:      (G, 3)  RGB in [0, 1]
        opacities:   (G,)
    """
    if frame.gaussian_pred is None:
        return None

    device = frame.gaussian_pred["means"].device
    T_WC_mat = _sim3_to_4x4(frame.T_WC)  # (1, 4, 4)
    R = T_WC_mat[0, :3, :3]  # (3, 3)
    t = T_WC_mat[0, :3, 3]  # (3,)

    preds = [frame.gaussian_pred]
    imgs = [frame.img]
    if include_cross and frame.gaussian_pred_cross is not None:
        preds.append(frame.gaussian_pred_cross)
        imgs.append(frame.img)  # same source image for SH residual

    all_means = []
    all_cov_triu = []
    all_colors = []
    all_opas = []

    row, col = torch.triu_indices(3, 3)
    s = max(1, int(spatial_stride))

    for pred, img_tensor in zip(preds, imgs):
        means = pred["means"][:, ::s, ::s, :]  # (B, H', W', 3)
        scales = pred["scales"][:, ::s, ::s, :]  # (B, H', W', 3)
        rotations = pred["rotations"][:, ::s, ::s, :]  # (B, H', W', 4)
        sh = pred["sh"][:, ::s, ::s, :, :]  # (B, H', W', 3, sh_degree)
        opas = pred["opacities"][:, ::s, ::s, :]  # (B, H', W', 1)

        # The downstream head outputs SH *residuals*; the original image
        # colour in SH space must be added to the DC component, matching
        # the logic in splatt3r_core/main.py:forward() (learn_residual).
        img_hwc = _get_original_img_hwc(img_tensor.to(means.device))  # (B, H, W, 3)
        img_hwc = img_hwc[:, ::s, ::s, :]  # subsample to match
        sh = sh.clone()
        sh[..., 0] = sh[..., 0] + RGB2SH(img_hwc)

        # Flatten spatial dims
        means_flat = means.reshape(-1, 3)  # (G, 3)
        scales_flat = scales.reshape(-1, 3)
        rots_flat = rotations.reshape(-1, 4)
        sh_flat = sh.reshape(-1, 3, sh.shape[-1])  # (G, 3, sh_degree)
        opas_flat = opas.reshape(-1)  # (G,)

        # Transform means to world: x_w = R @ x_c + t
        means_world = (R @ means_flat.T).T + t  # (G, 3)

        # Build covariance in camera space, then rotate to world
        cov_cam = build_covariance(scales_flat, rots_flat)  # (G, 3, 3)
        cov_world = R @ cov_cam @ R.T  # (G, 3, 3)
        cov_tri = cov_world[:, row, col]  # (G, 6)

        # Colour: SH zero-order → direct RGB via SH2RGB
        # Full SH = network_residual + RGB2SH(img), so
        # SH2RGB(sh0) = sh0 * C0 + 0.5  gives the final colour.
        sh0 = sh_flat[:, :, 0]  # (G, 3)
        C0 = 0.28209479177387814
        colors_rgb = (sh0 * C0 + 0.5).clamp(0, 1)  # (G, 3)

        all_means.append(means_world)
        all_cov_triu.append(cov_tri)
        all_colors.append(colors_rgb)
        all_opas.append(opas_flat)

    means_out = torch.cat(all_means, dim=0)
    cov_out = torch.cat(all_cov_triu, dim=0)
    colors_out = torch.cat(all_colors, dim=0)
    opas_out = torch.cat(all_opas, dim=0)

    return means_out, cov_out, colors_out, opas_out


@torch.inference_mode()
def splatt3r_render(model, frame, ref_frame, K=None, target_T_WC=None):
    """Render a target view via model.decoder (DecoderSplattingCUDA).

    Uses Gaussian predictions previously stored on *frame* by
    ``splatt3r_inference_mono`` or ``splatt3r_match_asymmetric``.

    Convention (asymmetric decode: view1=frame, view2=ref_frame):
      * gaussian_pred      – view1 self-prediction  (Gaussians in view1's frame)
      * gaussian_pred_cross – view2 cross-prediction (Gaussians in view1's frame)
      * context_pose = frame.T_WC  (view1's world pose)

    Args:
        model:        Splatt3R model whose ``.decoder`` is DecoderSplattingCUDA.
        frame:        Frame with gaussian_pred / gaussian_pred_cross populated.
        ref_frame:    The other frame in the pair (needed for SH residual image).
        K:            3×3 camera intrinsics tensor. *None* → estimated defaults.
        target_T_WC:  Target viewpoint (lietorch.Sim3).  *None* → same as
                      frame.T_WC (self-render / reconstruction quality check).

    Returns:
        Rendered colour image  (B, V=1, 3, H, W)  or *None* when Gaussians
        are unavailable.
    """
    if frame.gaussian_pred is None or frame.gaussian_pred_cross is None:
        print("[splatt3r_render] No Gaussian predictions available – skipping.")
        return None

    device = frame.gaussian_pred["means"].device
    _, h, w, _ = frame.gaussian_pred["means"].shape  # (B, H, W, 3)

    # ------------------------------------------------------------------
    # 1. Build covariance matrices  Σ = R S S^T R^T
    # ------------------------------------------------------------------
    cov1 = build_covariance(
        frame.gaussian_pred["scales"], frame.gaussian_pred["rotations"]
    )
    cov2 = build_covariance(
        frame.gaussian_pred_cross["scales"], frame.gaussian_pred_cross["rotations"]
    )

    # ------------------------------------------------------------------
    # 2. SH coefficients with RGB residual  (zero-order band)
    # ------------------------------------------------------------------
    img1_hwc = _get_original_img_hwc(frame.img.to(device))
    img2_hwc = _get_original_img_hwc(ref_frame.img.to(device))

    sh1 = frame.gaussian_pred["sh"].clone()
    sh_res1 = torch.zeros_like(sh1)
    sh_res1[..., 0] = RGB2SH(img1_hwc)
    sh1 = sh1 + sh_res1

    sh2 = frame.gaussian_pred_cross["sh"].clone()
    sh_res2 = torch.zeros_like(sh2)
    sh_res2[..., 0] = RGB2SH(img2_hwc)
    sh2 = sh2 + sh_res2

    # ------------------------------------------------------------------
    # 3. pred dicts in DecoderSplattingCUDA format
    # ------------------------------------------------------------------
    pred1 = {
        "means": frame.gaussian_pred["means"],
        "covariances": cov1,
        "sh": sh1,
        "opacities": frame.gaussian_pred["opacities"],
    }
    pred2 = {
        "means_in_other_view": frame.gaussian_pred_cross["means"],
        "covariances": cov2,
        "sh": sh2,
        "opacities": frame.gaussian_pred_cross["opacities"],
    }

    # ------------------------------------------------------------------
    # 4. Camera poses  (Sim3 → 4×4)
    # ------------------------------------------------------------------
    context_pose = _sim3_to_4x4(frame.T_WC)  # (1, 4, 4)
    if target_T_WC is None:
        target_pose = context_pose.clone()  # self-render
    else:
        target_pose = _sim3_to_4x4(target_T_WC)

    # ------------------------------------------------------------------
    # 5. Intrinsics
    # ------------------------------------------------------------------
    if K is None:
        K_use = _estimate_default_intrinsics(h, w, device)
    else:
        K_use = K.clone().to(device=device, dtype=torch.float32)
    if K_use.dim() == 2:
        K_use = K_use.unsqueeze(0)  # (1, 3, 3)

    # ------------------------------------------------------------------
    # 6. Construct batch & call model.decoder (DecoderSplattingCUDA)
    # ------------------------------------------------------------------
    batch = {
        "context": [{"camera_pose": context_pose}],
        "target": [{"camera_pose": target_pose, "camera_intrinsics": K_use}],
    }

    color, _ = model.decoder(batch, pred1, pred2, (h, w))
    return color  # (B, V=1, C=3, H, W)


@torch.inference_mode()
def splatt3r_symmetric_inference(model, frame_i, frame_j):
    """
    Symmetric inference using Splatt3R model.
    Predicts 3D points and Gaussian parameters for both views.
    """
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model.encoder._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model.encoder._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode()
def splatt3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    """Batch symmetric decoding for Splatt3R"""
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode()
def splatt3r_inference_mono(model, frame):
    """
    Monocular inference using Splatt3R.
    Predicts 3D points and Gaussian parameters from a single view.
    Gaussian params are stored on frame.gaussian_pred / frame.gaussian_pred_cross
    for subsequent rendering via splatt3r_render().
    """
    if frame.feat is None:
        frame.feat, frame.pos, _ = model.encoder._encode_image(
            frame.img, frame.img_true_shape
        )

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)

    # --- Store Gaussian predictions on frame for later rendering ---
    frame.gaussian_pred = _extract_gaussian_params(res11)
    frame.gaussian_pred_cross = _extract_gaussian_params(res21)

    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


def splatt3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    """Match features using Splatt3R"""
    X, C, D, Q = splatt3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


@torch.inference_mode()
def splatt3r_asymmetric_inference(model, frame_i, frame_j):
    """Asymmetric inference using Splatt3R.

    Also extracts Gaussian params and returns raw decoder result dicts
    so the caller can store them for Gaussian Splatting rendering.
    """
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model.encoder._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model.encoder._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q, (res11, res21)


def splatt3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    """Asymmetric matching using Splatt3R.

    Side-effect: stores Gaussian predictions on *frame_i* so that
    ``splatt3r_render(model, frame_i, frame_j, ...)`` can render
    novel views via Gaussian Splatting.
    """
    X, C, D, Q, (res_self, res_cross) = splatt3r_asymmetric_inference(
        model, frame_i, frame_j
    )

    # Store Gaussian predictions on frame_i (view1) for later rendering
    frame_i.gaussian_pred = _extract_gaussian_params(res_self)
    frame_i.gaussian_pred_cross = _extract_gaussian_params(res_cross)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # How rest of system expects it
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji


def _resize_pil_image(img, long_edge_size):
    """Resize PIL image"""
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, square_ok=False, return_transformation=False):
    """Resize image to specified size"""
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res

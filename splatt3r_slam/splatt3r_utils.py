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


def load_splatt3r(path=None, device="cuda", lora_path=None, head_path=None):
    """
    Load Splatt3R model (with Gaussian splatting capabilities).

    Args:
        path: Path to checkpoint. If None, checks checkpoints/ dir first,
              then downloads from HuggingFace.
        device: Device to load model on.
        head_path: Optional path to a fine-tuned Gaussian-head state_dict
              (`head_best.pt`, as saved by scripts/exp_head_only.py). This is
              the route that was measured to actually beat the released
              checkpoint (+1.00 dB psnr / -0.0173 lpips on a fixed 150-sample
              draw); see the splatt3r-finetuning-experiments skill. Only
              `gaussian_dpt` tensors are replaced -- the encoder and decoder
              are the released weights, untouched, which is why the retrieval
              database and every other consumer of encoder features stays
              valid without a refit.
        lora_path: Optional path to a trained LoRA adapter directory (e.g.
              checkpoints/lora/<scene>/, as saved by peft's
              save_pretrained() -- see splatt3r_core/lora.py and the
              splatt3r-lora-finetuning skill). If given, the adapter is
              loaded and kept LIVE/unmerged (peft.PeftModel wrapping
              model.encoder) rather than baked into the base weights --
              this is the "separately loaded, hot-swappable" design:
              swapping scenes means loading a different few-MB adapter
              directory, not a new multi-GB checkpoint file, and the base
              weights on disk are never duplicated per scene. The small
              per-forward-pass LoRA overhead this trades away is
              negligible next to a SLAM frame's other costs.

    Returns:
        Splatt3R model (MAST3RGaussians), with model.encoder possibly
        wrapped as a peft.PeftModel if lora_path was given.

    head_path and lora_path are mutually exclusive. The encoder-LoRA route was
    measured to be 49% WORSE than the released checkpoint and is retained only
    to reproduce that negative result; head_path is the one that works.
    """
    if head_path is not None and lora_path is not None:
        raise ValueError("head_path and lora_path are mutually exclusive")

    if path is None:
        # Check local checkpoints directory first
        local_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints",
            "epoch=19-step=1200.ckpt",
        )
        if os.path.exists(local_path):
            print(f"Found local Splatt3R checkpoint: {local_path}")
            weights_path = local_path
        else:
            from huggingface_hub import hf_hub_download

            model_name = "brandonsmart/splatt3r_v1.0"
            filename = "epoch=19-step=1200.ckpt"
            print(f"Local checkpoint not found, downloading from {model_name}")
            weights_path = hf_hub_download(repo_id=model_name, filename=filename)
    else:
        weights_path = path

    print(f"Loading Splatt3R model from {weights_path}")
    model = MAST3RGaussians.load_from_checkpoint(weights_path, device)
    model.eval()

    if head_path is not None:
        print(f"Loading fine-tuned Gaussian head from {head_path}")
        state = torch.load(head_path, map_location=device)
        missing, unexpected = model.encoder.load_state_dict(state, strict=False)
        # strict=False is required (the file holds only the head), so verify
        # explicitly rather than trusting a silent no-op: every tensor in the
        # file must land, and nothing outside the head may be reported missing.
        if unexpected:
            raise RuntimeError(
                f"{head_path} has {len(unexpected)} tensors the encoder does "
                f"not accept, e.g. {unexpected[:3]}")
        head_missing = [k for k in missing if "gaussian_dpt" in k]
        if head_missing:
            raise RuntimeError(
                f"{len(head_missing)} Gaussian-head tensors failed to load "
                f"from {head_path}, e.g. {head_missing[:3]}")
        print(f"  replaced {len(state)} head tensors "
              f"(encoder/decoder left at released weights)")
        model.eval()

    if lora_path is not None:
        # Lazy import: peft is only a runtime dependency when --lora is
        # actually used, not for normal (no-LoRA) SLAM runs.
        from peft import PeftModel

        print(f"Loading LoRA adapter from {lora_path} (kept live, not merged)")
        model.encoder = PeftModel.from_pretrained(model.encoder, lora_path)
        model.encoder.eval()

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
          sh (B,H,W,3,sh_degree), opacities (B,H,W,1),
          conf (B,H,W) – pointmap confidence for quality filtering.
    """
    d = {
        "means": res["means"].clone(),
        "scales": res["scales"].clone(),
        "rotations": res["rotations"].clone(),
        "sh": res["sh"].clone(),
        "opacities": res["opacities"].clone(),
    }
    if "conf" in res:
        d["conf"] = res["conf"].clone()  # (B, H, W)
    return d


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
def prepare_gaussians_local(
    local: dict,
    img_tensor: torch.Tensor,
    h: int,
    w: int,
    spatial_stride=1,
    depth_min=0.05,
    depth_max_percentile=0.98,
    max_scale=0.5,
    min_confidence=1.5,
    min_opacity=0.3,
    inflate_scales_for_stride=True,
    aa_sigma_scale=0.0,
    aa_compensate_opacity=False,
    max_anisotropy=0.0,
    streak_opacity=0.0,
    return_pitch=False,
    return_conf=False,
):
    """Transform ONE keyframe's camera-space Gaussians to world space.

    Unlike the old ``gaussians_to_world`` (removed), this never caches or
    stores its output anywhere. It is meant to be called fresh, every time
    a keyframe needs to be drawn, using that keyframe's *current* T_WC
    (``SharedKeyframes.T_WC[idx]``, kept live by the pose-graph backend).
    That means a loop-closure / bundle-adjustment correction is reflected
    the very next time the keyframe is baked -- there is no permanently
    mis-placed copy left behind in world space (see the splatt3r-gaussian-map
    skill for the failure mode this replaces).

    Filters out low-quality "splash" Gaussians caused by occluded / unseen
    regions where the model hallucinates noisy predictions:

    1. **Depth filter** – keeps Gaussians with camera-space z in
       [depth_min, percentile(z, depth_max_percentile)].
    2. **Scale filter** – keeps Gaussians whose max scale axis < *max_scale*.
    3. **Confidence filter** – keeps Gaussians whose per-pixel pointmap
       confidence ≥ *min_confidence* (if conf is available).

    Args:
        local: dict from SharedKeyframes.get_gaussians_local(idx) — flat
            (h*w, C) camera-space means/scales/rotations/sh/opacities/conf.
        img_tensor: this keyframe's normalised (C, H, W) or (1, C, H, W)
            image tensor (SharedKeyframes.img[idx]), used to recover the
            SH DC colour residual. Must match the (h, w) grid `local` was
            predicted at.
        h, w: spatial resolution of the Gaussian prediction grid.
        spatial_stride: subsample stride in H and W. stride=4 reduces the
            Gaussian count by 16x.
        depth_min: minimum camera-space depth to keep (metres).
        depth_max_percentile: percentile of camera-space depth used as
            the upper bound.
        max_scale: maximum allowed Gaussian scale (any axis). Larger -> splash.
        min_confidence: minimum pointmap confidence. Lower -> hallucinated.
        min_opacity: minimum opacity (after the SH/opacity decode) to keep.
            Lower -> translucent noise.
        inflate_scales_for_stride: if True (default), multiply scales by
            `spatial_stride` so subsampled splats still visually touch in the
            viewport. This is a viz-only compensation -- exporters writing a
            persisted .ply must pass False, or splats come out up to
            `spatial_stride` times too large.

    Returns:
        (means_world, cov_triu, colors, opacities), or None if nothing
        survives filtering.
        means_world: (G, 3)
        cov_triu:    (G, 6)  upper-triangle of world-space 3x3 covariance
        colors:      (G, 3)  RGB in [0, 1]
        opacities:   (G,)
    """
    device = local["means"].device
    s = max(1, int(spatial_stride))

    sh_dim = local["sh"].shape[-1]
    means_grid = local["means"].view(h, w, 3)[::s, ::s]
    means = means_grid.reshape(-1, 3)
    scales = local["scales"].view(h, w, 3)[::s, ::s].reshape(-1, 3)
    rotations = local["rotations"].view(h, w, 4)[::s, ::s].reshape(-1, 4)
    sh = local["sh"].view(h, w, 3, sh_dim)[::s, ::s].reshape(-1, 3, sh_dim)
    opas = local["opacities"].view(h, w)[::s, ::s].reshape(-1)
    conf = local.get("conf")
    conf = conf.view(h, w)[::s, ::s].reshape(-1) if conf is not None else None

    # The downstream head outputs SH *residuals*; the original image
    # colour in SH space must be added to the DC component, matching the
    # logic in splatt3r_core/main.py:forward() (learn_residual).
    img_hwc = _get_original_img_hwc(img_tensor.to(device))  # (1, H, W, 3)
    img_hwc = img_hwc[:, ::s, ::s, :].reshape(-1, 3)
    sh = sh.clone()
    sh[..., 0] = sh[..., 0] + RGB2SH(img_hwc)

    # ---- Quality filtering (camera space, before world transform) ----
    z = means[:, 2]  # camera-space depth
    valid = z > depth_min
    if valid.any() and depth_max_percentile < 1.0:
        z_valid = z[valid]
        z_upper = torch.quantile(z_valid, depth_max_percentile)
        valid = valid & (z <= z_upper)

    scale_max = scales.max(dim=-1).values
    valid = valid & (scale_max < max_scale)

    # Lattice pitch: the 3D distance from each Gaussian to its nearest
    # 4-neighbour on the SOURCE PIXEL GRID. Splatt3R emits one Gaussian per
    # pixel and nothing here densifies, so this spacing is fixed at injection
    # and is exactly the sampling rate the map can represent -- there is no
    # need for Mip-Splatting's max-over-training-views proxy, the quantity it
    # estimates is directly measurable here.
    #
    # MIN over the neighbours, not mean: across a depth discontinuity the
    # neighbour is metres away, and a mean would inflate every silhouette
    # Gaussian into a blob spanning the gap -- the "trailing streak" artifact,
    # manufactured on purpose. The min takes the in-surface spacing.
    if aa_sigma_scale > 0 or return_pitch:
        hs, ws = means_grid.shape[0], means_grid.shape[1]
        pitch = torch.full((hs, ws), float("inf"), device=device)
        if ws > 1:
            dx = (means_grid[:, 1:] - means_grid[:, :-1]).norm(dim=-1)
            pitch[:, :-1] = torch.minimum(pitch[:, :-1], dx)
            pitch[:, 1:] = torch.minimum(pitch[:, 1:], dx)
        if hs > 1:
            dy = (means_grid[1:, :] - means_grid[:-1, :]).norm(dim=-1)
            pitch[:-1, :] = torch.minimum(pitch[:-1, :], dy)
            pitch[1:, :] = torch.minimum(pitch[1:, :], dy)
        pitch = pitch.reshape(-1)
        pitch = torch.where(torch.isfinite(pitch), pitch,
                            torch.zeros_like(pitch))
    else:
        pitch = None

    if conf is not None and min_confidence > 0:
        valid = valid & (conf >= min_confidence)

    if min_opacity > 0:
        valid = valid & (opas > min_opacity)

    means = means[valid]
    scales = scales[valid]
    rotations = rotations[valid]
    sh = sh[valid]
    opas = opas[valid]
    if pitch is not None:
        pitch = pitch[valid]

    if means.shape[0] == 0:
        return None

    # 3D smoothing filter (Mip-Splatting, Yu et al. CVPR 2024), with the
    # band limit read off the lattice above instead of estimated from the
    # cameras. Adding sigma^2 I to the covariance is exact in the
    # scale/rotation parameterization: R diag(s^2) R^T + sigma^2 I
    # = R (diag(s^2) + sigma^2 I) R^T, since R R^T = I. So the rotation is
    # untouched and only the scales move.
    #
    # What this fixes: the network sizes each Gaussian to its own pixel
    # footprint at the source view, which leaves the surface just barely
    # covered there and visibly perforated from anywhere else -- the regular
    # dot lattice and the moire that beats against the output pixel grid.
    # No number of optimizer steps closes those gaps, because the supervision
    # views are at the source sampling rate where the holes are invisible.
    #
    # sigma = aa_sigma_scale * pitch. At the midpoint between two neighbours
    # the alpha ratio is exp(-0.5 (pitch/2 sigma)^2), so aa_sigma_scale = 0.5
    # puts that midpoint at 1 sigma and is the natural starting point.
    # Anisotropy clamp: shrink the long axis to at most `max_anisotropy` times
    # the short one. This is the trailing-streak artifact -- at a depth
    # discontinuity the two-view match is uncertain ALONG THE RAY and the head
    # emits a needle pointing at the camera, which reads as a smeared comet
    # tail from any other view. The existing max_scale=0.5 filter only catches
    # the absolute size, so a 2 cm x 2 cm x 40 cm needle passes it untouched.
    #
    # Clamped, not deleted: dropping them would leave holes exactly at object
    # silhouettes, which is where holes are most visible. Applied BEFORE the
    # band limit below, so the floor still guarantees coverage afterwards.
    # Computed unconditionally: the streak block below falls back to it when no
    # lattice pitch is available, and gating it on max_anisotropy made
    # --streak-opacity crash on every online run (aa_sigma defaults to 0 there,
    # so pitch is None) while every offline test passed, because those always ran
    # with the band limit on. A latent path the proxy never exercised.
    s_min = scales.min(dim=-1, keepdim=True).values
    if max_anisotropy > 0:
        scales = torch.minimum(scales, s_min * max_anisotropy)

    # Ray-elongated Gaussians, hidden rather than shrunk. Clamping the long axis
    # was measured harmful at every setting (17.16): it shrinks the footprint,
    # and the band limit only floors the SMALLEST scale, so it opens holes.
    # Lowering opacity instead leaves the footprint intact and lets whatever is
    # behind show through -- which on these sequences is almost always another
    # keyframe's correct surface, since a held-out viewpoint sits 0.057 m from
    # the nearest keyframe (17.16). Where nothing is behind, this trades a
    # streak for a small hole, which is the risk to watch in the black fraction.
    if streak_opacity > 0:
        s_max = scales.max(dim=-1).values
        if pitch is None and aa_sigma_scale <= 0:
            pass  # handled below with a clear error
        # Elongation measured against the lattice pitch, not against the other
        # axes: a Gaussian is a streak when it is long compared with the surface
        # sampling it belongs to, which is the quantity 17.2 established.
        if pitch is None:
            # The criterion is "long relative to the surface sampling it belongs
            # to" (17.32) and the lattice pitch IS that quantity. Falling back to
            # s_min silently turns it into the anisotropy ratio -- the thing
            # 17.16 measured as harmful -- and crushes opacity ~15x globally
            # (measured 0.07 mean on the first online attempt). Refuse rather
            # than approximate.
            raise ValueError(
                "streak_opacity needs the lattice pitch: pass aa_sigma_scale > 0 "
                "(or return_pitch) so it can be computed. Falling back to the "
                "anisotropy ratio is a different, measured-harmful lever.")
        opas = opas * torch.clamp(streak_opacity * pitch / s_max.clamp_min(1e-9),
                                  max=1.0)

    if pitch is not None and aa_sigma_scale > 0:
        sigma = aa_sigma_scale * pitch
        s_new = torch.sqrt(scales ** 2 + (sigma ** 2)[:, None])
        if aa_compensate_opacity:
            # Mip-Splatting's energy correction: the convolution lowers the
            # peak, and sqrt(|Sigma| / |Sigma'|) restores the integral. Off by
            # default because the perforation IS an alpha deficit at the
            # midpoints, and compensating gives back exactly what closes it.
            opas = opas * (scales.prod(-1) / s_new.prod(-1).clamp_min(1e-12))
        scales = s_new

    # Compensate for spatial subsampling: dropping every s-th pixel
    # increases the spacing between *retained* Gaussians by ~s in each
    # image-plane direction, but their footprint (~scale) doesn't grow to
    # match -- neighbouring splats stop overlapping and the surface reads
    # as a stippled/halftone dot pattern rather than continuous coverage
    # (worst at gs_resolution_scale=1.0, where nothing softens it).
    # Inflate scale linearly with stride so splats keep touching. This is
    # applied *after* the max_scale splash-artifact filter above, which
    # is meant to catch the network's own hallucinated large predictions
    # and should stay keyed to the raw per-pixel scale, not this display
    # compensation.
    #
    # Clamp the *inflated* result back to max_scale. The network's own
    # predicted scale already grows with depth/uncertainty (a pixel
    # covers more physical area at range, and the network is also
    # genuinely less confident about far geometry from two-view
    # matching) -- far-field points already start closer to max_scale
    # than near-field ones. Multiplying everything by a flat s
    # compounds that, so distant surfaces get disproportionately blurry
    # relative to near ones. Clamping means the gap-closing compensation
    # can't push a splat past the same ceiling we already use to decide
    # "this is as big as a legitimate splat should ever get" -- near-
    # field (small, well below the ceiling) still gets the full s-times
    # inflation it needs; far-field (already close to the ceiling)
    # stops growing once it hits it.
    #
    # inflate_scales_for_stride=False turns this off for callers that are
    # persisting the Gaussians rather than drawing them (evaluate.py's
    # save_gaussian_map): the inflation is a function of the *display*
    # subsampling factor, not of the scene, so baking it into an exported
    # .ply would hand an external viewer splats that are up to s times too
    # large.
    if s > 1 and inflate_scales_for_stride:
        scales = torch.clamp(scales * s, max=max_scale)

    # Colour: SH zero-order -> direct RGB via SH2RGB. Full SH =
    # network_residual + RGB2SH(img), so SH2RGB(sh0) = sh0 * C0 + 0.5.
    sh0 = sh[:, :, 0]
    colors_rgb = (sh0 * 0.28209479177387814 + 0.5).clamp(0, 1)

    # build_covariance()'s quaternion->matrix conversion divides by
    # (||q||^2 + eps) rather than normalising, so a near-zero-norm quaternion
    # (which the network does occasionally emit at low-signal pixels) becomes a
    # huge non-orthonormal "rotation" -- historically the trigger for the CUDA
    # rasterizer's asynchronous "illegal memory access". Normalise here so the
    # conversion is a true rotation regardless of the raw magnitude.
    rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    out = (means, scales, rotations, colors_rgb, opas)
    if return_pitch:
        # The lattice pitch, for callers that want to hold the band limit
        # SEPARATELY from the learned scale rather than baked into it (the
        # refiner: see LocalGaussianMap's scale_floor). Zero when the filter
        # is off, which makes the floor a no-op there.
        out = out + (pitch if pitch is not None else torch.zeros_like(opas),)
    if return_conf:
        # Surviving per-Gaussian backbone confidence, for callers conditioning
        # a prior on it (17.66.4's confidence-weighted fade). It has to be
        # returned from here rather than recomputed by the caller because the
        # `valid` mask above is what decides which Gaussians exist -- a caller
        # re-deriving it would be reimplementing four filters and would drift
        # from them silently. Ones when conf is unavailable, so the weight
        # degenerates to the uniform prior rather than to zero.
        out = out + (conf[valid] if conf is not None else torch.ones_like(opas),)
    return out


def bake_gaussians_world(
    local: dict,
    img_tensor: torch.Tensor,
    h: int,
    w: int,
    T_WC,
    spatial_stride=1,
    depth_min=0.05,
    depth_max_percentile=0.98,
    max_scale=0.5,
    min_confidence=1.5,
    min_opacity=0.3,
    inflate_scales_for_stride=True,
    aa_sigma_scale=0.0,
    aa_compensate_opacity=False,
    max_anisotropy=0.0,
    streak_opacity=0.0,
):
    """Place ONE keyframe's Gaussians in world space using its CURRENT pose.

    Camera-space preparation (subsampling, quality filtering, SH residual +
    image colour, quaternion normalisation) lives in
    ``prepare_gaussians_local``; this only applies the rigid placement. The
    split exists so the online refiner (``splatt3r_slam/refiner.py``)
    initialises from byte-identical camera-space Gaussians -- two
    preprocessing paths that drift apart is a failure mode this project has
    already paid for more than once.

    Nothing is cached. Called fresh every time a keyframe is drawn, with that
    keyframe's live ``SharedKeyframes.T_WC[idx]``, so a loop-closure
    correction is reflected on the next bake and no mis-placed copy is left
    behind (see the splatt3r-gaussian-map skill).

    Returns (means_world (G,3), cov_triu (G,6), colors (G,3), opacities (G,))
    or None if nothing survives filtering.
    """
    prepared = prepare_gaussians_local(
        local, img_tensor, h, w,
        spatial_stride=spatial_stride,
        depth_min=depth_min,
        depth_max_percentile=depth_max_percentile,
        max_scale=max_scale,
        min_confidence=min_confidence,
        min_opacity=min_opacity,
        inflate_scales_for_stride=inflate_scales_for_stride,
        aa_sigma_scale=aa_sigma_scale,
        aa_compensate_opacity=aa_compensate_opacity,
        max_anisotropy=max_anisotropy,
        streak_opacity=streak_opacity,
    )
    if prepared is None:
        return None
    means, scales, rotations, colors_rgb, opas = prepared

    T_WC_mat = _sim3_to_4x4(T_WC)  # (1, 4, 4); Sim3 scale folded into sR
    R = T_WC_mat[0, :3, :3]
    t = T_WC_mat[0, :3, 3]
    row, col = torch.triu_indices(3, 3)

    means_world = (R @ means.T).T + t
    cov_world = R @ build_covariance(scales, rotations) @ R.T
    cov_tri = cov_world[:, row, col]

    # Defensive final filter: never hand the rasterizer a non-finite input
    # (NaN means/sh straight from the network).
    finite = torch.isfinite(means_world).all(dim=-1) & torch.isfinite(cov_tri).all(dim=-1)
    if not bool(finite.all()):
        means_world = means_world[finite]
        cov_tri = cov_tri[finite]
        colors_rgb = colors_rgb[finite]
        opas = opas[finite]

    if means_world.shape[0] == 0:
        return None

    return means_world, cov_tri, colors_rgb, opas


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
    # Normalize the quaternions first, exactly as bake_gaussians_world()
    # does further down. build_covariance()'s quaternion_to_matrix() only
    # guards against an exact divide-by-zero (eps=1e-8), not against a
    # near-zero-norm quaternion, which yields a huge non-orthonormal
    # "rotation" and hence a degenerate covariance -- the documented
    # trigger for the CUDA rasterizer's illegal-memory-access crash. This
    # path fed raw predictions straight in, so the defence existed on only
    # one of the two paths that reach the same rasterizer.
    def _unit_quat(q):
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    cov1 = build_covariance(
        frame.gaussian_pred["scales"], _unit_quat(frame.gaussian_pred["rotations"])
    )
    cov2 = build_covariance(
        frame.gaussian_pred_cross["scales"],
        _unit_quat(frame.gaussian_pred_cross["rotations"]),
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

"""Self-predicted pseudo-depth, for datasets with no real depth sensor
(EuRoC: stereo without depth; ETH3D SLAM benchmark subset: monocular,
no depth at all).

IMPORTANT caveat (see the splatt3r-lora-finetuning skill): this
supervises training with the *base model's own* depth estimate, not
independent ground truth. It is not circular in the sense of "training
on its own LoRA output" (pseudo-depth is precomputed once from the
frozen base checkpoint, before any LoRA training starts, and never
updated during training), but it does mean training can only push the
model to be more SELF-CONSISTENT with its own existing geometry
estimate for these two families, not more ACCURATE against real-world
geometry the way TUM/7-Scenes' real sensor depth allows. Confidence-
threshold the output before trusting it.
"""
import cv2
import numpy as np
import torch
from dust3r.utils.image import ImgNorm


def _resize_long_side_multiple_of_16(img_hwc_uint8, long_side=512):
    h, w = img_hwc_uint8.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    new_w = max(16, (new_w // 16) * 16)
    new_h = max(16, (new_h // 16) * 16)
    return cv2.resize(img_hwc_uint8, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


@torch.inference_mode()
def predict_pseudo_depth(model, img_hwc_uint8, device, min_confidence=1.5):
    """Run the (base, non-LoRA) model's own mono self-prediction to get a
    depth map and confidence-based validity mask, resized back to the
    input image's native (H, W) so it aligns pixel-for-pixel with the
    original RGB frame -- matching what a real depth sensor's output
    would look like to the rest of this data pipeline (crop_resize_
    necessary et al, called downstream in data/euroc/euroc.py and
    data/eth3d/eth3d.py, expect native-resolution depth+image pairs).

    Args:
        model: loaded MAST3RGaussians instance (plain base checkpoint,
            not LoRA-attached -- see scripts/precompute_pseudo_depth.py).
        img_hwc_uint8: (H, W, 3) uint8 RGB, native resolution.
        min_confidence: pixels below this predicted confidence are
            marked invalid (same default as
            splatt3r_slam/splatt3r_utils.py: bake_gaussians_world).

    Returns:
        depthmap: (H, W) float32. Splatt3R's own predicted metric scale
            -- not tied to any real-world unit for these two families,
            since there's no ground truth to anchor it to. Zeroed out
            wherever valid_mask is False.
        valid_mask: (H, W) bool.
    """
    h0, w0 = img_hwc_uint8.shape[:2]
    small = _resize_long_side_multiple_of_16(img_hwc_uint8, 512)
    hs, ws = small.shape[:2]

    img_t = ImgNorm(small).unsqueeze(0).to(device)
    true_shape = torch.tensor([[hs, ws]], device=device)
    view = {"img": img_t, "true_shape": true_shape, "instance": ["0"], "idx": [0]}

    encoder = model.encoder
    base = encoder.get_base_model() if hasattr(encoder, "get_base_model") else encoder

    (shape1, _), (feat1, _), (pos1, _) = base._encode_symmetrized(view, view)
    dec1, _ = base._decoder(feat1, pos1, feat1, pos1)
    pred1 = base._downstream_head(1, [tok.float() for tok in dec1], shape1)

    means = pred1["means"][0]  # (hs, ws, 3), camera-space
    depth_small = means[..., 2].detach().float().cpu().numpy()
    depth_full = cv2.resize(depth_small, (w0, h0), interpolation=cv2.INTER_LINEAR)

    if "conf" in pred1:
        conf_small = pred1["conf"][0].detach().float().cpu().numpy()
        conf_full = cv2.resize(conf_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
        valid_full = (conf_full >= min_confidence) & (depth_full > 0)
    else:
        valid_full = depth_full > 0

    depth_full = np.where(valid_full, depth_full, 0.0).astype(np.float32)
    return depth_full, valid_full

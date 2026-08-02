import numpy as np
import torch
import torch.nn.functional as F

# Locked to the first frame's per-channel mean the first time
# normalize_exposure() is called; None until then. Module-level because
# create_frame() is called once per incoming frame with no other place to
# carry this across calls, and this pipeline only ever processes one
# sequence per process (see splatt3r-color-consistency skill).
_exposure_reference_mean = None


def reset_exposure_reference():
    """Forget the locked reference mean, so the next normalize_exposure()
    call re-anchors to whatever frame it sees next. Call this if you ever
    process more than one sequence within a single process."""
    global _exposure_reference_mean
    _exposure_reference_mean = None


def normalize_exposure(img, min_gain=0.4, max_gain=2.5, eps=1e-3):
    """Causally correct per-frame exposure/white-balance drift.

    Splatt3R bakes each frame's raw pixel colour directly into that
    frame's Gaussians (SH DC term = network residual + RGB2SH(raw image),
    see splatt3r_utils.bake_gaussians_world). Nothing downstream ever
    reconciles colour between two independent observations of the same
    physical surface, so if the source video's auto-exposure/white
    balance drifts over the sequence, revisiting an area later bakes in a
    visibly different-coloured copy on top of the earlier one (see the
    splatt3r-color-consistency skill for the full diagnosis and
    alternative fixes).

    This rescales each frame's per-channel mean to match the *first*
    frame seen (locked in module state, no look-ahead -- safe to call
    inline in the online SLAM loop). Gain is clamped to [min_gain,
    max_gain] so a frame that's briefly very dark/bright (e.g. pointed at
    a shadow) doesn't get wildly over-corrected.

    Args:
        img: (H, W, 3) float array in [0, 1].
    Returns:
        (H, W, 3) float array in [0, 1], same dtype as input.
    """
    global _exposure_reference_mean

    mean = img.reshape(-1, img.shape[-1]).mean(axis=0)

    if _exposure_reference_mean is None:
        _exposure_reference_mean = mean
        return img

    gain = (_exposure_reference_mean + eps) / (mean + eps)
    gain = np.clip(gain, min_gain, max_gain)
    return np.clip(img * gain, 0.0, 1.0).astype(img.dtype)


def img_gradient(img):
    device = img.device
    dtype = img.dtype
    b, c, h, w = img.shape

    gx_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gx_kernel = gx_kernel.repeat(c, 1, 1, 1)

    gy_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gy_kernel = gy_kernel.repeat(c, 1, 1, 1)

    gx = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gx_kernel,
        groups=img.shape[1],
    )

    gy = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gy_kernel,
        groups=img.shape[1],
    )

    return gx, gy

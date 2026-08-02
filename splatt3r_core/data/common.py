"""Shared low-level parsing utilities for the TUM/7-Scenes/EuRoC/ETH3D data
adapters in this directory. See the splatt3r-lora-finetuning skill for the
overall design (why these exist, what's real ground-truth depth vs.
self-predicted pseudo-depth, per-family gotchas).
"""
import bisect

import numpy as np

# Training-side exposure normalization. The SLAM deployment side rescales
# every incoming frame's exposure/white balance against the sequence's
# first frame (splatt3r_slam/image.py: normalize_exposure, used from
# splatt3r_slam/frame.py) because Splatt3R bakes raw pixel colour into
# each frame's Gaussians and nothing downstream reconciles exposure drift.
# The training data pipeline had NO such handling, so the model trained on
# raw auto-exposure video while deployment feeds it normalized video -- a
# train/inference distribution mismatch the failed LoRA experiment had to
# spend capacity on. This switch turns the training-side counterpart
# on/off for every family's Data adapter below.
NORMALIZE_EXPOSURE = True


class SequenceExposureLock:
    """One locked exposure gain per sequence, computed from that
    sequence's FIRST frame and applied identically to all of its frames.

    Deliberately NOT the per-frame normalization splatt3r_slam/image.py:
    normalize_exposure() does online (equalizing every frame to the first
    frame's mean): per-frame equalization would erase the cross-frame
    photometric variation the multi-view consistency signal is supposed to
    learn from -- instead we canonicalize each sequence's overall
    exposure/white balance with a single gain while preserving the
    sequence's internal photometric relationships. The gain is computed
    once, from the first frame of the sequence (as stored in the Data
    object's train/val slice), mapping its per-channel mean towards
    `target_mean`, clamped to [min_gain, max_gain] -- the same clamp range
    the deployment side uses. Locking happens eagerly in each Data
    adapter's __init__ (see lock_from_first_frame below), so the gain is
    deterministic and identical across DDP ranks and DataLoader workers
    (unlike a lazy "first frame seen" lock, which would depend on random
    sampling order).

    Images are uint8 (H, W, 3) at this point in the pipeline -- apply()
    returns uint8 so the downstream PIL-based crop/resize in
    data/data.py: crop_resize_if_necessary is unaffected. Applied BEFORE
    that crop/resize, matching the deployment side's order (exposure
    before resize).
    """

    def __init__(self, target_mean=0.5, min_gain=0.4, max_gain=2.5, eps=1e-3):
        self.target_mean = target_mean
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.eps = eps
        self._gains = {}

    def lock(self, sequence, first_frame):
        """Compute and lock this sequence's per-channel gain from its
        first frame (uint8 (H, W, 3)). Returns the locked gain."""
        img = first_frame.astype(np.float32) / 255.0
        mean = img.reshape(-1, img.shape[-1]).mean(axis=0)
        gain = (self.target_mean + self.eps) / (mean + self.eps)
        gain = np.clip(gain, self.min_gain, self.max_gain).astype(np.float32)
        self._gains[sequence] = gain
        return gain

    def gain(self, sequence):
        return self._gains[sequence]

    def apply(self, img, sequence):
        """Apply the sequence's locked gain. img: uint8 (H, W, 3) ->
        uint8, values re-clipped to valid range."""
        if sequence not in self._gains:
            raise KeyError(
                f"No locked exposure gain for sequence '{sequence}' -- "
                f"lock() must run first (Data adapter __init__ does this)"
            )
        out = np.clip(img.astype(np.float32) / 255.0 * self._gains[sequence], 0.0, 1.0)
        return (out * 255.0).round().astype(img.dtype)


def read_file_list(path, delimiter=None):
    """Parse a 'timestamp value0 [value1 ...]' file, '#'-comment lines
    skipped. delimiter=None splits on any whitespace (TUM/ETH3D style);
    pass ',' for EuRoC's CSV files. Returns {timestamp: [values...]}.
    """
    result = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter)
            parts = [p.strip() for p in parts]
            result[float(parts[0])] = parts[1:]
    return result


def associate(first_list, second_list, max_time_diff=0.02):
    """Standard TUM nearest-timestamp association. Returns a list of
    (first_timestamp, second_timestamp) pairs, each within max_time_diff
    seconds, each timestamp used at most once, sorted by first_timestamp.
    """
    first_keys = sorted(first_list.keys())
    second_keys = sorted(second_list.keys())
    potential = []
    for t1 in first_keys:
        i = bisect.bisect_left(second_keys, t1)
        for j in (i - 1, i):
            if 0 <= j < len(second_keys):
                t2 = second_keys[j]
                potential.append((abs(t1 - t2), t1, t2))
    potential.sort()
    matches = []
    used_first, used_second = set(), set()
    for diff, t1, t2 in potential:
        if diff > max_time_diff:
            break
        if t1 in used_first or t2 in used_second:
            continue
        used_first.add(t1)
        used_second.add(t2)
        matches.append((t1, t2))
    matches.sort()
    return matches


def quat_xyzw_to_rotmat(qx, qy, qz, qw):
    """Unit quaternion (xyzw order -- TUM/ETH3D convention) -> 3x3 rotation."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 2.0 / n if n > 0 else 0.0
    return np.array(
        [
            [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
            [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
            [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def quat_wxyz_to_rotmat(qw, qx, qy, qz):
    """Unit quaternion (wxyz order -- EuRoC convention) -> 3x3 rotation."""
    return quat_xyzw_to_rotmat(qx, qy, qz, qw)


def split_train_val(n, val_fraction=0.15):
    """Contiguous (not shuffled) train/val split -- adjacent frames in a
    continuous video walkthrough are near-duplicates, so a random shuffle
    would leak near-identical frames across the split.
    Returns (train_slice, val_slice).
    """
    n_val = max(1, int(n * val_fraction))
    return slice(0, n - n_val), slice(n - n_val, n)


def compute_coverage(data, sequence, device="cuda", pos_threshold=0.5, batch_size=32):
    """Pairwise view-overlap matrix for one sequence of any family Data
    object (only needs .color_paths[sequence], .c2ws[sequence],
    .get_view(sequence, idx, resolution)) -- plays the same role as
    ScanNet++'s precomputed data/scannetpp/coverage/<seq>.json, computed
    on the fly here instead since none of TUM/7-Scenes/EuRoC/ETH3D ship
    one. Uses utils.loss_mask.calculate_in_frustum_mask (the same overlap
    check the training loss itself uses for masking) after a coarse
    camera-position-distance prefilter, since exact O(n^2) is not
    tractable at these datasets' frame counts. See the
    splatt3r-lora-finetuning skill for measured timing.

    Returns a dense {i: {j: overlap_fraction}} dict (0.0 for
    filtered-out pairs, not missing keys) because
    data.data.DUST3RSplattingDataset.sample() does dense coverage[i][j]
    lookups over every frame index, not just nearby ones.
    """
    import torch

    from utils.loss_mask import calculate_in_frustum_mask

    n = len(data.color_paths[sequence])
    positions = np.stack([c2w[:3, 3] for c2w in data.c2ws[sequence]])

    candidate_pairs = []
    for i in range(n):
        d = np.linalg.norm(positions - positions[i], axis=-1)
        for j in np.where((d < pos_threshold) & (np.arange(n) != i))[0]:
            candidate_pairs.append((i, int(j)))

    coverage = {i: {j: 0.0 for j in range(n) if j != i} for i in range(n)}
    if not candidate_pairs:
        return coverage

    resolution = (224, 224)  # small + fast; only used for the overlap metric
    view_cache = {}

    def get(idx):
        if idx not in view_cache:
            v = data.get_view(sequence, idx, resolution)
            view_cache[idx] = (
                torch.from_numpy(v["depthmap"]).float(),
                torch.from_numpy(v["camera_intrinsics"]).float(),
                torch.from_numpy(v["camera_pose"]).float(),
            )
        return view_cache[idx]

    for start in range(0, len(candidate_pairs), batch_size):
        batch = candidate_pairs[start : start + batch_size]
        depth1 = torch.stack([get(i)[0] for i, j in batch]).unsqueeze(1).to(device)
        intr1 = torch.stack([get(i)[1] for i, j in batch]).unsqueeze(1).to(device)
        c2w1 = torch.stack([get(i)[2] for i, j in batch]).unsqueeze(1).to(device)
        depth2 = torch.stack([get(j)[0] for i, j in batch]).unsqueeze(1).to(device)
        intr2 = torch.stack([get(j)[1] for i, j in batch]).unsqueeze(1).to(device)
        c2w2 = torch.stack([get(j)[2] for i, j in batch]).unsqueeze(1).to(device)

        mask = calculate_in_frustum_mask(depth1, intr1, c2w1, depth2, intr2, c2w2)
        frac = mask.float().mean(dim=(1, 2, 3))

        for (i, j), f in zip(batch, frac.tolist()):
            coverage[i][j] = f

    return coverage

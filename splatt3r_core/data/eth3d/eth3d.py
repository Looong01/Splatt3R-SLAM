"""ETH3D (SLAM mono benchmark subset) -> Splatt3R training data adapter,
pooling every scene found under family_root/train/ into one Data source.

This ETH3D download is the monocular SLAM-benchmark subset (rgb.txt,
calibration.txt, groundtruth.txt, rgb/*.png) -- NOT the full ETH3D
dataset, which does have LiDAR-scanned ground-truth depth for some
scenes; that fuller version isn't what's downloaded here. So, like
EuRoC, depth comes from the base model's own self-prediction, precomputed
and cached by scripts/precompute_pseudo_depth.py (run that first --
get_view() raises a clear FileNotFoundError otherwise). See
pseudo_depth.py and the splatt3r-lora-finetuning skill.

Format: identical conventions to TUM (rgb.txt/groundtruth.txt: same
"timestamp filename" / "timestamp tx ty tz qx qy qz qw" layout, xyzw
quaternion order) except calibration.txt is a single line "fx fy cx cy"
(no distortion) instead of TUM's fixed per-freiburg-index constants.

61 scenes are available under datasets/eth3d/train/ as of writing --
pseudo-depth precompute cost scales with total frame count across
however many of them max_scenes lets through; see
scripts/precompute_pseudo_depth.py's own MAX_ETH3D_SCENES for where to
adjust this for a faster first pass.
"""
import glob
import os

import cv2
import numpy as np

from data.common import (
    NORMALIZE_EXPOSURE,
    SequenceExposureLock,
    associate,
    quat_xyzw_to_rotmat,
    read_file_list,
    split_train_val,
)
from data.data import crop_resize_if_necessary


class ETH3DData:
    def __init__(self, family_root, stage, pseudo_depth_root=None, val_fraction=0.15,
                 max_time_diff=0.02, max_scenes=None):
        self.stage = stage
        self.pseudo_depth_root = pseudo_depth_root or family_root

        self.sequences = []
        self.color_paths, self.c2ws, self.intrinsics = {}, {}, {}

        scene_dirs = sorted(glob.glob(os.path.join(family_root, "train", "*")))
        if max_scenes is not None:
            scene_dirs = scene_dirs[:max_scenes]

        for root in scene_dirs:
            rgb_txt = os.path.join(root, "rgb.txt")
            gt_txt = os.path.join(root, "groundtruth.txt")
            calib_txt = os.path.join(root, "calibration.txt")
            if not (os.path.exists(rgb_txt) and os.path.exists(gt_txt) and os.path.exists(calib_txt)):
                continue
            sequence = os.path.basename(os.path.normpath(root))

            fx, fy, cx, cy = np.loadtxt(calib_txt, dtype=np.float32)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            rgb_list = read_file_list(rgb_txt)
            gt_list = read_file_list(gt_txt)
            matches = associate(rgb_list, gt_list, max_time_diff)

            color_paths, c2ws = [], []
            for rgb_ts, gt_ts in matches:
                tx, ty, tz, qx, qy, qz, qw = (float(v) for v in gt_list[gt_ts])
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = quat_xyzw_to_rotmat(qx, qy, qz, qw)
                c2w[:3, 3] = [tx, ty, tz]
                color_paths.append(os.path.join(root, rgb_list[rgb_ts][0]))
                c2ws.append(c2w)

            if len(color_paths) < 10:
                continue

            train_sl, val_sl = split_train_val(len(color_paths), val_fraction)
            sl = train_sl if stage == "train" else val_sl

            self.sequences.append(sequence)
            self.color_paths[sequence] = color_paths[sl]
            self.c2ws[sequence] = c2ws[sl]
            self.intrinsics[sequence] = K

        # Exposure normalization (data/common.py: NORMALIZE_EXPOSURE) --
        # lock each sequence's gain from its first frame, eagerly, so it's
        # deterministic across DDP ranks / DataLoader workers.
        self.exposure_lock = SequenceExposureLock()
        if NORMALIZE_EXPOSURE:
            for sequence in self.sequences:
                self.exposure_lock.lock(sequence, self._load_color(sequence, 0))

    def _load_color(self, sequence, view_idx):
        """Raw on-disk colour image (uint8 (H, W, 3)), before any exposure
        normalization or crop/resize. Shared by get_view() and the
        first-frame exposure lock in __init__."""
        color_path = self.color_paths[sequence][view_idx]
        return cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)

    def pseudo_depth_path(self, sequence, view_idx):
        color_path = self.color_paths[sequence][view_idx]
        stem = os.path.splitext(os.path.basename(color_path))[0]
        return os.path.join(self.pseudo_depth_root, "train", sequence, "pseudo_depth", f"{stem}.npy")

    def get_view(self, sequence, view_idx, resolution):
        rgb_image = self._load_color(sequence, view_idx)
        if NORMALIZE_EXPOSURE:
            rgb_image = self.exposure_lock.apply(rgb_image, sequence)

        depth_path = self.pseudo_depth_path(sequence, view_idx)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(
                f"No cached pseudo-depth at {depth_path}. Run "
                f"scripts/precompute_pseudo_depth.py first -- see the "
                f"splatt3r-lora-finetuning skill."
            )
        depthmap = np.load(depth_path).astype(np.float32)

        c2w = self.c2ws[sequence][view_idx]
        intrinsics = self.intrinsics[sequence]

        rgb_image, depthmap, intrinsics = crop_resize_if_necessary(
            rgb_image, depthmap, intrinsics, resolution
        )

        return {
            "original_img": rgb_image,
            "depthmap": depthmap,
            "camera_pose": c2w,
            "camera_intrinsics": intrinsics,
            "dataset": "eth3d",
            "label": f"eth3d/{sequence}",
            "instance": f"{view_idx}",
            "is_metric_scale": False,  # pseudo-depth is the model's own scale, not real metres
            "sky_mask": depthmap <= 0.0,
        }

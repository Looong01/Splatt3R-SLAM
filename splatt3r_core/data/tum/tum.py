"""TUM RGB-D -> Splatt3R training data adapter, pooling ALL freiburg*
sequences found under a family root directory into one Data source.

Mirrors data/scannetpp/scannetpp.py's ScanNetPPData interface
(.sequences, .color_paths[seq], .depth_paths[seq], .c2ws[seq],
.intrinsics[seq], .get_view(seq, idx, resolution)) so it plugs directly
into the dataset-agnostic data/data.py: DUST3RSplattingDataset /
DUST3RSplattingTestDataset without needing to reimplement context/target
sampling.

Ground-truth poses, depth, and intrinsics all come from each TUM sequence
itself -- no ScanNet++ access needed. See the splatt3r-lora-finetuning
skill for the full design rationale.
"""
import glob
import os
import re

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

# TUM freiburg{1,2,3} calibration (fx, fy, cx, cy, k1, k2, p1, p2, k3),
# matching splatt3r_slam/dataloader.py: TUMDataset and config/intrinsics.yaml.
FREIBURG_CALIB = {
    1: (517.3, 516.5, 318.6, 255.3),
    2: (520.9, 521.0, 325.1, 249.7),
    3: (535.4, 539.2, 320.1, 247.6),
}

# TUM 16-bit depth PNG -> metres.
TUM_PNG_DEPTH_SCALE = 5000.0


class TUMData:
    """Pools every `rgbd_dataset_freiburg*` sequence found directly under
    `family_root` (e.g. datasets/tum/) into one Data source, one
    `self.sequences` entry per sequence directory.
    """

    def __init__(self, family_root, stage, val_fraction=0.15, max_time_diff=0.02):
        self.stage = stage
        self.png_depth_scale = TUM_PNG_DEPTH_SCALE

        self.sequences = []
        self.color_paths, self.depth_paths, self.c2ws, self.intrinsics = {}, {}, {}, {}

        seq_dirs = sorted(glob.glob(os.path.join(family_root, "rgbd_dataset_freiburg*")))
        for root in seq_dirs:
            if not os.path.exists(os.path.join(root, "rgb.txt")):
                continue
            sequence = os.path.basename(os.path.normpath(root))

            match = re.search(r"freiburg(\d+)", sequence)
            freiburg_idx = int(match.group(1)) if match else 1
            fx, fy, cx, cy = FREIBURG_CALIB.get(freiburg_idx, FREIBURG_CALIB[1])

            rgb_list = read_file_list(os.path.join(root, "rgb.txt"))
            depth_list = read_file_list(os.path.join(root, "depth.txt"))
            gt_list = read_file_list(os.path.join(root, "groundtruth.txt"))

            rgb_depth_matches = associate(rgb_list, depth_list, max_time_diff)
            depth_ts_by_rgb_ts = {rt: dt for rt, dt in rgb_depth_matches}
            rgb_gt_matches = associate({rt: None for rt, _ in rgb_depth_matches}, gt_list, max_time_diff)

            color_paths, depth_paths, c2ws = [], [], []
            for rgb_ts, gt_ts in rgb_gt_matches:
                depth_ts = depth_ts_by_rgb_ts[rgb_ts]
                tx, ty, tz, qx, qy, qz, qw = (float(v) for v in gt_list[gt_ts])
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = quat_xyzw_to_rotmat(qx, qy, qz, qw)
                c2w[:3, 3] = [tx, ty, tz]

                color_paths.append(os.path.join(root, rgb_list[rgb_ts][0]))
                depth_paths.append(os.path.join(root, depth_list[depth_ts][0]))
                c2ws.append(c2w)

            if len(color_paths) < 10:
                continue  # too few associated frames to be useful

            train_sl, val_sl = split_train_val(len(color_paths), val_fraction)
            sl = train_sl if stage == "train" else val_sl

            self.sequences.append(sequence)
            self.color_paths[sequence] = color_paths[sl]
            self.depth_paths[sequence] = depth_paths[sl]
            self.c2ws[sequence] = c2ws[sl]
            self.intrinsics[sequence] = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )

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
        rgb_path = self.color_paths[sequence][view_idx]
        return cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    def get_view(self, sequence, view_idx, resolution):
        rgb_image = self._load_color(sequence, view_idx)
        if NORMALIZE_EXPOSURE:
            rgb_image = self.exposure_lock.apply(rgb_image, sequence)

        depth_path = self.depth_paths[sequence][view_idx]
        depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depthmap = depthmap / self.png_depth_scale

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
            "dataset": "tum",
            "label": f"tum/{sequence}",
            "instance": f"{view_idx}",
            "is_metric_scale": True,
            "sky_mask": depthmap <= 0.0,
        }


# compute_coverage() moved to data/common.py -- it's fully generic across
# any family's Data object (only touches .color_paths/.c2ws/.get_view()),
# reused verbatim by all four data/<family>/<family>.py modules. Kept
# importable here too for backwards compatibility with anything that did
# `from data.tum.tum import compute_coverage`.
from data.common import compute_coverage  # noqa: E402,F401

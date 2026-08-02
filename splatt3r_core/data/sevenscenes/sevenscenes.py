"""Microsoft 7-Scenes -> Splatt3R training data adapter, pooling every
already-extracted `<scene>/seq-*/` sequence found under the family root
into one Data source. Real (sensor) depth, same interface contract as
data/tum/tum.py: TUMData / data/scannetpp/scannetpp.py: ScanNetPPData.

7-Scenes ships per-frame `frame-NNNNNN.{color.png,depth.png,pose.txt}`:
  - depth.png: uint16, millimetres, 65535 = invalid (sensor dropout).
  - pose.txt: plain 4x4 camera-to-world matrix, already the right
    convention (no quaternion decoding needed, unlike TUM/ETH3D/EuRoC).
  - intrinsics: not shipped per-frame/per-scene; 7-Scenes was captured
    with a fixed Kinect v1, and splatt3r_slam/dataloader.py:
    SevenScenesDataset already hardcodes the standard fx=fy=585,
    cx=320, cy=240 for all scenes -- reused verbatim here.

Only sequences that are actually extracted (a `seq-NN/` directory, not
just a `seq-NN.zip`) are picked up -- as of writing, only `seq-01` is
extracted for each of the 7 scenes (chess/fire/heads/office/pumpkin/
redkitchen/stairs), ~1000 frames each. Unzip more `seq-*.zip` files
under datasets/7-scenes/<scene>/ for more training data; nothing else
needs to change, this class re-globs on every construction.
"""
import glob
import os

import cv2
import numpy as np
from natsort import natsorted

from data.common import NORMALIZE_EXPOSURE, SequenceExposureLock, split_train_val
from data.data import crop_resize_if_necessary

SEVENSCENES_INTRINSICS = (585.0, 585.0, 320.0, 240.0)  # fx, fy, cx, cy
SEVENSCENES_DEPTH_SCALE = 1000.0  # mm -> m
SEVENSCENES_INVALID_DEPTH = 65535


class SevenScenesData:
    def __init__(self, family_root, stage, val_fraction=0.15):
        self.stage = stage

        self.sequences = []
        self.color_paths, self.depth_paths, self.c2ws, self.intrinsics = {}, {}, {}, {}

        fx, fy, cx, cy = SEVENSCENES_INTRINSICS
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        seq_dirs = sorted(glob.glob(os.path.join(family_root, "*", "seq-*")))
        for seq_dir in seq_dirs:
            if not os.path.isdir(seq_dir):
                continue  # skip seq-NN.zip, only extracted dirs
            scene_name = os.path.basename(os.path.dirname(seq_dir))
            seq_name = os.path.basename(seq_dir)
            sequence = f"{scene_name}_{seq_name}"

            color_files = natsorted(glob.glob(os.path.join(seq_dir, "*.color.png")))
            if len(color_files) < 10:
                continue

            color_paths, depth_paths, c2ws = [], [], []
            for color_path in color_files:
                stem = color_path[: -len(".color.png")]
                depth_path = stem + ".depth.png"
                pose_path = stem + ".pose.txt"
                if not (os.path.exists(depth_path) and os.path.exists(pose_path)):
                    continue
                c2w = np.loadtxt(pose_path, dtype=np.float32)
                if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
                    continue  # 7-Scenes marks some frames as tracking failures this way
                color_paths.append(color_path)
                depth_paths.append(depth_path)
                c2ws.append(c2w)

            if len(color_paths) < 10:
                continue

            train_sl, val_sl = split_train_val(len(color_paths), val_fraction)
            sl = train_sl if stage == "train" else val_sl

            self.sequences.append(sequence)
            self.color_paths[sequence] = color_paths[sl]
            self.depth_paths[sequence] = depth_paths[sl]
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
        rgb_path = self.color_paths[sequence][view_idx]
        return cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    def get_view(self, sequence, view_idx, resolution):
        rgb_image = self._load_color(sequence, view_idx)
        if NORMALIZE_EXPOSURE:
            rgb_image = self.exposure_lock.apply(rgb_image, sequence)

        depth_path = self.depth_paths[sequence][view_idx]
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depthmap = depth_raw.astype(np.float32) / SEVENSCENES_DEPTH_SCALE
        depthmap[depth_raw == SEVENSCENES_INVALID_DEPTH] = 0.0

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
            "dataset": "7-scenes",
            "label": f"7-scenes/{sequence}",
            "instance": f"{view_idx}",
            "is_metric_scale": True,
            "sky_mask": depthmap <= 0.0,
        }

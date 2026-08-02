"""EuRoC MAV -> Splatt3R training data adapter, pooling every sequence
found under the family root into one Data source.

EuRoC is stereo + IMU with no depth sensor -- depth here comes from the
base model's own self-prediction, precomputed and cached to disk by
scripts/precompute_pseudo_depth.py (run that BEFORE using this class for
training; get_view() raises a clear FileNotFoundError pointing at that
script if a frame's cached depth is missing). See pseudo_depth.py and the
splatt3r-lora-finetuning skill for what that means/its limitations. Only
cam0 is used (mono) -- no stereo matching, since pseudo-depth doesn't
need it.

Format specifics (all from the already-downloaded ASL/EuRoC dataset
layout, cross-checked against splatt3r_slam/dataloader.py: EurocDataset,
which already runs SLAM tracking -- not training -- on this data):
  - images: mav0/cam0/data/<timestamp_ns>.png, grayscale, heavily
    distorted (radial-tangential) -- undistorted here via OpenCV, same
    approach as EurocDataset (comment there: "the distortion is too
    much to handle for MASt3R").
  - intrinsics + distortion + T_BS (cam0-to-body extrinsics):
    mav0/cam0/sensor.yaml.
  - ground truth: mav0/state_groundtruth_estimate0/data.csv, BODY pose
    in world frame, quaternion in **wxyz** order (note: different from
    TUM/ETH3D/7-Scenes' xyzw/matrix conventions) -- camera pose is
    T_WC = T_WB @ T_BS.
  - timestamps are nanosecond integers; converted to seconds here so
    data.common.associate()'s max_time_diff has the same meaning
    (fraction of a second) across all four families.
"""
import glob
import os

import cv2
import numpy as np
import yaml

from data.common import (
    NORMALIZE_EXPOSURE,
    SequenceExposureLock,
    associate,
    quat_wxyz_to_rotmat,
    read_file_list,
    split_train_val,
)
from data.data import crop_resize_if_necessary


def _load_cam0_calibration(seq_root):
    with open(os.path.join(seq_root, "mav0", "cam0", "sensor.yaml")) as f:
        cam0 = yaml.safe_load(f)
    W, H = cam0["resolution"]
    fx, fy, cx, cy = cam0["intrinsics"]
    distortion = np.array(cam0["distortion_coefficients"], dtype=np.float32)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    T_BS = np.array(cam0["T_BS"]["data"], dtype=np.float32).reshape(4, 4)

    K_opt, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (W, H), 0, (W, H))
    mapx, mapy = cv2.initUndistortRectifyMap(K, distortion, None, K_opt, (W, H), cv2.CV_32FC1)
    return dict(W=W, H=H, K=K, K_opt=K_opt, distortion=distortion, mapx=mapx, mapy=mapy, T_BS=T_BS)


class EuRoCData:
    def __init__(self, family_root, stage, pseudo_depth_root=None, val_fraction=0.15, max_time_diff=0.05):
        self.stage = stage
        # Where scripts/precompute_pseudo_depth.py wrote cached depth for
        # this family; defaults to alongside the raw data.
        self.pseudo_depth_root = pseudo_depth_root or family_root

        self.sequences = []
        self.color_paths, self.c2ws, self.intrinsics, self.calib = {}, {}, {}, {}

        seq_dirs = sorted(glob.glob(os.path.join(family_root, "*")))
        for seq_root in seq_dirs:
            cam0_csv = os.path.join(seq_root, "mav0", "cam0", "data.csv")
            gt_csv = os.path.join(seq_root, "mav0", "state_groundtruth_estimate0", "data.csv")
            if not (os.path.exists(cam0_csv) and os.path.exists(gt_csv)):
                continue
            sequence = os.path.basename(os.path.normpath(seq_root))

            calib = _load_cam0_calibration(seq_root)

            cam0_list_raw = read_file_list(cam0_csv, delimiter=",")
            gt_list_raw = read_file_list(gt_csv, delimiter=",")
            # ns -> s, so max_time_diff means the same thing across families.
            cam0_list = {ts / 1e9: v for ts, v in cam0_list_raw.items()}
            gt_list = {ts / 1e9: v for ts, v in gt_list_raw.items()}

            matches = associate(cam0_list, gt_list, max_time_diff)

            color_paths, c2ws = [], []
            for cam_ts, gt_ts in matches:
                filename = cam0_list[cam_ts][0]
                px, py, pz = (float(v) for v in gt_list[gt_ts][0:3])
                qw, qx, qy, qz = (float(v) for v in gt_list[gt_ts][3:7])
                T_WB = np.eye(4, dtype=np.float32)
                T_WB[:3, :3] = quat_wxyz_to_rotmat(qw, qx, qy, qz)
                T_WB[:3, 3] = [px, py, pz]
                c2w = T_WB @ calib["T_BS"]

                color_paths.append(os.path.join(seq_root, "mav0", "cam0", "data", filename))
                c2ws.append(c2w)

            if len(color_paths) < 10:
                continue

            train_sl, val_sl = split_train_val(len(color_paths), val_fraction)
            sl = train_sl if stage == "train" else val_sl

            self.sequences.append(sequence)
            self.color_paths[sequence] = color_paths[sl]
            self.c2ws[sequence] = c2ws[sl]
            self.intrinsics[sequence] = calib["K_opt"]
            self.calib[sequence] = calib

        # Exposure normalization (data/common.py: NORMALIZE_EXPOSURE) --
        # lock each sequence's gain from its first frame, eagerly, so it's
        # deterministic across DDP ranks / DataLoader workers.
        self.exposure_lock = SequenceExposureLock()
        if NORMALIZE_EXPOSURE:
            for sequence in self.sequences:
                self.exposure_lock.lock(sequence, self._load_color(sequence, 0))

    def _load_color(self, sequence, view_idx):
        """Undistorted colour image (uint8 (H, W, 3)), before any exposure
        normalization or crop/resize. Shared by get_view() and the
        first-frame exposure lock in __init__."""
        color_path = self.color_paths[sequence][view_idx]
        calib = self.calib[sequence]
        raw = cv2.imread(color_path, cv2.IMREAD_GRAYSCALE)
        undistorted = cv2.remap(raw, calib["mapx"], calib["mapy"], cv2.INTER_LINEAR)
        return cv2.cvtColor(undistorted, cv2.COLOR_GRAY2RGB)

    def pseudo_depth_path(self, sequence, view_idx):
        color_path = self.color_paths[sequence][view_idx]
        stem = os.path.splitext(os.path.basename(color_path))[0]
        return os.path.join(self.pseudo_depth_root, sequence, "pseudo_depth", f"{stem}.npy")

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
            "dataset": "euroc",
            "label": f"euroc/{sequence}",
            "instance": f"{view_idx}",
            "is_metric_scale": False,  # pseudo-depth is the model's own scale, not real metres
            "sky_mask": depthmap <= 0.0,
        }

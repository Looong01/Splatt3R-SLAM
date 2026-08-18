"""Replica (NICE-SLAM rendered release) -> Splatt3R training data adapter.

Fifth family, added because the four existing ones (TUM, 7-Scenes, EuRoC,
ETH3D) are all classic real-sensor benchmarks, while Replica is the one every
recent Gaussian-splatting SLAM paper reports on (SplaTAM, MonoGS, GS-SLAM,
Photo-SLAM, Gaussian-SLAM) and a standard evaluation set in the VGGT-era
feed-forward literature.

It also removes a confound the other four cannot. Replica is *rendered*, so:

  - poses are exact, not tracked. Every other family's "ground truth" carries
    motion-capture or SLAM error, which lands in the supervision as pose noise.
  - depth is exact and complete. No sensor dropout, no 65535 sentinels, no
    range limit -- 7-Scenes and TUM both lose whole surfaces to those.
  - exposure is constant by construction, so the SequenceExposureLock is a
    no-op here and any colour effect measured on this family cannot be
    auto-exposure drift.

That makes Replica the cleanest available test of whether the head-only recipe
generalizes, and a negative on Replica would be much harder to explain away
than a negative on a sensor dataset.

Layout of the NICE-SLAM release:

    Replica/
      cam_params.json                     one intrinsics block for all scenes
      office0/traj.txt                    one line per frame, 16 numbers,
                                          row-major 4x4 camera-to-world
      office0/results/frame000000.jpg     colour
      office0/results/depth000000.png     uint16, scale in cam_params.json

Depth is uint16 scaled by `png_depth_scale` (6553.5 in the released files,
i.e. 1/10 mm), and 0 marks nothing -- the renderer fills every pixel. The
`sky_mask` therefore comes out empty, which is correct and is the point.
"""
import glob
import json
import os

import cv2
import numpy as np
from natsort import natsorted

from data.common import NORMALIZE_EXPOSURE, SequenceExposureLock, split_train_val
from data.data import crop_resize_if_necessary

# Fallbacks matching NICE-SLAM's configs/Replica/replica.yaml, used only if
# cam_params.json is missing; the file ships with the release and is preferred.
REPLICA_INTRINSICS = (600.0, 600.0, 599.5, 339.5)  # fx, fy, cx, cy
REPLICA_DEPTH_SCALE = 6553.5


class ReplicaData:
    def __init__(self, family_root, stage, val_fraction=0.15, max_scenes=None,
                 degrade=None):
        """degrade: None | "content" | "photometry" | "photometry-spatial" | "mixed".

        Kimi's round-24 causal test for why the Replica head never learned to
        use partial opacity. His corrected channel: on clean rendered images the
        BACKBONE's geometry is confident, multi-view conflict is rare, and
        saturation is never punished -- the dataset's exact depth never reaches
        the network at all. Two perturbations separate that from the obvious
        alternative:

          content     noise + blur on BOTH views -> matching uncertainty ->
                      geometric conflict. Predicts un-saturation.
          photometry  brightness jitter on the TARGET view only. Opacity is
                      view-independent and cannot hedge per-view brightness,
                      so this predicts NO un-saturation.

        Measured (17.73): content did NOT un-saturate (100.0%); photometry
        PARTIALLY did (99.4% -> 52.1% frac>0.99). Kimi's round-33 model for why
        it was only partial: a per-Gaussian colour residual can serve a GLOBAL
        brightness shift almost exactly (one scalar/channel), so the head barely
        needs opacity for it. A SPATIALLY-VARYING photometric effect -- one
        whose correction depends on where a point lands in the TARGET frame,
        not just on which source pixel produced it -- is a much harder mapping
        for a per-source-pixel colour predictor to learn exactly, so it should
        recruit more of the opacity channel. `photometry-spatial` tests that:

          photometry-spatial   per-channel WB jitter + radial vignetting on
                                ONE view. Pre-registered (round 33): frac>0.9
                                drops to 30-60%, well below plain photometry's
                                83.1% and still above TUM's unaided 12.5%.
        """
        self.degrade = degrade
        self.stage = stage
        self.sequences = []
        self.color_paths, self.depth_paths, self.c2ws, self.intrinsics = {}, {}, {}, {}

        fx, fy, cx, cy = REPLICA_INTRINSICS
        self.depth_scale = REPLICA_DEPTH_SCALE
        cam_json = os.path.join(family_root, "cam_params.json")
        if os.path.exists(cam_json):
            with open(cam_json) as f:
                cam = json.load(f).get("camera", {})
            fx = float(cam.get("fx", fx)); fy = float(cam.get("fy", fy))
            cx = float(cam.get("cx", cx)); cy = float(cam.get("cy", cy))
            self.depth_scale = float(cam.get("scale", self.depth_scale))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        scene_dirs = sorted(d for d in glob.glob(os.path.join(family_root, "*"))
                            if os.path.isdir(d)
                            and os.path.exists(os.path.join(d, "traj.txt")))
        if max_scenes:
            scene_dirs = scene_dirs[:max_scenes]

        for scene_dir in scene_dirs:
            sequence = os.path.basename(scene_dir)
            color_files = natsorted(glob.glob(
                os.path.join(scene_dir, "results", "frame*.jpg")))
            if len(color_files) < 10:
                continue
            traj = np.loadtxt(os.path.join(scene_dir, "traj.txt"),
                              dtype=np.float32)
            if traj.ndim == 1:
                traj = traj[None]
            # One pose per frame, in file order. A length mismatch means the
            # release was partially extracted; truncate rather than guess,
            # because pairing frame i with pose j silently trains on wrong
            # geometry and nothing downstream would catch it.
            n = min(len(color_files), len(traj))
            color_paths, depth_paths, c2ws = [], [], []
            for i in range(n):
                cp = color_files[i]
                dp = cp.replace("frame", "depth").replace(".jpg", ".png")
                if not os.path.exists(dp):
                    continue
                c2w = traj[i].reshape(4, 4)
                if not np.all(np.isfinite(c2w)):
                    continue
                color_paths.append(cp)
                depth_paths.append(dp)
                c2ws.append(c2w)
            if len(color_paths) < 10:
                continue

            train_sl, val_sl = split_train_val(len(color_paths), val_fraction)
            sl = train_sl if stage == "train" else val_sl
            self.sequences.append(sequence)
            self.color_paths[sequence] = color_paths[sl]
            self.depth_paths[sequence] = depth_paths[sl]
            self.c2ws[sequence] = list(np.asarray(c2ws)[sl])
            self.intrinsics[sequence] = K

        self.exposure_lock = SequenceExposureLock()
        if NORMALIZE_EXPOSURE:
            for sequence in self.sequences:
                self.exposure_lock.lock(sequence, self._load_color(sequence, 0))

    def _load_color(self, sequence, view_idx):
        rgb_path = self.color_paths[sequence][view_idx]
        return cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    def get_view(self, sequence, view_idx, resolution):
        rgb_image = self._load_color(sequence, view_idx)
        if self.degrade == "content":
            rng = np.random.default_rng(abs(hash((sequence, view_idx))) % 2**32)
            k = int(rng.integers(0, 3)) * 2 + 1
            if k > 1:
                rgb_image = cv2.GaussianBlur(rgb_image, (k, k), 0)
            rgb_image = np.clip(rgb_image.astype(np.float32)
                                + rng.normal(0, 6.0, rgb_image.shape), 0, 255
                                ).astype(np.uint8)
        elif self.degrade == "photometry":
            rng = np.random.default_rng(abs(hash((sequence, view_idx))) % 2**32)
            g = float(rng.uniform(0.8, 1.25))
            rgb_image = np.clip(rgb_image.astype(np.float32) * g, 0, 255
                                ).astype(np.uint8)
        elif self.degrade == "photometry-spatial":
            rng = np.random.default_rng(abs(hash((sequence, view_idx))) % 2**32)
            h, w = rgb_image.shape[:2]
            img = rgb_image.astype(np.float32)
            # per-channel WB jitter: still close to f_dc-servable (one scalar
            # per channel), kept mild so the spatial term below is what drives
            # any additional effect.
            wb = rng.uniform(0.9, 1.1, size=3).astype(np.float32)
            img = img * wb[None, None, :]
            # radial vignetting, jittered centre: darkens toward the image
            # edge by an amount that depends on TARGET-frame position, not on
            # which source pixel produced the point -- the effect a per-source
            # -pixel colour residual cannot anticipate without knowing where
            # in the OTHER view's frame this point will land.
            cy = h * float(rng.uniform(0.35, 0.65))
            cx = w * float(rng.uniform(0.35, 0.65))
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r2 = ((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2
            r2 = r2 / r2.max().clip(min=1e-6)
            v = float(rng.uniform(0.25, 0.55))
            falloff = (1.0 - v * r2)[:, :, None]
            img = img * falloff
            rgb_image = np.clip(img, 0, 255).astype(np.uint8)
        elif self.degrade == "mixed":
            # Round 37: the missing cell of the dissociation 2x2 (content-only,
            # photometry-spatial-only, neither=base already run; this is both
            # at once). Same rng object advanced through both blocks in
            # sequence, so applying them together does not change either
            # perturbation's own per-frame draw relative to running it alone
            # -- the calibration each was measured at is preserved.
            rng = np.random.default_rng(abs(hash((sequence, view_idx))) % 2**32)
            k = int(rng.integers(0, 3)) * 2 + 1
            if k > 1:
                rgb_image = cv2.GaussianBlur(rgb_image, (k, k), 0)
            rgb_image = np.clip(rgb_image.astype(np.float32)
                                + rng.normal(0, 6.0, rgb_image.shape), 0, 255
                                ).astype(np.uint8)
            h, w = rgb_image.shape[:2]
            img = rgb_image.astype(np.float32)
            wb = rng.uniform(0.9, 1.1, size=3).astype(np.float32)
            img = img * wb[None, None, :]
            cy = h * float(rng.uniform(0.35, 0.65))
            cx = w * float(rng.uniform(0.35, 0.65))
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r2 = ((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2
            r2 = r2 / r2.max().clip(min=1e-6)
            v = float(rng.uniform(0.25, 0.55))
            falloff = (1.0 - v * r2)[:, :, None]
            img = img * falloff
            rgb_image = np.clip(img, 0, 255).astype(np.uint8)
        if NORMALIZE_EXPOSURE:
            rgb_image = self.exposure_lock.apply(rgb_image, sequence)

        depth_raw = cv2.imread(self.depth_paths[sequence][view_idx],
                               cv2.IMREAD_UNCHANGED)
        depthmap = depth_raw.astype(np.float32) / self.depth_scale
        # Rendered depth has no dropout; 0 would mean "behind the far plane",
        # which does not occur in these scenes. Kept for interface parity.
        depthmap[depth_raw == 0] = 0.0

        c2w = self.c2ws[sequence][view_idx]
        intrinsics = self.intrinsics[sequence]
        rgb_image, depthmap, intrinsics = crop_resize_if_necessary(
            rgb_image, depthmap, intrinsics, resolution)

        return {
            "original_img": rgb_image,
            "depthmap": depthmap,
            "camera_pose": c2w,
            "camera_intrinsics": intrinsics,
            "dataset": "replica",
            "label": f"replica/{sequence}",
            "instance": f"{view_idx}",
            "is_metric_scale": True,
            "sky_mask": depthmap <= 0.0,
        }

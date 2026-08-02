"""Precompute self-predicted pseudo-depth for EuRoC and ETH3D (neither has
a real depth sensor / ground-truth depth in the format downloaded here --
see the splatt3r-lora-finetuning skill for the full explanation and its
limitations). Run this BEFORE scripts/train_lora_per_scene.py -- that
script's EuRoC/ETH3D loaders raise a clear FileNotFoundError if a frame's
cache entry is missing rather than silently doing something else.

Uses the BASE checkpoint only (no LoRA) -- pseudo-depth is a fixed
preprocessing step computed once, not something that updates as later
LoRA training progresses.

Idempotent / resumable: already-cached frames are skipped, so Ctrl-C and
re-run is safe.

Usage:
    cd /home/share-v5/Codes/Splatt3R-SLAM
    python3 scripts/precompute_pseudo_depth.py

This will take a while -- it's a real inference pass over every frame of
every sequence in both families. See MAX_ETH3D_SCENES below to bound the
ETH3D side (61 scenes available) for a faster first pass.
"""
import os
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Derived from this file's own location (scripts/ -> repo root), not
# hardcoded -- the previous absolute path silently broke the moment the
# repo was cloned or moved anywhere else.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(CORE, "src", "pixelsplat_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))
os.chdir(CORE)

import mast3r.utils.path_to_dust3r  # noqa
import cv2
import numpy as np
import torch

from data.euroc.euroc import EuRoCData
from data.eth3d.eth3d import ETH3DData
from main import MAST3RGaussians
from pseudo_depth import predict_pseudo_depth

BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints", "epoch=19-step=1200.ckpt")
EUROC_ROOT = os.path.join(REPO_ROOT, "datasets", "euroc")
ETH3D_ROOT = os.path.join(REPO_ROOT, "datasets", "eth3d")

# 61 ETH3D scenes are available; cap for a faster first pass. None = all.
MAX_ETH3D_SCENES = 15
MIN_CONFIDENCE = 1.5


def run_family(name, data_objs, device):
    print(f"\n{'=' * 70}\nPrecomputing pseudo-depth: {name}\n{'=' * 70}")
    t0 = time.time()
    n_done, n_skipped, n_seq = 0, 0, 0

    print("Loading base model (no LoRA)...")
    model = MAST3RGaussians.load_from_checkpoint(BASE_CKPT, map_location=device).to(device)
    model.eval()

    for data in data_objs:
        for sequence in data.sequences:
            n_seq += 1
            n_frames = len(data.color_paths[sequence])
            print(f"  [{name}] {sequence}: {n_frames} frames ({data.stage})")
            for view_idx in range(n_frames):
                out_path = data.pseudo_depth_path(sequence, view_idx)
                if os.path.exists(out_path):
                    n_skipped += 1
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                color_path = data.color_paths[sequence][view_idx]
                if name == "euroc":
                    calib = data.calib[sequence]
                    raw = cv2.imread(color_path, cv2.IMREAD_GRAYSCALE)
                    undistorted = cv2.remap(raw, calib["mapx"], calib["mapy"], cv2.INTER_LINEAR)
                    img = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)

                depth, _valid = predict_pseudo_depth(model, img, device, min_confidence=MIN_CONFIDENCE)
                np.save(out_path, depth)
                n_done += 1

    del model
    torch.cuda.empty_cache()
    print(f"{name}: {n_done} frames computed, {n_skipped} already cached, "
          f"{n_seq} sequences, {time.time() - t0:.1f}s")


def main():
    device = "cuda"

    euroc_data = [
        EuRoCData(EUROC_ROOT, "train"),
        EuRoCData(EUROC_ROOT, "val"),
    ]
    eth3d_data = [
        ETH3DData(ETH3D_ROOT, "train", max_scenes=MAX_ETH3D_SCENES),
        ETH3DData(ETH3D_ROOT, "val", max_scenes=MAX_ETH3D_SCENES),
    ]

    run_family("euroc", euroc_data, device)
    run_family("eth3d", eth3d_data, device)

    print("\nDone. You can now run scripts/train_lora_per_scene.py.")


if __name__ == "__main__":
    main()

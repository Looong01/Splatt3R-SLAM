"""Precompute self-predicted pseudo-depth for TUM -- the causal test for the
depth-source question (skill 17.85.11, rounds 34-36).

Not a general-purpose tool like scripts/precompute_pseudo_depth.py (which
exists because EuRoC/ETH3D have no real depth at all); TUM ALREADY has real
sensor depth. This script exists solely to produce the alternative-universe
depth needed for the one CAUSAL test in the depth-source inquiry: train a TUM
head on pseudo-depth, holding everything else (family, scenes, recipe, seed)
fixed, and see whether deployment gain moves.

Same pattern as scripts/precompute_pseudo_depth.py: base checkpoint only (no
LoRA), idempotent (skips already-cached frames), one .npy per frame under
<sequence>/pseudo_depth/.

Usage:
    cd /home/share-v5/Codes/Splatt3R-SLAM
    python3 scripts/precompute_pseudo_depth_tum.py
"""
import os
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

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

from data.tum.tum import TUMData
from main import MAST3RGaussians
from pseudo_depth import predict_pseudo_depth

BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints", "epoch=19-step=1200.ckpt")
TUM_ROOT = os.path.join(REPO_ROOT, "datasets", "tum")
MIN_CONFIDENCE = 1.5


def main():
    device = "cuda"
    print("Loading base model (no LoRA)...")
    model = MAST3RGaussians.load_from_checkpoint(BASE_CKPT, map_location=device).to(device)
    model.eval()

    # Both stages, so the resulting cache covers every frame TUMData("train")
    # and TUMData("val", depth_source="pseudo") will ever ask for -- same
    # reasoning as precompute_pseudo_depth.py's EuRoC/ETH3D coverage.
    t0 = time.time()
    n_done = n_skipped = n_seq = 0
    seen = set()
    for stage in ("train", "val"):
        data = TUMData(TUM_ROOT, stage, depth_source="sensor")  # colour paths only; depth_source irrelevant here
        for sequence in data.sequences:
            n_frames = len(data.color_paths[sequence])
            print(f"  [tum/{stage}] {sequence}: {n_frames} frames")
            n_seq += 1
            for view_idx in range(n_frames):
                key = (sequence, data.color_paths[sequence][view_idx])
                if key in seen:
                    continue
                seen.add(key)
                out_path = data.pseudo_depth_path(sequence, view_idx)
                if os.path.exists(out_path):
                    n_skipped += 1
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                img = data._load_color(sequence, view_idx)
                depth, _valid = predict_pseudo_depth(model, img, device, min_confidence=MIN_CONFIDENCE)
                np.save(out_path, depth)
                n_done += 1

    del model
    torch.cuda.empty_cache()
    print(f"tum: {n_done} frames computed, {n_skipped} already cached, "
          f"{n_seq} sequence-stage entries, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

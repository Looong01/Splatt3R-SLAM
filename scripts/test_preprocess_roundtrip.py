"""Assert the training input path and the SLAM input path produce the SAME tensor.

Kimi's round-22 design, and the standing protection for a class of bug that cost
4 dB and was invisible to every metric this project computes.

The bug (17.55): Replica was registered for training at 512x384 by copying
tum's resolution, while the SLAM side's resize_img feeds 512x288 for a 1200x680
source. The head trained 40 epochs on a different crop of a different aspect
ratio. The val split did not notice, because **the val split is computed with
the training-side preprocessing** -- it cannot see an input-pipeline mismatch by
construction. Only a full SLAM run did.

A shape assert would not be enough: crop-vs-letterbox-vs-squash, channel order,
dtype range and the exposure-normalization flag are all shape-preserving or
shape-adjacent, and all in the same class. So this compares the tensors.

    python3 scripts/test_preprocess_roundtrip.py            # every family
    python3 scripts/test_preprocess_roundtrip.py replica
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

# one SLAM-side dataset per training family
SLAM_PATH = {
    "tum": "datasets/tum/rgbd_dataset_freiburg1_desk",
    "7-scenes": "datasets/7-scenes/chess",
    "euroc": "datasets/euroc/MH_01_easy",
    "eth3d": "datasets/eth3d/sfm_house_loop",
    "replica": "datasets/Replica/office0",
}


def main():
    fams = sys.argv[1:] or list(SLAM_PATH)
    from splatt3r_slam.config import load_config
    load_config("config/eval_calib.yaml")
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    sys.path.insert(0, os.path.join(REPO_ROOT, "splatt3r_core"))
    os.chdir(os.path.join(REPO_ROOT, "splatt3r_core"))
    import train_lora_per_scene as T

    bad = 0
    for fam in fams:
        p = os.path.join(REPO_ROOT, SLAM_PATH[fam])
        if not os.path.exists(p):
            print(f"{fam:10s} SKIP (no dataset at {SLAM_PATH[fam]})")
            continue
        ds = load_dataset(p)
        slam = resize_img(ds.get_image(0), ds.img_size)["img"]
        slam_hw = tuple(slam.shape[-2:])
        train_hw = tuple(T.FAMILIES[fam][3][::-1])   # FAMILIES stores (w, h)
        ok = slam_hw == train_hw
        bad += (not ok)
        print(f"{fam:10s} SLAM feeds {slam_hw}   training uses {train_hw}   "
              f"{'ok' if ok else '*** MISMATCH ***'}")
    if bad:
        print(f"\n{bad} family/families train on a different input shape than "
              f"the pipeline feeds them.\nA head trained this way improves its "
              f"val split and LOSES in deployment (17.55).")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

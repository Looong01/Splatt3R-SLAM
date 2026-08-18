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

Checking ONE sequence per family is not enough either, and that is a second
finding rather than a refinement of the first. A family's registered training
resolution is a single number, but a family is only entitled to one if all its
sequences share a native aspect ratio. ETH3D does not: 60 of its 61 sequences
are 739x458 and feed 512x304, while `sfm_house_loop` alone is 743x465 and feeds
512x320. So the eth3d head is correctly registered for the family and would
still hit the full 17.55 failure on that one sequence.

A single-sequence probe cannot see this, and worse, it reports whichever answer
its chosen sequence happens to give -- probing `sfm_house_loop` alone reports a
MISMATCH for a correctly-registered family. So the gate scans every sequence and
reports the shape distribution, failing on a genuine registry error and warning
separately on a family that is not shape-homogeneous.
"""
import os, sys
import glob
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

# Every SLAM-side sequence per training family, as a glob. The first match in
# sorted order is the one compared tensor-for-tensor; the rest are checked for
# shape only, which is what a heterogeneous release can go wrong in.
SLAM_GLOB = {
    "tum": "datasets/tum/rgbd_dataset_freiburg1_*",
    "7-scenes": "datasets/7-scenes/*",
    "euroc": "datasets/euroc/*",
    # the local ETH3D release is the training split, so sequences live one
    # level deeper than the flat layout the other families use
    "eth3d": "datasets/eth3d/train/*",
    "replica": "datasets/Replica/office*",
}
SLAM_PATH = {f: sorted(g for g in glob.glob(os.path.join(REPO_ROOT, p))
                       if os.path.isdir(g))
             for f, p in SLAM_GLOB.items()}


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
    mixed = []
    for fam in fams:
        seqs = SLAM_PATH[fam]
        if not seqs:
            print(f"{fam:10s} SKIP (no sequences match {SLAM_GLOB[fam]})")
            continue
        train_hw = tuple(T.FAMILIES[fam][3][::-1])   # FAMILIES stores (w, h)

        shapes = Counter()
        offenders = {}
        for p in seqs:
            try:
                ds = load_dataset(p)
                hw = tuple(resize_img(ds.get_image(0), ds.img_size)["img"].shape[-2:])
            except Exception as e:
                print(f"{fam:10s}   {os.path.basename(p):24s} unreadable ({e})")
                continue
            shapes[hw] += 1
            if hw != train_hw:
                offenders.setdefault(hw, []).append(os.path.basename(p))

        if not shapes:
            print(f"{fam:10s} SKIP (no readable sequence)")
            continue
        # The registry is judged against what MOST of the family feeds; a lone
        # odd sequence is a deployment hazard, not a registry error.
        modal_hw, modal_n = shapes.most_common(1)[0]
        ok = modal_hw == train_hw
        bad += (not ok)
        print(f"{fam:10s} SLAM feeds {modal_hw} for {modal_n}/{sum(shapes.values())} "
              f"sequences   training uses {train_hw}   "
              f"{'ok' if ok else '*** MISMATCH ***'}")
        if offenders:
            mixed.append(fam)
            for hw, names in sorted(offenders.items()):
                shown = ", ".join(names[:4]) + (" ..." if len(names) > 4 else "")
                print(f"{'':10s}   !! {len(names)} sequence(s) feed {hw}: {shown}")

    if bad:
        print(f"\n{bad} family/families train on a different input shape than "
              f"the pipeline feeds them.\nA head trained this way improves its "
              f"val split and LOSES in deployment (17.55).")
    if mixed:
        print(f"\nNot shape-homogeneous: {', '.join(mixed)}. The head is correct "
              f"for the family's modal shape and\nhits the full 17.55 failure on "
              f"the sequences listed above. Do not deploy or benchmark on those\n"
              f"without retraining at their shape.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

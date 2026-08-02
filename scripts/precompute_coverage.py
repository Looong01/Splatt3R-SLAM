"""Precompute and cache the per-family coverage matrices.

`DUST3RSplattingDataset` picks its context/target views from a coverage matrix
(how much of one view's pixels reproject into another), and computing it is the
expensive part of standing up a new dataset family -- O(n^2) frame pairs before
the position-distance prefilter. Training computes it lazily on first use, which
is fine when training and precompute happen on the same GPU, but wastes a
training slot when they don't.

This script does only the precompute, so it can run on an otherwise-idle GPU
while a training run occupies the other one. It writes to exactly the paths
`train_lora_per_scene.py` and the `exp_head_only*.py` experiments read from
(same VAL_FRACTION and pos_threshold, hence the same cache tag), so a
subsequent training run finds the cache and skips straight past this step.

Only TUM was cached before this existed; 7-scenes / euroc / eth3d each need a
run of this before they can be trained.

Note euroc and eth3d have no ground-truth depth -- run
scripts/precompute_pseudo_depth.py for those first, or the data classes will
report zero usable sequences.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/precompute_coverage.py 7-scenes
    CUDA_VISIBLE_DEVICES=1 python3 scripts/precompute_coverage.py --all
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_lora_per_scene as T  # noqa: E402  (sets up sys.path/chdir on import)


def precompute(family, device="cuda"):
    data_cls, family_root, extra_kwargs, _res = T.FAMILIES[family]
    tag = f"valfrac{T.VAL_FRACTION}_pos{T.COVERAGE_POS_THRESHOLD}"
    os.makedirs(T.COVERAGE_CACHE_ROOT, exist_ok=True)

    for stage in ("train", "val"):
        cache = os.path.join(T.COVERAGE_CACHE_ROOT, f"{family}_{stage}_{tag}.pkl")
        if os.path.exists(cache):
            print(f"[{family}/{stage}] already cached: {cache}", flush=True)
            continue

        data = data_cls(family_root, stage, val_fraction=T.VAL_FRACTION, **extra_kwargs)
        if len(data.sequences) == 0:
            raise RuntimeError(
                f"no usable sequences for '{family}' under {family_root}"
                + (" -- run scripts/precompute_pseudo_depth.py first?"
                   if family in ("euroc", "eth3d") else ""))
        frames = sum(len(data.color_paths[s]) for s in data.sequences)
        print(f"[{family}/{stage}] {len(data.sequences)} sequences, {frames} frames "
              f"-> {cache}", flush=True)

        t0 = time.time()
        T.build_pooled_coverage(data, device, cache_path=cache)
        print(f"[{family}/{stage}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", nargs="?", choices=sorted(T.FAMILIES))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not args.all and args.family is None:
        ap.error("give a family name or --all")

    families = sorted(T.FAMILIES) if args.all else [args.family]
    for f in families:
        precompute(f)
    print("\nall requested coverage caches present", flush=True)


if __name__ == "__main__":
    main()

"""Experiment A's verdict table: joint N-view against pairwise, paired by window.

Both arms of `diag_nview_scale.py` score the SAME windows of the SAME frames
against the SAME sensor, so this pairs them window-by-window and reports a
paired test rather than two independent medians -- the difference matters
because scene content dominates the absolute level of every one of these
statistics, and pairing removes it.

The decision rule was fixed before the numbers were seen (17.36):

  VGGT scale spread << Splatt3R's  ->  an N-view backbone targets the parent of
                                       the seams; changing backbones has a hard
                                       justification.
  VGGT scale spread ~= Splatt3R's  ->  the bias is prior-bound, no backbone
                                       helps, and the map-side verdict stands.
"""
import argparse
import glob
import os

import numpy as np
from scipy.stats import binomtest, wilcoxon

# The neighbour step is the PRIMARY endpoint (Kimi, round 16): it is what an
# adjacent-cluster seam actually sees. The scale spread is secondary -- it also
# contains long-range wander, which accumulates differently depending on whether
# the error process is white or a random walk, and the two arms need not share
# that structure.
COLS = {"neighbour": 2, "scale": 1, "within": 3}


def load(arm, seq, d):
    p = os.path.join(d, f"{arm}_{seq}.npz")
    return np.load(p, allow_pickle=True)["rows"] if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/expA")
    ap.add_argument("--metric", default="neighbour", choices=tuple(COLS))
    ap.add_argument("--arms", default="splatt3r,vggt16",
                    help="two file prefixes to compare, control first")
    args = ap.parse_args()
    ctrl, test = args.arms.split(",")

    seqs = sorted({os.path.basename(p).split("_", 1)[1][:-4]
                   for p in glob.glob(os.path.join(args.dir, "*.npz"))})
    col = COLS[args.metric]

    print(f"metric: {args.metric}, median over windows, %\n")
    print(f"{'sequence':10s} {'n':>3} {ctrl:>9} {test:>9} "
          f"{'ratio':>7} {'paired p':>9}")
    pooled_s, pooled_v = [], []
    seq_ratio = []
    for seq in seqs:
        s, v = load(ctrl, seq, args.dir), load(test, seq, args.dir)
        if s is None or v is None:
            continue
        # pair on the window index, never on row order: a dropped window in one
        # arm would otherwise silently shift every later comparison
        sm = {int(r[0]): r for r in s}
        vm = {int(r[0]): r for r in v}
        keys = sorted(set(sm) & set(vm))
        if not keys:
            continue
        sv = np.array([sm[k][col] for k in keys])
        vv = np.array([vm[k][col] for k in keys])
        pooled_s.append(sv)
        pooled_v.append(vv)
        try:
            p = wilcoxon(sv, vv).pvalue
        except ValueError:
            p = float("nan")
        seq_ratio.append(np.median(vv) / np.median(sv))
        print(f"{seq:10s} {len(keys):>3} {np.median(sv)*100:>8.2f}% "
              f"{np.median(vv)*100:>8.2f}% {seq_ratio[-1]:>7.2f} "
              f"{p:>9.4f}")

    if not pooled_s:
        print("\nno paired sequences yet")
        return 1
    sv, vv = np.concatenate(pooled_s), np.concatenate(pooled_v)
    p = wilcoxon(sv, vv).pvalue
    print(f"\n{'POOLED':10s} {len(sv):>3} {np.median(sv)*100:>8.2f}% "
          f"{np.median(vv)*100:>8.2f}% {np.median(vv)/np.median(sv):>7.2f} "
          f"{p:>9.2e}")
    win = float((vv < sv).mean())
    print(f"\nwindow level (over-powered: windows inside a sequence are not "
          f"independent)\n  {test} lower in {win*100:.0f}% of {len(sv)} windows")
    # The honest degrees of freedom is the number of sequences, not windows.
    k = int(sum(r < 1.0 for r in seq_ratio))
    sp = binomtest(k, len(seq_ratio), 0.5).pvalue
    print(f"sequence level (the honest df)\n  {test} lower in "
          f"{k}/{len(seq_ratio)} sequences, sign test p = {sp:.4f}\n"
          f"  median per-sequence ratio {np.median(seq_ratio):.2f}")
    print("\npre-registered rule (17.36, sharpened round 16): ratio <= 0.50 "
          "with >=6/7 sequences\n  -> green; ratio >= 0.80 or <=4/7 -> red "
          "(prior-bound); between -> yellow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

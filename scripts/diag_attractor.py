"""Did head training move the optimizer's ATTRACTOR, or only its starting point?

Kimi's round-33 measurement, and the one that decides how the head-training
result should be written up.

The context (17.79.1): an injection-time OPACITY prior decays elevenfold under
~2300 online steps -- the optimizer re-derives the opacities the data supports
and the prior washes out. Head training also changes the opacities that enter
the map. If it were the same kind of intervention it should wash out the same
way, and yet the head's deployment benefit survives (-7.9% / -9.9% on desk and
room).

Two possibilities, and they are distinguishable by looking at where the two maps
END rather than where they start:

  same attractor      base and head maps converge to the SAME post-polish
                      distribution. Then the surviving benefit does not live in
                      opacity at all, and the write-up must attribute it to
                      scale/colour instead.
  moved attractor     they exit polish at DIFFERENT distributions. Then head
                      training changed what the data asks for rather than only
                      where the optimizer starts, which is exactly why it
                      persists where an injected prior does not.

Reads the post-polish .ply directly, so it measures the shipped artifact rather
than anything reconstructed.

    python3 scripts/diag_attractor.py A.ply B.ply [labels...]
"""
import sys

import numpy as np
from plyfile import PlyData


def load(path):
    p = PlyData.read(path)["vertex"]
    names = set(p.data.dtype.names)
    op = np.asarray(p["opacity"], dtype=np.float64)
    # stored as logit; the renderer applies a sigmoid
    op = 1.0 / (1.0 + np.exp(-op))
    sc = np.stack([np.asarray(p[f"scale_{i}"], dtype=np.float64) for i in range(3)], 1)
    # stored as log
    sc = np.exp(sc)
    return op, sc, names


def describe(tag, op, sc):
    mx = sc.max(1)
    print(f"{tag:22s} n={len(op):>10,}")
    print(f"{'':22s} opacity  median {np.median(op):.4f}  mean {op.mean():.4f}  "
          f"frac>0.9 {(op > 0.9).mean() * 100:5.1f}%  frac<0.1 {(op < 0.1).mean() * 100:5.1f}%")
    print(f"{'':22s} maxscale median {np.median(mx) * 1000:7.3f} mm  "
          f"p90 {np.percentile(mx, 90) * 1000:8.3f}  p99 {np.percentile(mx, 99) * 1000:9.3f}")
    return np.median(op), (op > 0.9).mean(), np.median(mx), np.percentile(mx, 90)


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".ply")]
    labels = [a for a in sys.argv[1:] if not a.endswith(".ply")]
    if len(paths) < 2:
        print(__doc__)
        return 1
    stats = []
    for i, p in enumerate(paths):
        op, sc, _ = load(p)
        stats.append(describe(labels[i] if i < len(labels) else p.split("/")[-2], op, sc))
        print()
    (om_a, fa, sm_a, sp_a), (om_b, fb, sm_b, sp_b) = stats[0], stats[1]
    print("post-polish difference (B vs A):")
    print(f"  median opacity   {om_a:.4f} -> {om_b:.4f}   ({(om_b - om_a):+.4f})")
    print(f"  frac>0.9         {fa * 100:.1f}% -> {fb * 100:.1f}%   ({(fb - fa) * 100:+.1f} pp)")
    print(f"  median maxscale  {sm_a * 1000:.3f} -> {sm_b * 1000:.3f} mm "
          f"({(sm_b / max(sm_a, 1e-12) - 1) * 100:+.1f}%)")
    print(f"  p90 maxscale     {sp_a * 1000:.3f} -> {sp_b * 1000:.3f} mm "
          f"({(sp_b / max(sp_a, 1e-12) - 1) * 100:+.1f}%)")
    print("\nSAME attractor -> the surviving benefit is not in opacity; attribute "
          "it to scale/colour.\nMOVED attractor -> head training changed what the "
          "data asks for, which is why it\npersists where an injected prior "
          "decays (17.79.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

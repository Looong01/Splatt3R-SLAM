"""Pooled verdict for experiment M: does averaging reach what coupling reached?

Pre-registered in round 17, against the pairwise 5.35% and the joint 1.68% of
§17.45:

    m=4 <= 2.7%   the 1/sqrt(m) model holds -- aggregation is the production
                  answer, on the current backbone
    m=4 >  3.5%   a correlated floor is real -- aggregation caps out
"""
import glob
import os

import numpy as np

VGGT16 = 1.68
PAIRWISE = 5.35


def main():
    rows = {}
    rho = {}
    for p in sorted(glob.glob("logs/expM/*.npz")):
        z = np.load(p, allow_pickle=True)
        seq = str(z["seq"])
        sp = z["spreads"].item()
        st = z["steps"].item()
        rows[seq] = (sp, st)
        b = np.asarray(z["icc_between"], dtype=float)
        w = np.asarray(z["icc_within"], dtype=float)
        if b.size > 4:
            vw = float(w[:, 0].mean())
            nbar = float(w[:, 1].mean())
            vb = max(float(np.var(b, ddof=1)) - vw / nbar, 0.0)
            rho[seq] = vb / max(vb + vw, 1e-12)

    ms = [1, 2, 4, 8]
    print(f"{'sequence':10s} {'n':>3} " +
          " ".join(f"{'m=' + str(m):>8}" for m in ms) + f" {'rho':>6}")
    pooled = {m: [] for m in ms}
    for seq, (sp, st) in sorted(rows.items()):
        n = len(sp.get(1, []))
        cells = []
        for m in ms:
            v = sp.get(m, [])
            cells.append(f"{np.median(v) * 100:>7.2f}%" if len(v) else f"{'-':>8}")
            pooled[m].extend(v)
        print(f"{seq.replace('rgbd_dataset_freiburg1_', ''):10s} {n:>3} "
              + " ".join(cells) + f" {rho.get(seq, float('nan')):>6.3f}")

    print()
    base = np.median(pooled[1]) * 100
    print(f"{'POOLED':10s} {len(pooled[1]):>3} " + " ".join(
        f"{np.median(pooled[m]) * 100:>7.2f}%" if pooled[m] else f"{'-':>8}"
        for m in ms))
    print(f"{'sqrt(m)':10s} {'':>3} " + " ".join(
        f"{base / np.sqrt(m):>7.2f}%" for m in ms))
    if rho:
        r = float(np.median(list(rho.values())))
        print(f"\nmedian rho {r:.3f} over {len(rho)} sequences -> averaging "
              f"cannot go below {np.sqrt(r):.2f} x the m=1 spread "
              f"({base * np.sqrt(r):.2f}%)")
    m4 = np.median(pooled[4]) * 100 if pooled[4] else float("nan")
    verdict = ("AGGREGATION WINS" if m4 <= 2.7 else
               "AGGREGATION CAPS OUT" if m4 > 3.5 else "AMBIGUOUS")
    print(f"\nm=4 pooled {m4:.2f}%   pre-registered lines <=2.70 / >3.50   "
          f"-> {verdict}")
    print(f"reference: pairwise {PAIRWISE}%, joint 16-view {VGGT16}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Does averaging m two-view predictions do what one joint forward does?

§17.45 proved that COUPLING removes the per-cluster scale jitter: one joint
forward over 16 views lands at 1.68% where pairwise prediction sits at 5.35%.
It proved nothing about AVERAGING. Kimi's round-17 lead point, and he is right:
joint attention can enforce cross-view consistency, which is informationally
stronger than cancelling the independent part of m noisy draws. If the draws
carry a component determined by the target view's own content -- the same prior
mistake whoever the partner is -- averaging saturates at that floor while
coupling does not.

The whole aggregation route rests on that difference, so it gets measured
before it gets built.

Model: log r_{k,p} = mu + a_k + e_{k,p}, where a_k is the target view's own
content effect (shared across partners) and e_{k,p} is the per-forward draw.

    rho = Var(a) / (Var(a) + Var(e))          <- the decisive number
    rho ~ 0    draws are independent, spread falls as 1/sqrt(m)
    rho > 0.3  averaging caps out at sqrt(rho) of the m=1 spread

Reported alongside the thing that actually matters: the same scale spread and
neighbour step as `diag_nview_scale.py`, recomputed after aggregating m
predictions per view, so the numbers drop straight into that table.

Aggregation is SENSOR-FREE (per-pixel median over the m predicted depth maps,
which is what production could do); the sensor enters only the scoring.

Pre-registered (round 17), against the pairwise 5.35% and the joint 1.68%:
    m=4 <= 2.7%   the sqrt(m) model holds, aggregation is the production answer
    m=4 >  3.5%   a correlated floor is real, aggregation caps out
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--n-views", type=int, default=16)
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--m-max", type=int, default=8)
    ap.add_argument("--overlap", action="store_true",
                    help="spread windows over the sequence instead of tiling")
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    out_path = os.path.abspath(args.out) if args.out else None
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    load_config(args.config)
    dev = args.device
    root = os.path.abspath(args.dataset)
    ds = load_dataset(root)
    ds_ts = np.array([float(t) for t in ds.timestamps])

    d_ts, d_paths = [], []
    for line in open(os.path.join(root, "depth.txt")):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) == 2:
            d_ts.append(float(p[0])); d_paths.append(p[1])
    d_ts = np.array(d_ts)

    spacing = 0.5
    if args.traj:
        from eval_map_quality import load_tum_traj
        est_ts, _ = load_tum_traj(args.traj)
        if len(est_ts) > 2:
            spacing = float(np.median(np.diff(np.sort(est_ts))))

    span = spacing * (args.n_views - 1)
    if args.overlap:
        # The m-curve is paired WITHIN a window, so overlap costs far less here
        # than it did in diag_nview_scale -- and tiling gave only 1-3 windows
        # per sequence, which is where the m=8 curve started wandering.
        starts = np.linspace(ds_ts[0], max(ds_ts[-1] - span, ds_ts[0]),
                             args.n_windows)
    else:
        n_w = max(1, min(args.n_windows, int((ds_ts[-1] - ds_ts[0]) // span)))
        starts = ds_ts[0] + np.arange(n_w) * span
    windows = []
    for s in starts:
        idx = [int(np.argmin(np.abs(ds_ts - (s + k * spacing))))
               for k in range(args.n_views)]
        if len(set(idx)) == args.n_views:
            windows.append(idx)

    def sensor(i, h, w):
        j = int(np.argmin(np.abs(d_ts - ds_ts[i])))
        if abs(d_ts[j] - ds_ts[i]) > 0.05:
            return None
        d = np.asarray(Image.open(os.path.join(root, d_paths[j])),
                       dtype=np.float32) / 5000.0
        return np.asarray(Image.fromarray(d).resize((w, h), Image.NEAREST))

    from splatt3r_slam.splatt3r_utils import (load_splatt3r,
                                              splatt3r_asymmetric_inference)
    from splatt3r_slam.frame import create_frame
    import lietorch
    model = load_splatt3r(device=dev)
    os.chdir(CORE)
    I = lietorch.Sim3(torch.tensor([0, 0, 0, 0, 0, 0, 1, 1.0],
                                   dtype=torch.float32, device=dev))

    def partners_of(k, n):
        """The m_max nearest other views, nearest first.

        Ranked by |offset| rather than taken at fixed offsets so that EVERY
        view has exactly m partners at every m. With fixed offsets the edge
        views drop out as m grows, which silently changes the view set between
        m levels and breaks the pairing the whole curve depends on.
        """
        return sorted((j for j in range(n) if j != k),
                      key=lambda j: (abs(j - k), j))[:args.m_max]

    ms = [m for m in (1, 2, 4, 8) if m <= args.m_max]
    spreads = {m: [] for m in ms}
    steps = {m: [] for m in ms}
    icc_between, icc_within = [], []
    t0 = time.time()

    for wn, idx in enumerate(windows):
        frames = [create_frame(i, ds.get_image(i), I, img_size=ds.img_size,
                               device=dev) for i in idx]
        n = len(idx)
        # draws[k] = list of this target view's predicted depth maps, one per
        # partner, in nested partner order
        draws, sens = {}, {}
        for k in range(n):
            hw = None
            got = []
            for p in partners_of(k, n):
                with torch.no_grad():
                    X, _, _, _, _ = splatt3r_asymmetric_inference(
                        model, frames[k], frames[p])
                hh, ww = [int(v) for v in frames[k].img_shape.flatten()[:2]]
                got.append(X[0, ..., 2].float().cpu().numpy().reshape(hh, ww))
                hw = (hh, ww)
            if not got:
                continue
            d = sensor(idx[k], *hw)
            if d is None:
                continue
            draws[k] = got
            sens[k] = d
        del frames
        if len(draws) < 4:
            continue

        ok_common = {k: (sens[k] > 0.3) & (sens[k] < 6.0) for k in draws}

        # The decisive number: how much of a draw is the target view's own
        # content (shared by every partner) rather than the forward's lottery.
        # Centred WITHIN the window, because the spread statistic is also
        # within-window -- pooling raw values across windows put the
        # between-window scale differences into Var(target) and inflated rho
        # to 0.70 while the measured curve was falling as 1/sqrt(m).
        per_view = []
        for k in draws:
            m_ok = ok_common[k]
            rs = [np.log(np.median(z[m_ok & (z > 0.05)] /
                                   sens[k][m_ok & (z > 0.05)]))
                  for z in draws[k] if (m_ok & (z > 0.05)).sum() > 2000]
            if len(rs) >= 3:
                per_view.append((np.mean(rs), np.var(rs, ddof=1), len(rs)))
        if len(per_view) >= 4:
            mu = np.median([v[0] for v in per_view])
            for mean_k, var_k, n_k in per_view:
                icc_between.append(mean_k - mu)
                icc_within.append((var_k, n_k))

        for m in ms:
            rk = []
            for k in sorted(draws):
                sub = draws[k][:m]
                if len(sub) < m:
                    continue
                # SENSOR-FREE aggregation: per-pixel median of the m predicted
                # depth maps, which is what production could actually do
                cons = np.median(np.stack(sub), axis=0)
                good = ok_common[k] & (cons > 0.05)
                if good.sum() < 2000:
                    continue
                rk.append((k, float(np.median(cons[good] / sens[k][good]))))
            if len(rk) < 4:
                continue
            ks = np.array([k for k, _ in rk])
            r = np.array([v for _, v in rk])
            spreads[m].append(float(np.median(np.abs(r / np.median(r) - 1.0))))
            adj = np.diff(ks) == 1
            nb = np.abs(r[1:] / r[:-1] - 1.0)[adj]
            if nb.size:
                steps[m].append(float(np.median(nb)))
        print(f"  window {wn + 1}/{len(windows)}  {time.time() - t0:.0f}s",
              flush=True)

    name = os.path.basename(root)
    print(f"\n{name}   {len(spreads[ms[0]])} windows of {args.n_views} views, "
          f"partners = the {args.m_max} nearest views, nearest first")
    print(f"{'m':>3} {'scale spread':>13} {'neighbour step':>15} "
          f"{'sqrt(m) predicts':>17}")
    base = np.median(spreads[ms[0]]) if spreads[ms[0]] else float("nan")
    for m in ms:
        if not spreads[m]:
            continue
        print(f"{m:>3} {np.median(spreads[m])*100:>12.2f}% "
              f"{np.median(steps[m])*100:>14.2f}% "
              f"{base/np.sqrt(m)*100:>16.2f}%")
    if len(icc_between) > 4:
        vw = float(np.mean([v for v, _ in icc_within]))
        nbar = float(np.mean([n for _, n in icc_within]))
        # Var of the per-view MEANS already contains Var(draw)/n_partners;
        # subtracting it is what makes this Var(target) and not a mixture.
        vb = max(float(np.var(icc_between, ddof=1)) - vw / nbar, 0.0)
        print(f"\nrho = Var(target) / (Var(target) + Var(draw)) = "
              f"{vb / max(vb + vw, 1e-12):.3f}"
              f"   [target {vb:.5f}, draw {vw:.5f}, partners {nbar:.1f}]")
        print(f"  floor implied by rho: spread(inf) / spread(1) = "
              f"{np.sqrt(vb / max(vb + vw, 1e-12)):.2f}")
        print("  ~0 -> draws independent, averaging works as 1/sqrt(m); "
              ">0.3 -> capped")
    if out_path:
        np.savez(out_path, seq=name, ms=ms,
                 spreads={m: spreads[m] for m in ms},
                 steps={m: steps[m] for m in ms},
                 icc_between=icc_between, icc_within=np.array(icc_within))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

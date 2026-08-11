"""Depth accuracy as a function of angular parallax: the dose-response curve.

§17.34 measured that the perceptual benefit of refinement is almost perfectly
predicted by a sequence's translation/rotation ratio (Spearman −0.964), and
attributed it to Splatt3R always being fed a temporally adjacent pair
(`tracker.py:29`), which under rotation has almost no baseline. That is a
correlation plus a mechanism story. This is the transfer function underneath it:

    depth error  vs  angular parallax  rho = baseline / median_depth

measured directly, by running the network on keyframe pairs already in the
trajectory that span a wide range of rho, and scoring the predicted depth
against the RGB-D sensor.

Two things it decides:

1. **Where the knee is**, which sets `tau` for the baseline-aware pairing of
   experiment B. Triangulation arithmetic (sigma_d ~ d * sigma_disp / (rho * f),
   with sigma_disp ~ 1 px, f = 517, d = 2 m) predicts 6.5% at rho = 0.03 (which
   is what the current adjacent-pair regime gives, and matches the 9.6% measured
   in §17.17), 1.4% at rho = 0.1, and 0.4% at rho = 0.3 — below other error
   sources, so diminishing returns. Prior: the knee is in rho ∈ [0.1, 0.3].
2. **Whether the mechanism is real at all.** If depth error is flat in rho, the
   correlation in §17.34 has some other cause and baseline-aware pairing cannot
   help.

Unlike §17.29's failed upper-bound arms, the sensor is used here only as a
*yardstick* for a relative comparison between pairs — never injected into the
map — so none of the three confounds that invalidated those arms applies.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from eval_map_quality import associate, load_tum_traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-pairs", type=int, default=120)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import (load_splatt3r, resize_img,
                                              splatt3r_asymmetric_inference)
    from splatt3r_slam.frame import create_frame
    import lietorch

    load_config(args.config)
    dev = args.device
    # ABSOLUTE path: this script chdir()s into splatt3r_core below (the model
    # import needs it), and a dataset opened on a relative path then reads empty
    # images -- cv2 returns None rather than raising, so the failure surfaces
    # much later as an OpenCV assert inside normalize_exposure.
    ds = load_dataset(os.path.abspath(args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(args.traj)

    d_ts, d_paths = [], []
    for line in open(os.path.join(os.path.abspath(args.dataset), "depth.txt")):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) == 2:
            d_ts.append(float(p[0])); d_paths.append(p[1])
    d_ts = np.array(d_ts)

    # Keyframe dataset indices, via the trajectory timestamps (never frame_id --
    # main.py subsamples, and that mismatch has cost this project a retracted
    # result before; see 13.12a).
    kf_di = []
    for t in est_ts:
        j = int(np.argmin(np.abs(ds_ts - t)))
        kf_di.append(j if abs(ds_ts[j] - t) < 0.02 else -1)

    model = load_splatt3r(device=dev)
    os.chdir(CORE)

    def frame_of(i):
        # create_frame does its own resize_img; handing it an already-resized
        # tensor silently produces an unusable frame (this cost one empty run).
        T = lietorch.Sim3(torch.tensor(
            [*est_T[i][:3, 3], 0, 0, 0, 1, 1.0], dtype=torch.float32, device=dev))
        return create_frame(kf_di[i], ds.get_image(kf_di[i]), T,
                            img_size=ds.img_size, device=dev)

    def sensor_depth(i, h, w):
        j = int(np.argmin(np.abs(d_ts - est_ts[i])))
        if abs(d_ts[j] - est_ts[i]) > 0.02:
            return None
        d = np.asarray(Image.open(os.path.join(os.path.abspath(args.dataset)
                                               if os.path.isabs(args.dataset)
                                               else os.path.join(REPO_ROOT, args.dataset),
                                               d_paths[j])),
                       dtype=np.float32) / 5000.0
        return np.asarray(Image.fromarray(d).resize((w, h), Image.NEAREST))

    n = len(est_ts)
    pos = np.array([est_T[i][:3, 3] for i in range(n)])
    rng = np.random.default_rng(0)
    cand = []
    for _ in range(args.max_pairs * 8):
        i, j = rng.integers(0, n, 2)
        if i == j or kf_di[i] < 0 or kf_di[j] < 0:
            continue
        cand.append((int(i), int(j), float(np.linalg.norm(pos[i] - pos[j]))))
    # spread the sample over the available baseline range instead of over pairs,
    # or the dose curve is dominated by whatever spacing the trajectory happens
    # to prefer
    cand.sort(key=lambda x: x[2])
    step = max(1, len(cand) // args.max_pairs)
    cand = cand[::step][:args.max_pairs]

    rows = []
    for i, j, base in cand:
        try:
            fi, fj = frame_of(i), frame_of(j)
            with torch.no_grad():
                X, C, D, Q, _ = splatt3r_asymmetric_inference(model, fi, fj)
        except Exception as e:
            if not rows:
                print(f"  pair ({i},{j}) failed: {type(e).__name__}: {e}",
                      flush=True)
            continue
        hh, ww = [int(v) for v in fi.img_shape.flatten()[:2]]
        # X is (2, h, w, 3): index 0 is view i's own pointmap in view i's frame,
        # which is the quantity the sensor at view i can score.
        z = X[0, ..., 2].float().cpu().numpy().reshape(-1)
        if z.size != hh * ww:
            continue
        d = sensor_depth(i, hh, ww)
        if d is None:
            continue
        zz = z.reshape(hh, ww)
        ok = (d > 0.3) & (d < 6.0) & (zz > 0.05)
        if ok.sum() < 2000:
            continue
        # DIRECTED overlap: what fraction of view i's pixels the pair actually
        # constrains. Random keyframe pairs make baseline and overlap covary --
        # distant pairs see less of each other -- so without this the curve is
        # flat for a reason that has nothing to do with parallax. The network's
        # own confidence on view i is the cheapest available proxy and it is the
        # quantity the Gaussian head is gated on anyway (min_confidence).
        conf = C[0].float().cpu().numpy().reshape(-1)
        overlap = float((conf > 1.5).mean())
        r = zz[ok] / d[ok]
        med_d = float(np.median(d[ok]))
        rho = base / max(med_d, 1e-6)
        # scale-invariant error: the per-pair median ratio is the pair's own
        # scale, which SLAM would absorb; what hurts the map is the SPREAD about
        # it, which is what misplaces one cluster relative to another.
        rel = np.median(np.abs(r / np.median(r) - 1.0))
        rows.append((rho, float(rel), float(np.median(r)), base, med_d, overlap))

    if not rows:
        print("no usable pairs"); return 1
    a = np.array(rows)
    order = np.argsort(a[:, 0])
    a = a[order]
    print(f"\n{len(a)} pairs")
    print(f"{'rho bin':>14} {'n':>4} {'depth spread':>13} {'median ratio':>13}")
    edges = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 10]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum() == 0:
            continue
        print(f"{lo:6.3f}-{hi:6.3f} {int(m.sum()):>4} "
              f"{np.median(a[m, 1]) * 100:>12.2f}% {np.median(a[m, 2]):>13.4f}")
    from scipy.stats import spearmanr
    print(f"\n{'overlap bin':>14} {'n':>4} {'depth spread':>13} {'median rho':>11}")
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, 1.01)):
        m = (a[:, 5] >= lo) & (a[:, 5] < hi)
        if m.sum():
            print(f"{lo:6.2f}-{hi:6.2f} {int(m.sum()):>4} "
                  f"{np.median(a[m,1])*100:>12.2f}% {np.median(a[m,0]):>11.3f}")
    print(f"\nspearman(rho, depth spread)      = {spearmanr(a[:,0], a[:,1]).statistic:+.3f}")
    print(f"spearman(overlap, depth spread)  = {spearmanr(a[:,5], a[:,1]).statistic:+.3f}")
    print(f"spearman(rho, overlap)           = {spearmanr(a[:,0], a[:,5]).statistic:+.3f}"
          f"   <- if strongly negative, rho and overlap are confounded")
    hi_ov = a[a[:, 5] > np.median(a[:, 5])]
    if len(hi_ov) > 6:
        print(f"\nWITHIN high-overlap pairs only (n={len(hi_ov)}): "
              f"spearman(rho, spread) = {spearmanr(hi_ov[:,0], hi_ov[:,1]).statistic:+.3f}")
    print("A negative correlation is the mechanism of 17.34; flat means the "
          "pairing story is wrong and baseline-aware pairing cannot help.")
    np.save(os.path.join(REPO_ROOT, "logs/parallax_dose.npy"), a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

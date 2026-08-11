"""Experiment A: does a joint N-view prediction remove the per-pair scale jitter?

§17.36 measured Splatt3R's **per-pair scale error** at 9.10% (cross-checking
§17.17's 9.6% by a different path) and showed it is flat in parallax -- a
conditional bias that depends on the pair, not on the pair's geometry. §17.41
then closed every map-side route to the seams. What is left is the backbone:
an N-view model predicts all views in one forward and by construction cannot
have per-pair *independent* jitter. If VGGT's within-cluster scale spread is far
below 9%, it targets the parent of the seams; if it is also ~9%, the bias is
prior-bound and no backbone helps.

The design is matched to the point of sharing this file, because the failure
mode this section keeps recording is a comparison whose two sides were not
measuring the same thing:

  * same frames, same windows, same sensor, same masks, same statistic;
  * the ONLY difference between arms is joint (VGGT, one forward over N views)
    versus pairwise (Splatt3R, N forwards over temporally adjacent pairs, which
    is exactly what `tracker.py` feeds it in production);
  * the statistic is **pose-free**: predicted depth in each view's own camera
    frame against that view's RGB-D sensor. No trajectory, no alignment, no
    Umeyama -- so SLAM drift cannot enter the number in either arm.

Three quantities per window of N views (r_k = median over pixels of z_k / d_k):

    scale spread   median_k |r_k / median(r) - 1|     <- 9.10% for Splatt3R
    neighbour step median_k |r_{k+1} / r_k - 1|       <- what a seam sees
    within-view    median_k median_pix |ratio / r_k - 1|   <- the shape floor,
                                                            ~3% for Splatt3R

Both arms are gauge-free: normalizing by the window median removes VGGT's
unknown global scale and Splatt3R's metric bias alike.

Usage:
    python3 scripts/diag_nview_scale.py --arm vggt --dataset ... --out x.npz
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


def load_depth_index(root):
    """(timestamps, paths, metres-per-unit).

    TUM ships depth.txt and uint16/5000. Replica (NICE-SLAM release) ships no
    index at all -- depth is results/depth%06d.png alongside the frames, one per
    frame, scaled by 6553.5 (cam_params.json). Getting the scale wrong would not
    change any statistic here (they are all ratios normalized per window), but
    it would make the printed median ratio meaningless, so it is read rather
    than assumed.
    """
    if not os.path.exists(os.path.join(root, "depth.txt")):
        import glob as _g
        import json as _j
        paths = sorted(_g.glob(os.path.join(root, "results", "depth*.png")))
        scale = 6553.5
        cj = os.path.join(os.path.dirname(root), "cam_params.json")
        if os.path.exists(cj):
            scale = float(_j.load(open(cj))["camera"].get("scale", scale))
        return (np.arange(len(paths), dtype=float),
                [os.path.relpath(p, root) for p in paths], scale)
    ts, paths = [], []
    for line in open(os.path.join(root, "depth.txt")):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) == 2:
            ts.append(float(p[0]))
            paths.append(p[1])
    return np.array(ts), paths, 5000.0


def window_stats(scored, n_views, min_frac=0.75):
    """scored: list of (view index k, ratio array z/d on valid pixels).

    Views are dropped individually rather than voiding the window -- one view
    whose depth frame is missing must not cost the other fifteen. The
    neighbour step is then taken only over pairs that are still adjacent.
    """
    if len(scored) < max(2, int(round(min_frac * n_views))):
        return None
    ks = np.array([k for k, _ in scored])
    r = np.array([np.median(x) for _, x in scored])
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        return None
    med = np.median(r)
    scale_spread = float(np.median(np.abs(r / med - 1.0)))
    adj = np.diff(ks) == 1
    nb = np.abs(r[1:] / r[:-1] - 1.0)[adj]
    neighbour = float(np.median(nb)) if nb.size else float("nan")
    within = float(np.median([np.median(np.abs(x / np.median(x) - 1.0))
                              for _, x in scored]))
    return scale_spread, neighbour, within, float(med), float(len(scored))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("vggt", "splatt3r"), required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--traj", default=None,
                    help="SLAM trajectory, used only to copy the real keyframe "
                         "spacing; falls back to --spacing")
    ap.add_argument("--n-views", type=int, default=16)
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--spacing", type=float, default=0.0,
                    help="seconds between views in a window; 0 = from --traj")
    ap.add_argument("--batch", type=int, default=4,
                    help="vggt only: windows per forward, halved on OOM")
    ap.add_argument("--pair-mode", action="store_true",
                    help="vggt only: predict each view from a 2-view forward "
                         "with its temporal neighbour, instead of one joint "
                         "forward over the window. This is the control that "
                         "separates 'joint context' from 'different model' -- "
                         "without it a low VGGT spread could be either.")
    ap.add_argument("--no-overlap", action="store_true",
                    help="tile windows by one full span instead of spreading "
                         "them, so the windows are actually independent")
    ap.add_argument("--conf-keep", type=float, default=0.7,
                    help="fraction of pixels kept by each model's own "
                         "confidence, matched across arms by FRACTION")
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head", default=None,
                    help="load a fine-tuned Gaussian head on top of the base "
                         "checkpoint. Kimi round 23: head-only training "
                         "optimizes PER-PAIR render quality, which can improve "
                         "renders while de-calibrating CROSS-PAIR scale -- the "
                         "candidate explanation for 17.55's baked regression.")
    ap.add_argument("--density", action="store_true",
                    help="also report predicted-Gaussian density, mean "
                         "confidence and the fraction passing the gate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    # Resolved before anything can chdir: the splatt3r arm moves into
    # splatt3r_core to import the model, so a relative --out silently pointed at
    # a directory that does not exist there. It failed AFTER printing the
    # summary, which made a run with no saved output look complete.
    out_path = os.path.abspath(args.out) if args.out else None
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset

    load_config(args.config)
    dev = args.device
    root = os.path.abspath(args.dataset)
    ds = load_dataset(root)
    ds_ts = np.array([float(t) for t in ds.timestamps])
    d_ts, d_paths, d_scale = load_depth_index(root)

    spacing = args.spacing
    if spacing <= 0 and args.traj:
        from eval_map_quality import load_tum_traj
        est_ts, _ = load_tum_traj(args.traj)
        spacing = float(np.median(np.diff(np.sort(est_ts)))) if len(est_ts) > 2 else 0.5
    if spacing <= 0:
        spacing = 0.5

    # Windows of N frames at the production keyframe spacing, spread over the
    # whole sequence. Frames come from the dataset, not from the SLAM keyframe
    # list, because the question is about the network and the head-run
    # trajectories have too few keyframes to form even one window.
    span = spacing * (args.n_views - 1)
    t0, t1 = ds_ts[0], ds_ts[-1] - span
    if t1 <= t0:
        print(f"sequence too short for {args.n_views} views at {spacing:.3f}s")
        return 1
    if args.no_overlap:
        # At keyframe spacing a 16-view window covers most of a TUM fr1
        # sequence, so `linspace` over what is left puts 12 windows on top of
        # each other -- measured at 80-97% overlap, which makes the window-level
        # n a fiction. Tiling by exactly one span gives independent windows.
        n_w = max(1, min(args.n_windows, int((ds_ts[-1] - ds_ts[0]) // span)))
        starts = t0 + np.arange(n_w) * span
    else:
        starts = np.linspace(t0, t1, args.n_windows)
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
                       dtype=np.float32) / d_scale
        return np.asarray(Image.fromarray(d).resize((w, h), Image.NEAREST))

    # Kimi's round-16 check: the window-median normalization absorbs a global
    # SCALE but not a global SHIFT, and a shift would silently distort every
    # ratio in this file. Fit z = a*d + b on the first window's views and report
    # the shift as a fraction of the median depth; it must be ~0.
    shift_probe = []
    depth_probe = []

    def score(z, conf, di):
        """z, conf: (h, w) numpy. Returns the per-pixel ratio on valid pixels."""
        h, w = z.shape
        d = sensor(di, h, w)
        if d is None:
            return None
        ok = (d > 0.3) & (d < 6.0) & (z > 0.05) & np.isfinite(z)
        if args.conf_keep < 1.0 and conf is not None:
            thr = np.quantile(conf[ok], 1.0 - args.conf_keep) if ok.sum() else 0
            ok &= conf >= thr
        if ok.sum() < 2000:
            return None
        zz, dd = z[ok].astype(np.float64), d[ok].astype(np.float64)
        if len(shift_probe) < args.n_views:
            a, b = np.polyfit(dd, zz, 1)
            shift_probe.append(b / max(np.median(zz), 1e-9))
        # If the prediction compresses the depth range (slope < 1 with a
        # positive offset), then r_k depends on how far away the CONTENT of
        # view k happens to be -- a deterministic response to the scene, not
        # per-pair jitter. Collected so the two can be told apart afterwards.
        depth_probe.append((float(np.median(dd)), float(np.median(zz / dd))))
        return zz / dd

    rows, dens = [], []
    t_start = time.time()

    if args.arm == "vggt":
        from vggt.models.vggt import VGGT
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
        amp = torch.bfloat16

        # 640x480 -> 518x392: width 518, height rounded to a multiple of 14,
        # which is VGGT's own "crop" preprocessing and involves no cropping at
        # 4:3 (392 < 518). Aspect is preserved to 0.9%, and since the sensor is
        # resampled to the same grid the comparison stays pixel-aligned.
        H, W = 392, 518

        def load_window(idx):
            ims = []
            for i in idx:
                im = ds.get_image(i)  # HWC float [0,1], undistorted
                im = np.asarray(Image.fromarray(np.uint8(im * 255)).resize(
                    (W, H), Image.BICUBIC), dtype=np.float32) / 255.0
                ims.append(torch.from_numpy(im).permute(2, 0, 1))
            return torch.stack(ims)

        b = max(1, args.batch)
        w_i = 0
        while w_i < len(windows):
            chunk = windows[w_i:w_i + b]
            try:
                ims = torch.stack([load_window(x) for x in chunk]).to(dev)
                if args.pair_mode:
                    # (B, S, ...) -> (B*S, 2, ...): view k paired with its
                    # temporal neighbour, one independent forward each, so the
                    # only thing removed relative to the joint arm is the shared
                    # context. Batched over every pair in every window at once.
                    B, S = ims.shape[:2]
                    p = torch.arange(S, device=ims.device) + 1
                    p[-1] = S - 2
                    pairs = torch.stack([ims, ims[:, p]], dim=2)
                    with torch.no_grad(), torch.autocast(dev, dtype=amp):
                        out = model(pairs.reshape(B * S, 2, *ims.shape[2:]))
                    depth = out["depth"][:, 0, ..., 0].float().cpu().numpy()
                    dconf = out["depth_conf"][:, 0].float().cpu().numpy()
                    depth = depth.reshape(B, S, *depth.shape[1:])
                    dconf = dconf.reshape(B, S, *dconf.shape[1:])
                    del pairs
                else:
                    with torch.no_grad(), torch.autocast(dev, dtype=amp):
                        out = model(ims)
                    depth = out["depth"][..., 0].float().cpu().numpy()
                    dconf = out["depth_conf"].float().cpu().numpy()
                del out, ims
            except RuntimeError as e:
                # torch.OutOfMemoryError is not always the class that reaches
                # here (it varies with where in the DPT head the allocation
                # fails), so match on the message and re-raise anything else.
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                if b == 1:
                    raise
                b = max(1, b // 2)
                print(f"  OOM -> batch {b}", flush=True)
                continue
            for bi, idx in enumerate(chunk):
                scored = []
                for k, di in enumerate(idx):
                    r = score(depth[bi, k], dconf[bi, k], di)
                    if r is not None:
                        scored.append((k, r))
                st = window_stats(scored, args.n_views)
                if st:
                    # the window index is carried so the two arms can be paired
                    # window-by-window: both arms score the SAME frames, so the
                    # comparison is a paired test, not two independent medians
                    rows.append((w_i + bi,) + st)
            w_i += len(chunk)
            print(f"  {w_i}/{len(windows)} windows  batch={b}  "
                  f"{time.time() - t_start:.0f}s", flush=True)
        peak = torch.cuda.max_memory_allocated() / 2**30

    else:
        from splatt3r_slam.splatt3r_utils import (load_splatt3r,
                                                  splatt3r_asymmetric_inference)
        from splatt3r_slam.frame import create_frame
        import lietorch
        model = load_splatt3r(device=dev)
        if args.head:
            h = args.head if os.path.isabs(args.head) else os.path.join(
                REPO_ROOT, args.head)
            sd = torch.load(h, map_location=dev)
            missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
            assert not unexpected, f"unexpected keys: {list(unexpected)[:3]}"
            assert not [k for k in missing if "gaussian_dpt" in k], "head key missing"
            print(f"loaded head {os.path.basename(h)}", flush=True)
        os.chdir(CORE)
        I = lietorch.Sim3(torch.tensor([0, 0, 0, 0, 0, 0, 1, 1.0],
                                       dtype=torch.float32, device=dev))

        for wn, idx in enumerate(windows):
            # One frame object per view, reused across both pairs it takes part
            # in -- `splatt3r_asymmetric_inference` caches `frame.feat`, so the
            # encoder runs once per view instead of twice.
            frames = [create_frame(i, ds.get_image(i), I, img_size=ds.img_size,
                                   device=dev) for i in idx]
            scored = []
            for k in range(len(idx)):
                # production pairing: the temporally adjacent keyframe
                p = k + 1 if k + 1 < len(idx) else k - 1
                try:
                    with torch.no_grad():
                        X, C, _, _, _ = splatt3r_asymmetric_inference(
                            model, frames[k], frames[p])
                except Exception as e:
                    print(f"  window {wn} view {k} failed: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                hh, ww = [int(v) for v in frames[k].img_shape.flatten()[:2]]
                z = X[0, ..., 2].float().cpu().numpy().reshape(hh, ww)
                c = C[0].float().cpu().numpy().reshape(hh, ww)
                if args.density:
                    dens.append((float(c.mean()),
                                 float((c > 1.5).mean())))
                r = score(z, c, idx[k])
                if r is not None:
                    scored.append((k, r))
            del frames
            st = window_stats(scored, args.n_views)
            if st:
                rows.append((wn,) + st)
            print(f"  {wn + 1}/{len(windows)} windows  "
                  f"{time.time() - t_start:.0f}s", flush=True)
        peak = torch.cuda.max_memory_allocated() / 2**30

    if not rows:
        print("no usable windows")
        return 1
    a = np.array(rows)
    name = os.path.basename(root)
    print(f"\n{args.arm:9s} {name}  {len(a)} windows of {args.n_views} views "
          f"@ {spacing:.3f}s   peak {peak:.1f} GiB   {time.time()-t_start:.0f}s")
    print(f"  scale spread   {np.median(a[:,1])*100:6.2f}%   "
          f"[{np.percentile(a[:,1],25)*100:.2f}, {np.percentile(a[:,1],75)*100:.2f}]")
    print(f"  neighbour step {np.median(a[:,2])*100:6.2f}%   "
          f"[{np.percentile(a[:,2],25)*100:.2f}, {np.percentile(a[:,2],75)*100:.2f}]")
    print(f"  within-view    {np.median(a[:,3])*100:6.2f}%")
    print(f"  median ratio   {np.median(a[:,4]):6.4f}  "
          f"(vggt is gauge-free, splatt3r's is the metric bias)")
    print(f"  views used     {np.median(a[:,5]):.1f} of {args.n_views}")
    if dens:
        d = np.array(dens)
        print(f"  mean conf      {d[:,0].mean():.3f}   "
              f"fraction above gate {d[:,1].mean()*100:.1f}%   (n={len(d)} views)")
    if shift_probe:
        print(f"  shift probe    {np.median(shift_probe)*100:+6.2f}% of median "
              f"depth   (nonzero = the model compresses the depth range)")
    if len(depth_probe) > 8:
        from scipy.stats import spearmanr
        dp = np.array(depth_probe)
        print(f"  r_k vs view depth  spearman {spearmanr(dp[:,0], dp[:,1]).statistic:+.3f}"
              f"   (strongly negative = the spread is a scene response, "
              f"not per-pair jitter)")
    ov = 1.0 - (starts[1] - starts[0]) / span if len(starts) > 1 else 0.0
    print(f"  window overlap {max(ov, 0.0)*100:5.1f}%   "
          f"(non-zero means the windows are not independent)")
    if out_path:
        np.savez(out_path, rows=a, arm=args.arm, seq=name,
                 n_views=args.n_views, spacing=spacing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

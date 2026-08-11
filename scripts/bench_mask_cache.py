"""Lever 4: per-view mask caching -- how big is the set, and is it stable?

Two questions the throughput claim in skill 16.11 rests on, neither of which
had been measured when that section was written:

1. **Which signal?** 16.11 proposed `radii > 0`. That is written in the
   rasterizer's preprocess from the frustum test and the projected extent, so
   it is blind to occlusion -- and occlusion is precisely what the geometric
   cull (lever 1) already fails to represent, the reason it stops at 2.1x
   against a 5.5x ceiling. The alternative is the true gradient support: after
   a backward, exactly the Gaussians that contributed colour to a pixel have a
   nonzero `f_dc` gradient. If `radii > 0` and the gradient support are the
   same size, 16.11 was right by accident; if `radii > 0` is much larger, the
   design has to change and the projected 2.6x is wrong.

2. **Is it stable?** Caching is only sound if a view's support barely moves
   between visits. 16.11 justified it with per-view masks being disjoint
   ACROSS views (IoU 0.000) -- which says caching does not thrash between
   views, but says nothing about drift within one view over training. What is
   actually needed is the overlap of one view's support with ITSELF, N
   optimizer steps later. That is what decides the refresh interval, and it is
   measured here.

Also reports the plain sizes (map, culled, radii>0, gradient support) so the
attainable speedup can be computed rather than assumed.

Usage:
    python3 scripts/bench_mask_cache.py \
        --kfgauss logs/frames_head/rgbd_dataset_freiburg1_360_kfgauss.pt \
        --traj logs/frames_head/rgbd_dataset_freiburg1_360.txt \
        --frames-traj logs/frames_head/rgbd_dataset_freiburg1_360_frames.txt \
        --dataset datasets/tum/rgbd_dataset_freiburg1_360
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

from eval_map_quality import associate, load_tum_traj, umeyama_sim3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames-traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--aa-sigma", type=float, default=0.0,
                    help="build the map with the 3D smoothing filter before "
                         "testing the cull. The false-negative check in 17.1 "
                         "was run on the unfiltered map, and the filter shifts "
                         "the scale distribution the footprint bound depends "
                         "on, so it does not carry over untested.")
    ap.add_argument("--warm-steps", type=int, default=0,
                    help="optimizer steps to run BEFORE the size/false-negative "
                         "table, so the cull is checked against a refined map "
                         "and not only against the baked one")
    ap.add_argument("--cull", type=int, default=4)
    ap.add_argument("--drift-steps", type=int, default=200,
                    help="optimizer steps between the two observations of a "
                         "view, for the self-overlap measurement")
    ap.add_argument("--n-probe", type=int, default=8,
                    help="views to measure sizes and drift on")
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.refiner import (LocalGaussianMap, sim3_to_mat,
                                       gaussians_from_keyframe, render_map,
                                       _optimizer_for, _gaussian_window, _ssim)
    from refine_gaussian_map import uniform_subsample

    load_config(args.config)
    os.chdir(CORE)
    dev = args.device
    torch.manual_seed(0)

    ds = load_dataset(os.path.join(REPO_ROOT, args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(os.path.join(REPO_ROOT, args.traj))
    frm_ts, frm_T = load_tum_traj(os.path.join(REPO_ROOT, args.frames_traj))

    fr_pairs = associate(frm_ts, ds_ts)
    sel = uniform_subsample(fr_pairs, args.n_train)
    frames = []
    for fi, di in sel:
        img = resize_img(ds.get_image(di), ds.img_size)["img"]
        tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
        frames.append((frm_T[fi], tgt))

    blob = torch.load(os.path.join(REPO_ROOT, args.kfgauss), map_location="cpu")
    parts, kf_pose_data = [], []
    for k, kf in enumerate(blob["keyframes"]):
        h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
        local = {key: kf[key].to(dev) for key in
                 ("means", "scales", "rotations", "sh", "opacities", "conf")}
        got = gaussians_from_keyframe(local, kf["img"].to(dev), h, w, k, dev,
                                      min_confidence=args.min_confidence,
                                      aa_sigma_scale=args.aa_sigma)
        if got is None:
            continue
        parts.append(got)
        kf_pose_data.append(kf["T_WC"].to(dev))
    kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in kf_pose_data]))
    model = LocalGaussianMap(*[torch.cat([p[i] for p in parts])
                               for i in range(6)]).to(dev)
    extent = float((model.means.max(0).values - model.means.min(0).values).norm() / 2)
    K = torch.as_tensor(ds.camera_intrinsics.K_frame, dtype=torch.float32, device=dev)
    print(f"map {model.n:,} gaussians, {kf_mats.shape[0]} keyframes, "
          f"{len(frames)} views, extent {extent:.2f} m", flush=True)

    opt = _optimizer_for(model, extent)
    win = _gaussian_window(device=dev)

    def step(c2w, tgt, idx, want_radii=False):
        """One full optimizer step on a submitted subset. Returns (support,
        radii_mask) as GLOBAL boolean masks."""
        mw, cw = model.world(kf_mats, idx)
        h, w = tgt.shape[-2:]
        if want_radii:
            from src.pixelsplat_src.cuda_splatting import render_cuda
            from splatt3r_slam.refiner import C0
            ext = torch.as_tensor(c2w, dtype=torch.float32, device=dev)[None]
            intr = K[None].clone()
            intr[:, 0, :] /= w
            intr[:, 1, :] /= h
            rgb_, op_ = model.rgb(idx), model.opacity(idx)
            img, radii, _ = render_cuda(
                ext, intr, torch.full((1,), 0.01, device=dev),
                torch.full((1,), 100.0, device=dev), (h, w),
                torch.zeros((1, 3), device=dev), mw[None], cw[None],
                ((rgb_ - 0.5) / C0)[:, :, None][None], op_.reshape(-1)[None],
                use_sh=True, return_extras=True)
            pred = img.reshape(1, 3, h, w)
            rad = radii.reshape(-1) > 0
        else:
            pred = render_map(mw, cw, model.rgb(idx), model.opacity(idx),
                              c2w, K, (h, w), dev)
            rad = None
        loss = (0.8 * (pred - tgt).abs().mean()
                + 0.2 * (1 - _ssim(pred.clamp(0, 1), tgt, win)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = model.f_dc.grad.detach().clone()
        sup = grad.abs().sum(-1) > 0
        opt.step()

        def to_global(local_mask):
            if local_mask is None:
                return None
            if idx is None:
                return local_mask
            g = torch.zeros(model.n, dtype=torch.bool, device=dev)
            g[idx] = local_mask
            return g
        return sup, to_global(rad), grad

    if args.warm_steps:
        rng0 = np.random.default_rng(1)
        for _ in range(args.warm_steps):
            vi = int(rng0.integers(len(frames)))
            c2w, tgt = frames[vi]
            step(c2w, tgt, model.visible_exact(kf_mats, c2w, K, tgt.shape[-2:]))
        print(f"warmed {args.warm_steps} steps before the cull check", flush=True)

    # --- Q1: how big is each candidate set? ---
    print("\n--- set sizes (fraction of the map) ---", flush=True)
    print(f"{'view':>5} {'block':>8} {'exact':>8} {'radii>0':>8} {'grad sup':>9} "
          f"{'miss':>7} {'miss|grad|':>9} {'t_block':>7} {'t_exact':>7}", flush=True)
    probe = list(range(0, len(frames), max(1, len(frames) // args.n_probe)))[:args.n_probe]
    first = {}
    for vi in probe:
        c2w, tgt = frames[vi]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx = model.visible_subset(kf_mats, c2w, K, tgt.shape[-2:], tiles=args.cull)
        torch.cuda.synchronize()
        t_block = time.perf_counter() - t0
        t0 = time.perf_counter()
        ex = model.visible_exact(kf_mats, c2w, K, tgt.shape[-2:])
        torch.cuda.synchronize()
        t_exact = time.perf_counter() - t0
        n_cull = model.n if idx is None else idx.numel()
        n_ex = model.n if ex is None else ex.numel()
        sup, rad, last_grad = step(c2w, tgt, idx, want_radii=True)
        first[vi] = sup.clone()
        # The only test that matters for a cull: does it drop anything that
        # carries gradient? A false positive costs throughput; a false negative
        # silently removes a term from the loss.
        ex_mask = torch.ones(model.n, dtype=torch.bool, device=dev)
        if ex is not None:
            ex_mask = torch.zeros(model.n, dtype=torch.bool, device=dev)
            ex_mask[ex] = True
        miss = int((sup & ~ex_mask).sum())
        # Counting missed Gaussians overstates the harm if they are all
        # near-zero-gradient border cases, so report the share of the total
        # gradient magnitude they carry as well -- that is what the optimizer
        # would actually lose.
        gmag = last_grad.abs().sum(-1)
        gshare = float(gmag[sup & ~ex_mask].sum() / gmag.sum().clamp_min(1e-20))
        print(f"{vi:>5} {n_cull / model.n:>7.1%} {n_ex / model.n:>7.1%} "
              f"{rad.float().mean():>7.1%} {sup.float().mean():>8.1%} "
              f"{miss:>7,} {gshare:>9.2e} {t_block * 1e3:>7.1f} "
              f"{t_exact * 1e3:>7.1f}", flush=True)

    # --- throughput of each cull, same steps, same seed ---
    print("\n--- steps/s by cull ---", flush=True)
    for name in ("none", "block", "exact"):
        rng = np.random.default_rng(0)
        n = args.drift_steps if name != "none" else max(args.drift_steps // 4, 10)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            vi = int(rng.integers(len(frames)))
            c2w, tgt = frames[vi]
            if name == "none":
                idx = None
            elif name == "block":
                idx = model.visible_subset(kf_mats, c2w, K, tgt.shape[-2:],
                                           tiles=args.cull)
            else:
                idx = model.visible_exact(kf_mats, c2w, K, tgt.shape[-2:])
            step(c2w, tgt, idx)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  {name:>6}: {n / dt:.2f} it/s  ({dt / n * 1e3:.0f} ms/step, "
              f"{n} steps)", flush=True)

    # --- Q2: does one view's support drift while the map trains? ---
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(args.drift_steps):
        vi = int(rng.integers(len(frames)))
        c2w, tgt = frames[vi]
        idx = model.visible_subset(kf_mats, c2w, K, tgt.shape[-2:], tiles=args.cull)
        step(c2w, tgt, idx)
    dt = time.perf_counter() - t0
    print(f"\n{args.drift_steps} culled steps in {dt:.1f} s "
          f"({args.drift_steps / dt:.2f} it/s)", flush=True)

    print(f"\n--- support self-overlap after {args.drift_steps} steps ---",
          flush=True)
    print(f"{'view':>5} {'n_before':>10} {'n_after':>10} {'IoU':>7} "
          f"{'recall':>7}  (recall = kept | needed)", flush=True)
    recalls = []
    for vi in probe:
        c2w, tgt = frames[vi]
        idx = model.visible_subset(kf_mats, c2w, K, tgt.shape[-2:], tiles=args.cull)
        sup, _, _ = step(c2w, tgt, idx)
        a, b = first[vi], sup
        inter = int((a & b).sum())
        union = int((a | b).sum())
        # The number that matters is not IoU but RECALL of the cached set
        # against the set now needed: a Gaussian in the cache that no longer
        # contributes costs a little throughput, one that contributes and is
        # missing loses gradient silently.
        rec = inter / max(int(b.sum()), 1)
        recalls.append(rec)
        print(f"{vi:>5} {int(a.sum()):>10,} {int(b.sum()):>10,} "
              f"{inter / max(union, 1):>7.3f} {rec:>7.3f}", flush=True)
    print(f"\nmean recall of a {args.drift_steps}-step-stale cache: "
          f"{sum(recalls) / len(recalls):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

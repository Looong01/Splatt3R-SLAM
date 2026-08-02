"""Causal replay — item (f) of the online-refinement plan: does the measured
refinement gain survive the causality constraint of a live system?

Every gain number in the skill (desk 12.44 -> 14.45 at 50 views / 3000 iters)
was measured POST HOC: the optimizer saw supervision drawn from the whole
sequence from the first iteration. Online, at the moment keyframe k is created
only supervision up to k exists, and early Gaussians are optimized long before
later views arrive. If the gain evaporates under that constraint, process
integration / densification / buffer work (a)-(e) is wasted.

This script replays the trajectory offline under the causal constraint and
compares against a post-hoc control arm run in the SAME harness:

  --mode posthoc   full map from iteration 1, all supervision views available
                   from the start, single Adam for the whole run. Replicates
                   refine_local.py --stage 2; must reproduce its ~14.38 final
                   held-out psnr (regression check for this harness).
  --mode causal    keyframes are injected at their trajectory timestamps;
                   a supervision view becomes usable only once its own dataset
                   timestamp has passed; optimization happens in rounds between
                   keyframe arrivals with iterations allocated in proportion to
                   elapsed sequence time (a live refiner runs at a roughly
                   constant rate). The optimizer is rebuilt on every injection
                   with refiner._optimizer_for, the same Adam-moment carry-over
                   the online loop uses.

Held constant across arms -- the variables under test are ONLY the timing of
supervision availability and Gaussian injection:

  * same map (same kfgauss dump), same held-out views scored at mapped
    ground-truth poses, same supervision view SET (uniform_subsample of the
    same candidate pool), same total iteration budget, same loss/LRs;
  * extent for the means LR is computed from the FULL map's means in both
    arms. run_refiner computes it once from whatever keyframes exist at first
    creation; that is a real online behaviour but it would confound the causal
    variable with a learning-rate difference, so it is held fixed here and
    flagged as a run_refiner design note instead.

One deliberate simplification: time is quantized at keyframe arrivals -- within
round k every view with timestamp <= the round's END is treated as available.
This is optimistic by at most one inter-keyframe interval (~1.3 s on desk, ~7%
of the sequence) and uniform across rounds; the alternative (round-start
availability) silently drops every view that arrives after the LAST keyframe,
which is an artifact of the quantization rather than a property of causality.

Usage:
    python3 scripts/refine_causal.py --mode causal \\
        --kfgauss logs/frames_head/rgbd_dataset_freiburg1_desk_kfgauss.pt \\
        --traj logs/frames_head/rgbd_dataset_freiburg1_desk.txt \\
        --frames-traj logs/frames_head/rgbd_dataset_freiburg1_desk_frames.txt \\
        --dataset datasets/tum/rgbd_dataset_freiburg1_desk
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

# Importing refine_local performs its sys.path setup (splatt3r_core, mast3r,
# pixelsplat) and pulls in eval_map_quality; the render/loss helpers are
# reused verbatim so both arms share refine_local.py's exact code path.
from refine_local import render, gaussian_window, ssim, DSSIM_WEIGHT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("posthoc", "causal"), required=True)
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames-traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--n-held", type=int, default=50)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampling", choices=("uniform", "recent", "mixed"), default="uniform",
                    help="causal mode only: how to draw supervision from the "
                         "views available so far. 'uniform' == an unbounded "
                         "reservoir (on desk's 50 views the reservoir never "
                         "evicts, so this IS the reservoir arm). 'recent' = "
                         "the last --recent-window arrivals only. 'mixed' = "
                         "70%% recent / 30%% older, the SupervisionFrames "
                         "default. Item (e): the window length and mix "
                         "currently have no principled value.")
    ap.add_argument("--recent-window", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    import lpips as lpips_lib
    from eval_map_quality import associate, load_tum_traj, umeyama_sim3
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.refiner import (
        LocalGaussianMap, _optimizer_for, gaussians_from_keyframe, sim3_to_mat)
    from refine_gaussian_map import uniform_subsample

    load_config(args.config)
    CORE = os.path.join(REPO_ROOT, "splatt3r_core")
    os.chdir(CORE)
    dev = args.device
    torch.manual_seed(args.seed)

    # --- data loading: byte-identical to refine_local.py ---
    ds = load_dataset(os.path.join(REPO_ROOT, args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(os.path.join(REPO_ROOT, args.traj))
    frm_ts, frm_T = load_tum_traj(os.path.join(REPO_ROOT, args.frames_traj))
    gt_ts, gt_T = load_tum_traj(os.path.join(REPO_ROOT, args.dataset, "groundtruth.txt"))

    pairs = associate(est_ts, gt_ts)
    s_, R_, t_ = umeyama_sim3(np.array([est_T[i, :3, 3] for i, _ in pairs]),
                              np.array([gt_T[j, :3, 3] for _, j in pairs]))
    Rt = R_.T

    def to_map(c2w_gt):
        m = np.eye(4)
        m[:3, :3] = Rt @ c2w_gt[:3, :3]
        m[:3, 3] = Rt @ (c2w_gt[:3, 3] - t_) / s_
        return m

    kf_idx = {j for _, j in associate(est_ts, ds_ts)}
    gt_pairs = associate(ds_ts, gt_ts)
    non_kf = [(i, j) for i, j in gt_pairs if i not in kf_idx]
    held_c = non_kf[:: max(1, len(non_kf) // args.n_held)][: args.n_held]
    held_set = {i for i, _ in held_c}

    fr_pairs = associate(frm_ts, ds_ts)
    cand = [(fi, di) for fi, di in fr_pairs if di not in held_set]
    sel = uniform_subsample(cand, args.n_train)

    def load_view(item, mapped):
        a, b = item
        di = a if mapped else b
        img = resize_img(ds.get_image(di), ds.img_size)["img"]
        tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
        return (to_map(gt_T[b]) if mapped else frm_T[a], tgt)

    # Supervision at SLAM-estimated poses (deployable protocol); each view
    # carries its dataset timestamp as its causal availability time.
    train_views = [(ds_ts[di], *load_view((fi, di), mapped=False)) for fi, di in sel]
    held_frames = [load_view(c, mapped=True) for c in held_c]
    held_ts = np.array([ds_ts[di] for di, _ in held_c])
    print(f"train={len(train_views)}  held-out={len(held_frames)}", flush=True)

    # --- per-keyframe parts (arrival order) ---
    blob = torch.load(os.path.join(REPO_ROOT, args.kfgauss), map_location="cpu")
    parts = []
    kf_pose_data = []
    for k, kf in enumerate(blob["keyframes"]):
        h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
        local = {key: kf[key].to(dev) for key in
                 ("means", "scales", "rotations", "sh", "opacities", "conf")}
        got = gaussians_from_keyframe(local, kf["img"].to(dev), h, w, k, dev)
        if got is None:
            continue
        parts.append(got)
        kf_pose_data.append(kf["T_WC"].to(dev))
    kf_mats_all = sim3_to_mat(torch.stack([p.reshape(-1) for p in kf_pose_data]))
    n_kf = len(parts)
    # est_ts[i] is keyframe i's own timestamp (verified line-by-line against the
    # dumped T_WC in refine_local.py). Arrival order must match blob order.
    assert n_kf == len(est_ts), f"{n_kf} keyframes vs {len(est_ts)} trajectory poses"
    assert np.all(np.diff(est_ts) >= 0), "keyframe trajectory is not time-ordered"

    means_all = torch.cat([p[0] for p in parts])
    print(f"map: {means_all.shape[0]:,} gaussians over {n_kf} keyframes", flush=True)
    # Held constant across arms; see module docstring.
    extent = float((means_all.max(0).values - means_all.min(0).values).norm() / 2)

    win = gaussian_window(device=dev)
    lp = lpips_lib.LPIPS(net="alex").to(dev)
    K = ds.camera_intrinsics.K_frame
    rng = np.random.default_rng(args.seed)
    tag = f"[{args.tag}] " if args.tag else ""

    @torch.no_grad()
    def evaluate(model, kf_mats, subset=None):
        mses, lps = [], []
        mw, cw = model.world(kf_mats)
        rgb_, op_ = model.rgb(), model.opacity()
        views = held_frames if subset is None else [held_frames[i] for i in subset]
        for c2w, tgt in views:
            pred = render(mw, cw, rgb_, op_, c2w, K, tgt.shape[-2:], dev).clamp(0, 1)
            mses.append(torch.mean((pred - tgt) ** 2).item())
            lps.append(lp(pred * 2 - 1, tgt * 2 - 1).item())
        mse = sum(mses) / len(mses)
        return -10 * math.log10(max(mse, 1e-12)), sum(lps) / len(lps)

    def report(model, kf_mats, it, extra=""):
        p, l = evaluate(model, kf_mats)
        print(f"  {tag}iter {it:5} | held-out psnr={p:7.4f}  lpips={l:.4f}" + extra,
              flush=True)
        return p, l

    def train_step(model, opt, kf_mats, view):
        _, c2w, tgt = view
        mw, cw = model.world(kf_mats)
        pred = render(mw, cw, model.rgb(), model.opacity(), c2w, K,
                      tgt.shape[-2:], dev)
        loss = ((1 - DSSIM_WEIGHT) * (pred - tgt).abs().mean()
                + DSSIM_WEIGHT * (1 - ssim(pred.clamp(0, 1), tgt, win)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.item()

    if args.mode == "posthoc":
        model = LocalGaussianMap(
            *[torch.cat([p[i] for p in parts]) for i in range(6)]).to(dev)
        opt = torch.optim.Adam(model.param_groups(extent), eps=1e-15)
        report(model, kf_mats_all, 0, "  (init)")
        for it in range(1, args.iters + 1):
            view = train_views[int(rng.integers(len(train_views)))]
            loss = train_step(model, opt, kf_mats_all, view)
            if it % args.eval_every == 0:
                report(model, kf_mats_all, it, f"  train_loss={loss:.4f}")
        return 0

    # --- causal replay ---
    av_ts = np.array([v[0] for v in train_views])
    kf_ts = np.array(est_ts, dtype=np.float64)
    t_end = max(kf_ts[-1], float(av_ts.max()))
    # Round k starts at keyframe k's arrival and covers [kf_ts[k], kf_ts[k+1]);
    # the last round covers [kf_ts[-1], t_end]. A view is usable in the first
    # round whose END time is >= the view's own timestamp (see docstring).
    bounds = np.concatenate([kf_ts, [t_end]])
    seg = np.diff(bounds)
    seg_iters = np.rint(args.iters * seg / seg.sum()).astype(int)
    seg_iters[-1] += args.iters - int(seg_iters.sum())  # fix rounding drift

    model = None
    opt = None
    kf_mats = None
    it_done = 0
    next_report = args.eval_every
    deferred = 0
    for k in range(n_kf):
        # --- keyframe k arrives: inject its Gaussians (the online non-stationarity) ---
        new = list(parts[k])
        if model is None:
            model = LocalGaussianMap(*[t.clone() for t in new]).to(dev)
            opt = None
        else:
            with torch.no_grad():
                merged = [
                    torch.cat([model.means.detach(), new[0]]),
                    torch.cat([model.log_scales.detach().exp(), new[1]]),
                    torch.cat([model.quat.detach(), new[2]]),
                    torch.cat([model.rgb().detach(), new[3]]),
                    torch.cat([model.opacity().detach(), new[4]]),
                    torch.cat([model.kf_id, new[5]]),
                ]
            model = LocalGaussianMap(*merged).to(dev)
        opt = _optimizer_for(model, extent, opt)
        kf_mats = kf_mats_all[: k + 1]

        avail = [v for v in train_views if v[0] <= bounds[k + 1] + 1e-9]
        n_it = int(seg_iters[k]) + deferred
        if not avail:
            deferred = n_it
            print(f"  {tag}kf {k:3} t={bounds[k]:8.3f}  +{new[0].shape[0]:,} gaussians "
                  f"| no supervision yet, deferring {n_it} iters", flush=True)
            continue
        deferred = 0
        print(f"  {tag}kf {k:3} t={bounds[k]:8.3f}  +{new[0].shape[0]:,} gaussians "
              f"-> {model.n:,}  | {len(avail)} views, {n_it} iters", flush=True)
        for _ in range(n_it):
            if args.sampling == "recent":
                pool = avail[-args.recent_window:]
                view = pool[int(rng.integers(len(pool)))]
            elif args.sampling == "mixed":
                old_pool = avail[:-args.recent_window]
                if old_pool and rng.random() >= 0.7:
                    view = old_pool[int(rng.integers(len(old_pool)))]
                else:
                    view = avail[-args.recent_window:][
                        int(rng.integers(min(args.recent_window, len(avail))))]
            else:
                view = avail[int(rng.integers(len(avail)))]
            loss = train_step(model, opt, kf_mats, view)
            it_done += 1
            if it_done >= next_report:
                report(model, kf_mats, it_done, f"  train_loss={loss:.4f}")
                next_report += args.eval_every
    if it_done % args.eval_every != 0:
        report(model, kf_mats, it_done, "  (final)")
    # Assimilation vs forgetting: late-arriving held-out views score the
    # newest content, early ones the regions the camera has left.
    med = np.median(held_ts)
    pe, le = evaluate(model, kf_mats, [i for i, t in enumerate(held_ts) if t <= med])
    pl, ll = evaluate(model, kf_mats, [i for i, t in enumerate(held_ts) if t > med])
    print(f"  {tag}final halves: early {pe:.4f}/{le:.4f}  late {pl:.4f}/{ll:.4f}",
          flush=True)
    print(f"  {tag}causal replay done: {it_done} iters, {model.n:,} gaussians",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

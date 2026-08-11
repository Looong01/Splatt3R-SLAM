"""Score a map on held-out frames STRATIFIED BY BASELINE to the mapping clusters.

§17.48 measured that a referee correction improves the map's absolute geometry
2.2-6.4x and makes the rendered image worse. Two readings, and only the second
is safe on the evidence:

  1. absolute geometric accuracy and rendered quality genuinely conflict;
  2. THIS evaluation cannot see the improvement and can see the disruption.

Kimi's round-18 point, adopted: reading 2 is testable today, on maps that
already exist. Every held-out frame in the standard protocol sits ~0.057 m from
its nearest keyframe, where §17.17's arithmetic puts a 4.5% depth error at
0.66 px -- below the noise. But a frame's *visible content* is not necessarily
mapped by the nearest keyframe: in a sequence that revisits, a wall can be
rendered from a cluster metres away. At 2-3 m an 8% scale error is 10-30 px,
and there the correction must be visible.

So the stratifying variable is not distance to the nearest keyframe. It is the
**alpha-weighted distance to the clusters that actually paint the frame**,
which the renderer can report directly:

    baseline_f = sum_k a_k(f) * ||t_k - c_f||  /  sum_k a_k(f)

with a_k(f) the alpha cluster k contributes to frame f. Frames are then split
at --split metres and scored separately.

Pre-registered (round 18): a corrected map that is geometrically better should
win the wide-baseline subset by >= 0.3 dB. If it wins there, "the objectives
conflict" is false and the honest claim is a trade-off curve. If it does not
win even there, the conflict is real.
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

from eval_map_quality import associate, load_tum_traj, umeyama_sim3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True, nargs="+",
                    help="one or more maps to score on the SAME stratified set")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-held", type=int, default=100)
    ap.add_argument("--split", type=float, default=0.5,
                    help="metres; frames above this are the wide-baseline set")
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--aa-sigma", type=float, default=0.5)
    ap.add_argument("--streak-opacity", type=float, default=0.0)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(p) for p in args.kfgauss]
    os.chdir(REPO_ROOT)
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.refiner import (LocalGaussianMap, sim3_to_mat,
                                       gaussians_from_keyframe, render_map)
    import lpips as lpips_lib

    load_config(args.config)
    dev = args.device
    ds = load_dataset(os.path.abspath(args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(args.traj)
    gt_ts, gt_T = load_tum_traj(os.path.join(os.path.abspath(args.dataset),
                                             "groundtruth.txt"))
    pairs = associate(est_ts, gt_ts)
    s_, R_, t_ = umeyama_sim3(np.array([est_T[i, :3, 3] for i, _ in pairs]),
                              np.array([gt_T[j, :3, 3] for _, j in pairs]))
    Rt = R_.T
    kf_set = {j for _, j in associate(est_ts, ds_ts)}
    cand = [(i, j) for i, j in associate(ds_ts, gt_ts) if i not in kf_set]
    held = cand[:: max(1, len(cand) // args.n_held)][: args.n_held]

    lp = lpips_lib.LPIPS(net="alex").to(dev)
    K = torch.as_tensor(ds.camera_intrinsics.K_frame, dtype=torch.float32,
                        device=dev)

    views = []
    for di, gj in held:
        img = resize_img(ds.get_image(di), ds.img_size)["img"]
        c2w = np.eye(4)
        c2w[:3, :3] = Rt @ gt_T[gj][:3, :3]
        c2w[:3, 3] = Rt @ (gt_T[gj][:3, 3] - t_) / s_
        # resize_img returns [-1,1]; the rasterizer outputs [0,1]. Comparing
        # the two spaces directly scores ~3 dB on a map that renders correctly
        # -- eval_map_quality.py carries the same warning in a comment, and
        # this file hit it anyway (17.40 was the same trap in another costume).
        views.append((c2w, torch.as_tensor(img, dtype=torch.float32,
                                           device=dev) * 0.5 + 0.5))

    os.chdir(CORE)
    results, baselines = {}, None
    for path, label in zip(args.kfgauss, labels):
        blob = torch.load(os.path.join(REPO_ROOT, path)
                          if not os.path.isabs(path) else path,
                          map_location="cpu")
        parts, poses = [], []
        for k, kf in enumerate(blob["keyframes"]):
            h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
            local = {x: kf[x].to(dev) for x in
                     ("means", "scales", "rotations", "sh", "opacities", "conf")}
            got = gaussians_from_keyframe(local, kf["img"].to(dev), h, w, k, dev,
                                          min_confidence=args.min_confidence,
                                          aa_sigma_scale=args.aa_sigma,
                                          streak_opacity=args.streak_opacity)
            if got is None:
                continue
            parts.append(got)
            poses.append(kf["T_WC"].to(dev))
        kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in poses]))
        model = LocalGaussianMap(*[torch.cat([p[i] for p in parts])
                                   for i in range(6)]).to(dev)
        n_kf = int(model.kf_id.max()) + 1
        cam = kf_mats[:, :3, 3]

        rows = []
        need_base = baselines is None
        base_here = []
        with torch.no_grad():
            mw, cw = model.world(kf_mats)
            rgb_, op_ = model.rgb(), model.opacity()
            ones = torch.ones_like(rgb_)
            for vi, (c2w, tgt) in enumerate(views):
                h, w = tgt.shape[-2:]
                pred = render_map(mw, cw, rgb_, op_, c2w, K, (h, w),
                                  dev).clamp(0, 1)
                mse = float(torch.mean((pred - tgt) ** 2))
                l = float(lp(pred, tgt, normalize=True).mean())
                rows.append((mse, l))
                if need_base:
                    # alpha each cluster contributes to THIS frame -- the
                    # renderer's own answer to "who painted this pixel"
                    a = []
                    for k in range(n_kf):
                        m = model.kf_id == k
                        if not bool(m.any()):
                            a.append(0.0); continue
                        al = render_map(mw[m], cw[m], ones[m], op_[m], c2w, K,
                                        (h, w), dev).clamp(0, 1).mean()
                        a.append(float(al))
                    a = np.array(a)
                    d = np.linalg.norm(
                        cam.cpu().numpy() - np.asarray(c2w)[:3, 3], axis=1)
                    base_here.append(float((a * d).sum() / max(a.sum(), 1e-9)))
        if need_base:
            baselines = np.array(base_here)
        results[label] = np.array(rows)

    wide = baselines > args.split
    print(f"\n{len(baselines)} held-out frames, alpha-weighted baseline to the "
          f"clusters that paint them:")
    print(f"  median {np.median(baselines):.3f} m   range "
          f"[{baselines.min():.3f}, {baselines.max():.3f}]   "
          f"{int(wide.sum())} above {args.split} m")
    print(f"\n{'map':22s} {'all psnr/lpips':>22} {'near':>22} {'WIDE':>22}")
    for label, r in results.items():
        def cell(m):
            if m.sum() == 0:
                return f"{'--':>22}"
            return (f"{-10 * np.log10(max(r[m, 0].mean(), 1e-12)):>10.3f} / "
                    f"{r[m, 1].mean():.4f}")
        allm = np.ones(len(baselines), bool)
        print(f"{label:22s} {cell(allm)} {cell(~wide)} {cell(wide)}")
    print("\npre-registered: the corrected map should win the WIDE column by "
          ">= 0.3 dB if\nits geometric gain is image-real. If it does not win "
          "there either, the\nconflict between metric accuracy and rendered "
          "quality is real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

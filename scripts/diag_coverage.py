"""How much of a held-out view is black, and is that black recoverable?

The large black regions in the GUI captures were dismissed in skill 17.9 as
"not a defect -- unmapped, no keyframe covered them". That is only half an
answer. A pixel can be black for three different reasons, and only one of them
is genuinely unrecoverable:

  (a) never observed by ANY frame of the sequence -- the camera never looked
      there. Nothing can fix this.
  (b) observed, but only by frames that did not become KEYFRAMES. The map bakes
      keyframes only (46 of ~750 frames on 360), so this content was seen and
      then thrown away. Recoverable by injecting from tracked frames, or by a
      denser keyframe policy.
  (c) observed by a keyframe, but removed by the injection filters -- the depth
      percentile (0.98), the confidence threshold, the opacity threshold, the
      max-scale filter. Recoverable by relaxing whichever one is responsible.

This script separates them. For each held-out view it renders the accumulated
alpha of the map to find the black pixels, then re-projects every keyframe's
RAW prediction (before filtering) and every tracked frame's pose to ask which
of the three cases each black pixel falls in.

The answer decides whether "fill the black" is a research problem or a config
change.

Usage:
    python3 scripts/diag_coverage.py \
        --kfgauss logs/frames_head/rgbd_dataset_freiburg1_360_kfgauss.pt \
        --traj    logs/frames_head/rgbd_dataset_freiburg1_360.txt \
        --frames-traj logs/frames_head/rgbd_dataset_freiburg1_360_frames.txt \
        --dataset datasets/tum/rgbd_dataset_freiburg1_360
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
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames-traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-held", type=int, default=25)
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--aa-sigma", type=float, default=0.5)
    ap.add_argument("--alpha-thresh", type=float, default=0.1)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.refiner import (LocalGaussianMap, sim3_to_mat,
                                       gaussians_from_keyframe, render_map)

    load_config(args.config)
    os.chdir(CORE)
    dev = args.device

    ds = load_dataset(os.path.join(REPO_ROOT, args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(os.path.join(REPO_ROOT, args.traj))
    frm_ts, frm_T = load_tum_traj(os.path.join(REPO_ROOT, args.frames_traj))
    gt_ts, gt_T = load_tum_traj(
        os.path.join(REPO_ROOT, args.dataset, "groundtruth.txt"))

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
    held = []
    for di, gj in held_c:
        img = resize_img(ds.get_image(di), ds.img_size)["img"]
        tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
        held.append((to_map(gt_T[gj]), tgt))

    blob = torch.load(os.path.join(REPO_ROOT, args.kfgauss), map_location="cpu")
    K = torch.as_tensor(ds.camera_intrinsics.K_frame, dtype=torch.float32, device=dev)

    # Two maps: the deployed one (filters on) and an UNFILTERED one, which is
    # every Gaussian the network actually predicted. The difference between
    # their coverage is case (c) -- content the filters threw away.
    def build(min_conf, min_opacity, depth_pct, max_scale):
        parts, poses = [], []
        for k, kf in enumerate(blob["keyframes"]):
            h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
            local = {key: kf[key].to(dev) for key in
                     ("means", "scales", "rotations", "sh", "opacities", "conf")}
            got = gaussians_from_keyframe(
                local, kf["img"].to(dev), h, w, k, dev,
                min_confidence=min_conf, min_opacity=min_opacity,
                depth_max_percentile=depth_pct, max_scale=max_scale,
                aa_sigma_scale=args.aa_sigma)
            if got is None:
                continue
            parts.append(got)
            poses.append(kf["T_WC"].to(dev))
        mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in poses]))
        m = LocalGaussianMap(*[torch.cat([p[i] for p in parts])
                               for i in range(6)]).to(dev)
        return m, mats

    print("building maps...", flush=True)
    # One map per filter RELAXED IN ISOLATION, so the black each is responsible
    # for is separable. Relaxing them all at once only bounds the total.
    variants = [
        ("deployed",      args.min_confidence, 0.3, 0.98, 0.5),
        ("no depth-pct",  args.min_confidence, 0.3, 1.00, 0.5),
        ("no confidence", 0.0,                 0.3, 0.98, 0.5),
        ("no opacity",    args.min_confidence, 0.0, 0.98, 0.5),
        ("no max-scale",  args.min_confidence, 0.3, 0.98, 1e9),
        ("none",          0.0,                 0.0, 1.00, 1e9),
    ]
    built = []
    for name, mc, mo, dp, ms in variants:
        m, mt = build(mc, mo, dp, ms)
        built.append((name, m, mt))
        print(f"  {name:14} {m.n:>10,} gaussians", flush=True)
    m_dep, mats = built[0][1], built[0][2]
    m_raw, mats_raw = built[-1][1], built[-1][2]

    # Every tracked-frame pose, for case (b): was this direction ever looked at
    # by a non-keyframe? Approximated by the frame's optical axis and position,
    # which is enough to answer "did the camera ever point here".
    fr = [frm_T[i] for i, _ in associate(frm_ts, ds_ts)]
    fr_pos = torch.as_tensor(np.array([f[:3, 3] for f in fr]),
                             dtype=torch.float32, device=dev)
    fr_fwd = torch.as_tensor(np.array([f[:3, 2] for f in fr]),
                             dtype=torch.float32, device=dev)
    kf_pos = mats[:, :3, 3]
    print(f"{len(fr)} tracked frames, {mats.shape[0]} keyframes", flush=True)

    tot = {name: 0.0 for name, _, _ in built}
    n_v = 0
    with torch.no_grad():
        for c2w, tgt in held:
            h, w = tgt.shape[-2:]
            for name, mm, mt in built:
                mw, cw = mm.world(mt)
                a = render_map(mw, cw, torch.ones_like(mm.rgb()), mm.opacity(),
                               c2w, K, (h, w), dev).clamp(0, 1).mean(1)
                tot[name] += float((a < args.alpha_thresh).float().mean())
            n_v += 1
    bd = tot["deployed"] / n_v
    br = tot["none"] / n_v
    print(f"\nblack fraction by filter, each relaxed alone:")
    for name, mm, _ in built:
        v = tot[name] / n_v
        print(f"  {name:14} {v:6.2%}   (-{bd - v:5.2%} vs deployed, "
              f"{mm.n:>10,} gaussians)")
    print(f"\nblack pixels on held-out views (alpha < {args.alpha_thresh}):")
    print(f"  deployed map      {bd:6.2%}")
    print(f"  unfiltered map    {br:6.2%}")
    print(f"  recoverable by relaxing the injection filters (case c): "
          f"{bd - br:6.2%} of the frame, "
          f"{100.0 * (bd - br) / max(bd, 1e-9):.1f}% of the black")
    print(f"  remaining black   {br:6.2%}  -- either never observed (a) or "
          f"observed only by non-keyframes (b)")

    # Case (b) upper bound: how much closer is the nearest TRACKED frame than
    # the nearest KEYFRAME, for the held-out viewpoints? If tracked frames are
    # everywhere the keyframes are, (b) is empty and the rest is case (a).
    d_kf, d_fr = [], []
    for c2w, _ in held:
        p = torch.as_tensor(c2w[:3, 3], dtype=torch.float32, device=dev)
        d_kf.append(float((kf_pos - p).norm(dim=1).min()))
        d_fr.append(float((fr_pos - p).norm(dim=1).min()))
    print(f"\nnearest-camera distance to a held-out viewpoint:")
    print(f"  nearest keyframe      {np.mean(d_kf):.3f} m")
    print(f"  nearest tracked frame {np.mean(d_fr):.3f} m")
    print("  If these are close, the keyframes already cover wherever the "
          "camera went and case (b) is small: the remaining black is content "
          "the camera never looked at, and no amount of injection recovers it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

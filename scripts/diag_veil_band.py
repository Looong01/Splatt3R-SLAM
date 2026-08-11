"""Is the low-parallax penalty an information boundary, or layered veils?

§17.36 moved the attribution of §17.34's -0.964 correlation from the network's
input pairing (falsified: depth quality is flat in parallax) to the supervision.
Two mechanisms remain and they have opposite consequences:

  M1/M3'  the supervision itself. Under rotation every view of a surface sits at
          nearly the same place, so the photometric loss constrains fewer
          directions and refinement has less room. Nothing in the map can fix
          this -- it is an ABSENCE defect in §17.27's sense, an information
          boundary set by the trajectory.
  M2      layered veils. Rotation revisits surfaces from many angles, stacking
          more mutually-disagreeing clusters, and those obstruct refinement.
          That is a DISAGREEMENT defect: engineering, and fixable.

Calling it a boundary when M2 holds would be simply wrong, so this discriminates
before the wording is chosen.

The metric (Kimi's, replacing a weaker one of mine -- "mean contributing clusters
per pixel" does not discriminate, since translation sequences stack layers too by
orbiting an object): **veil-band fraction** = the fraction of covered pixels
whose top two contributing clusters differ in depth by 0.5-20 cm. That band is
the veil: close enough to be the same surface, far enough to be a second copy of
it rather than its thickness.

Computed by rendering each cluster ALONE (alpha and alpha-weighted depth). K
renders of 1/K of the map costs about one full render, so this is cheap.

Then the decisive step: partial correlation. If Delta-lpips still tracks the
translation/rotation ratio after controlling for veil-band fraction, M1/M3'
dominates and the boundary framing is right. If it collapses, M2 dominates and
the framing is wrong.
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
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-views", type=int, default=8)
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--aa-sigma", type=float, default=0.5)
    ap.add_argument("--band", type=float, nargs=2, default=(0.005, 0.20))
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
    held = cand[:: max(1, len(cand) // args.n_views)][: args.n_views]

    blob = torch.load(args.kfgauss, map_location="cpu")
    os.chdir(CORE)
    parts, poses = [], []
    for k, kf in enumerate(blob["keyframes"]):
        h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
        local = {x: kf[x].to(dev) for x in
                 ("means", "scales", "rotations", "sh", "opacities", "conf")}
        got = gaussians_from_keyframe(local, kf["img"].to(dev), h, w, k, dev,
                                      min_confidence=args.min_confidence,
                                      aa_sigma_scale=args.aa_sigma)
        if got is None:
            continue
        parts.append(got)
        poses.append(kf["T_WC"].to(dev))
    kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in poses]))
    model = LocalGaussianMap(*[torch.cat([p[i] for p in parts])
                               for i in range(6)]).to(dev)
    K = torch.as_tensor(ds.camera_intrinsics.K_frame, dtype=torch.float32, device=dev)
    n_kf = int(model.kf_id.max()) + 1

    lo, hi = args.band
    fracs, layers = [], []
    with torch.no_grad():
        mw, cw = model.world(kf_mats)
        op_ = model.opacity()
        for di, gj in held:
            img = resize_img(ds.get_image(di), ds.img_size)["img"]
            h, w = img.shape[-2:]
            c2w = np.eye(4)
            c2w[:3, :3] = Rt @ gt_T[gj][:3, :3]
            c2w[:3, 3] = Rt @ (gt_T[gj][:3, 3] - t_) / s_
            w2c = np.linalg.inv(c2w)
            Rw = torch.as_tensor(w2c[:3, :3], dtype=torch.float32, device=dev)
            tw = torch.as_tensor(w2c[:3, 3], dtype=torch.float32, device=dev)
            zview = mw @ Rw.transpose(0, 1) + tw
            zview = zview[:, 2]
            A, Z = [], []
            for k in range(n_kf):
                m = (model.kf_id == k)
                if not bool(m.any()):
                    A.append(None); continue
                one = torch.ones((int(m.sum()), 3), device=dev)
                a = render_map(mw[m], cw[m], one, op_[m], c2w, K, (h, w),
                               dev).clamp(0, 1).mean(1)[0]
                zc = (zview[m] / 10.0).clamp(0, 1)[:, None].expand(-1, 3).contiguous()
                zr = render_map(mw[m], cw[m], zc, op_[m], c2w, K, (h, w),
                                dev).clamp(0, 1).mean(1)[0] * 10.0
                A.append(a); Z.append(zr)
            av = torch.stack([x for x in A if x is not None])      # (K,h,w)
            zv = torch.stack(Z)
            top = av.topk(2, dim=0)
            i1, i2 = top.indices[0], top.indices[1]
            a1, a2 = top.values[0], top.values[1]
            z1 = torch.gather(zv, 0, i1[None])[0]
            z2 = torch.gather(zv, 0, i2[None])[0]
            # Both layers must be real contributors, or the "second" layer is
            # just numerical dust and every pixel would count as a veil.
            both = (a1 > 0.2) & (a2 > 0.1)
            d = (z1 - z2).abs()
            covered = a1 > 0.2
            fracs.append(float(((both & (d > lo) & (d < hi)).float().sum()
                                / covered.float().sum().clamp_min(1))))
            layers.append(float((av > 0.1).float().sum(0)[covered].mean()))
    print(f"{os.path.basename(args.kfgauss)}")
    print(f"  veil-band fraction  {np.mean(fracs):.4f} +- {np.std(fracs):.4f}")
    print(f"  mean layers/pixel   {np.mean(layers):.3f}   (the weaker metric, "
          f"for comparison)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

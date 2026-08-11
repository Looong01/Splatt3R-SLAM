"""A spatial instrument: how big is the colour step where two clusters meet?

Everything this project measures is per-view: psnr, lpips, the per-frame affine.
§17.39 and §17.40 showed those instruments are deaf to the defect that remains --
there is no per-view appearance variation in the map at all, yet the seams are
visible in every GUI capture. The only possibility left is that the defect is
**spatial**: a step between two clusters *inside one frame*, which a per-frame
transform cannot represent by construction.

So this builds the missing instrument, to Kimi's round-5 design.

  1. Render each cluster alone (alpha, depth, colour). K renders of 1/K of the
     map cost about one full render.
  2. Per pixel, take the argmax-alpha cluster. Boundaries are 4-neighbour jumps
     in that id map -- **derived from the renderer**, not extracted from image
     edges, which is what makes them trustworthy.
  3. Mode-filter the id map first. Where two clusters interleave, the raw argmax
     flickers pixel to pixel and every pixel in the overlap would count as a
     boundary (the concern raised as Q55). A 5x5 majority vote removes the
     flicker and keeps genuine borders.
  4. Keep boundary pixels with rendered depth difference < 5 cm (a real occlusion
     edge is a different thing and SHOULD have a colour step) and accumulated
     alpha > 0.8 (both sides actually covered).
  5. The statistic is the brightness step across the boundary, normalized by the
     local gradient energy, so a boundary that happens to fall on a texture edge
     is not counted as a seam.
  6. The null: shift every boundary by 10 px and recompute. A map with no seams
     scores the same as its own shifted null; the ratio is the signal.

Usage:
    python3 scripts/diag_seam_step.py --kfgauss ... --traj ... --dataset ...
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
import torch.nn.functional as F

from eval_map_quality import associate, load_tum_traj, umeyama_sim3


def mode_filter(idx, n_id, k=5):
    """Majority id in a k x k window, via one-hot box filtering."""
    oh = F.one_hot(idx.long(), n_id).permute(2, 0, 1)[None].float()
    sm = F.avg_pool2d(F.pad(oh, (k // 2,) * 4, mode="replicate"), k, stride=1)
    return sm[0].argmax(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-views", type=int, default=6)
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--aa-sigma", type=float, default=0.5)
    ap.add_argument("--streak-opacity", type=float, default=0.0)
    ap.add_argument("--depth-tol", type=float, default=0.05)
    ap.add_argument("--shift", type=int, default=10)
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
                                      aa_sigma_scale=args.aa_sigma,
                                      streak_opacity=args.streak_opacity)
        if got is None:
            continue
        parts.append(got)
        poses.append(kf["T_WC"].to(dev))
    kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in poses]))
    model = LocalGaussianMap(*[torch.cat([p[i] for p in parts])
                               for i in range(6)]).to(dev)
    K = torch.as_tensor(ds.camera_intrinsics.K_frame, dtype=torch.float32, device=dev)
    n_kf = int(model.kf_id.max()) + 1

    steps, nulls, fracs = [], [], []
    with torch.no_grad():
        mw, cw = model.world(kf_mats)
        rgb_, op_ = model.rgb(), model.opacity()
        ones = torch.ones_like(rgb_)
        for di, gj in held:
            img = resize_img(ds.get_image(di), ds.img_size)["img"]
            h, w = img.shape[-2:]
            c2w = np.eye(4)
            c2w[:3, :3] = Rt @ gt_T[gj][:3, :3]
            c2w[:3, 3] = Rt @ (gt_T[gj][:3, 3] - t_) / s_
            w2c = np.linalg.inv(c2w)
            Rw = torch.as_tensor(w2c[:3, :3], dtype=torch.float32, device=dev)
            tw = torch.as_tensor(w2c[:3, 3], dtype=torch.float32, device=dev)
            zview = (mw @ Rw.transpose(0, 1) + tw)[:, 2]
            full = render_map(mw, cw, rgb_, op_, c2w, K, (h, w), dev).clamp(0, 1)
            A, Z = [], []
            for k in range(n_kf):
                m = model.kf_id == k
                if not bool(m.any()):
                    A.append(torch.zeros((h, w), device=dev))
                    Z.append(torch.zeros((h, w), device=dev))
                    continue
                A.append(render_map(mw[m], cw[m], ones[m], op_[m], c2w, K,
                                    (h, w), dev).clamp(0, 1).mean(1)[0])
                zc = (zview[m] / 10.0).clamp(0, 1)[:, None].expand(-1, 3).contiguous()
                Z.append(render_map(mw[m], cw[m], zc, op_[m], c2w, K, (h, w),
                                    dev).clamp(0, 1).mean(1)[0] * 10.0)
            av, zv = torch.stack(A), torch.stack(Z)
            ids = mode_filter(av.argmax(0), n_kf)
            alpha = av.max(0).values
            zsel = torch.gather(zv, 0, ids[None])[0]

            lum = full.mean(1)[0]
            gx = (lum[:, 1:] - lum[:, :-1]).abs()
            gy = (lum[1:, :] - lum[:-1, :]).abs()
            grad = F.pad(gx, (0, 1)) + F.pad(gy, (0, 0, 0, 1))
            gnorm = F.avg_pool2d(F.pad(grad[None, None], (3,) * 4,
                                       mode="replicate"), 7, 1)[0, 0]

            # ALWAYS an adjacent-pixel comparison. The null shifts WHICH pixels
            # are looked at, never how far apart they are -- comparing x with
            # x+11 instead of x with x+1 measures a different quantity entirely
            # and produced a null 8x larger than the signal on the first
            # attempt, i.e. an artifact, not a seam verdict.
            i0, i1 = ids[:, :-1], ids[:, 1:]
            a0, a1 = alpha[:, :-1], alpha[:, 1:]
            z0, z1 = zsel[:, :-1], zsel[:, 1:]
            l0, l1 = lum[:, :-1], lum[:, 1:]
            gn = gnorm[:, :-1]
            covered = (a0 > 0.8) & (a1 > 0.8) & ((z0 - z1).abs() < args.depth_tol)
            border = (i0 != i1) & covered
            step_map = (l0 - l1).abs() / (gn + 0.02)

            def measure(mask):
                if int(mask.sum()) < 50:
                    return None, 0.0
                return float(step_map[mask].mean()), float(mask.float().mean())

            s0, f0 = measure(border)
            # the null: the same border mask displaced sideways, so it lands on
            # ordinary positions with the same spatial statistics
            shifted = torch.zeros_like(border)
            shifted[:, args.shift:] = border[:, :-args.shift]
            s1, _ = measure(shifted & covered & ~border)

            if s0 is not None and s1 is not None:
                steps.append(s0); nulls.append(s1); fracs.append(f0)

    if not steps:
        print("no usable boundaries"); return 1
    print(f"\n{os.path.basename(args.kfgauss)}  ({len(steps)} views)")
    print(f"  seam pixels          {np.mean(fracs):.4f} of the frame")
    print(f"  seam step            {np.mean(steps):.4f}")
    print(f"  shifted-null step    {np.mean(nulls):.4f}")
    print(f"  ratio seam/null      {np.mean(steps) / max(np.mean(nulls), 1e-9):.4f}")
    print("  >1 means cluster borders carry a colour step that ordinary "
          "positions do not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

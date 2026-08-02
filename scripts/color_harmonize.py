"""Plan 2 (splatt3r-color-consistency): overlap-based colour harmonization,
measured on the raw baked map.

Splatt3R bakes each frame's raw pixel colour into its Gaussians and nothing
reconciles two observations of one surface, so a revisit after exposure
drift paints a second, differently-coloured copy of the same surface. Plan 2
fit a per-keyframe colour gain against the already-placed map in the overlap
region and applies it to that keyframe's whole contribution -- the same idea
as exposure compensation in panorama stitching, solvable in closed form
because spatial proximity gives explicit correspondence.

This script measures the idea OFFLINE on a keyframe dump, in causal order
(keyframe k fits against keyframes 0..k-1, already corrected -- the order
the live system would have to use), and scores held-out psnr/lpips of the
raw, unoptimized map before vs after. It answers: is harmonization worth
wiring into the bake path at all, now that the refiner exists (whose
optimizer already harmonizes photometrically)?

Method:
  1. Build per-keyframe world-space Gaussians from the dump (no training).
  2. Voxel hash (default 10 mm, the scale §15.7 measured for cluster
     overlap) over earlier keyframes' mean colours per voxel.
  3. For kf k: least-squares per-channel gain between its overlapped
     Gaussians' colours and the earlier occupants' voxel means, clamped to
     [0.6, 1.67]; applied to ALL of kf k's colours. Chain forward.
  4. Held-out psnr/lpips of the composed map, before vs after.

Usage:
    python3 scripts/color_harmonize.py \\
        --kfgauss logs/frames_head/<seq>_kfgauss.pt \\
        --traj logs/frames_head/<seq>.txt \\
        --frames-traj logs/frames_head/<seq>_frames.txt \\
        --dataset datasets/tum/<seq>
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

from refine_local import render  # noqa: E402  (its import sets up sys.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames-traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--gain-clamp", type=float, nargs=2, default=(0.6, 1.67),
                    metavar=("LO", "HI"))
    ap.add_argument("--n-held", type=int, default=50)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    import lpips as lpips_lib
    from eval_map_quality import associate, load_tum_traj, umeyama_sim3
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.refiner import gaussians_from_keyframe, sim3_to_mat

    load_config(args.config)
    CORE = os.path.join(REPO_ROOT, "splatt3r_core")
    os.chdir(CORE)
    dev = args.device

    ds = load_dataset(os.path.join(REPO_ROOT, args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(os.path.join(REPO_ROOT, args.traj))
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

    held_frames = []
    for di, gi in held_c:
        img = resize_img(ds.get_image(di), ds.img_size)["img"]
        tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
        held_frames.append((to_map(gt_T[gi]), tgt))
    print(f"held-out={len(held_frames)}", flush=True)

    # --- per-keyframe world Gaussians (no optimization anywhere) ---
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
    kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in kf_pose_data]))
    n_kf = len(parts)

    def world_of(part):
        means, scales, rots, rgb, opas, kf_id = part
        k = int(kf_id[0].item())
        A, b = kf_mats[k, :3, :3], kf_mats[k, :3, 3]
        return means @ A.T + b, rgb

    worlds = [world_of(p) for p in parts]
    print(f"map: {sum(wm.shape[0] for wm, _ in worlds):,} gaussians over "
          f"{n_kf} keyframes, voxel={args.voxel} m", flush=True)

    # --- sequential harmonization (vectorized; causal order) ---
    lo_cl, hi_cl = args.gain_clamp
    harm_rgb = [rgb.clone() for _, rgb in worlds]

    def voxel_ids(means):
        v = torch.floor(means / args.voxel).long()
        v = v - v.min(0).values
        B = int(v.max().item()) + 2
        return (v[:, 0] * B + v[:, 1]) * B + v[:, 2]

    # accumulated voxel -> (sum colour, count), sorted keys, corrected colours
    acc_keys = torch.empty(0, dtype=torch.long, device=dev)
    acc_sum = torch.empty(0, 3, device=dev)
    acc_cnt = torch.empty(0, device=dev)
    report = []
    for k in range(n_kf):
        wm = worlds[k][0]
        vk = voxel_ids(wm)
        if k > 0 and acc_keys.numel() > 0:
            loc = torch.searchsorted(acc_keys, vk).clamp_(max=acc_keys.numel() - 1)
            hit = acc_keys[loc] == vk
            if int(hit.sum()) >= 100:
                own = harm_rgb[k][hit]
                ref = acc_sum[loc[hit]] / acc_cnt[loc[hit], None].clamp_min(1)
                # least-squares per-channel gain, no offset (gain alone keeps
                # black/white points anchored)
                gain = (ref * own).sum(0) / (own * own).sum(0).clamp_min(1e-8)
                gain = gain.clamp(lo_cl, hi_cl)
                harm_rgb[k] = (harm_rgb[k] * gain).clamp(0, 1)
                report.append((k, int(hit.sum()), gain.tolist()))
        # merge this keyframe into the voxel map with CORRECTED colours
        all_k = torch.cat([acc_keys, vk])
        all_c = torch.cat([acc_sum, harm_rgb[k].float()])
        all_n = torch.cat([acc_cnt, torch.ones_like(vk, dtype=torch.float32)])
        uk, inv = torch.unique(all_k, return_inverse=True)
        acc_keys = uk
        acc_sum = torch.zeros(len(uk), 3, device=dev).index_add_(0, inv, all_c)
        acc_cnt = torch.zeros(len(uk), device=dev).index_add_(0, inv, all_n)
    for k, n_hit, g in report:
        print(f"  kf {k:3}: {n_hit:6} overlapped -> gain "
              f"[{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}]", flush=True)

    # --- held-out scoring of the COMPOSED map, before vs after ---
    lp = lpips_lib.LPIPS(net="alex").to(dev)
    K = ds.camera_intrinsics.K_frame

    def composed(rgb_list):
        means = torch.cat([p[0] for p in parts])
        scales = torch.cat([p[1] for p in parts])
        rots = torch.cat([p[2] for p in parts])
        rgb = torch.cat(rgb_list)
        opas = torch.cat([p[4] for p in parts])
        kf_id = torch.cat([p[5] for p in parts])
        A = kf_mats[kf_id, :3, :3]
        b = kf_mats[kf_id, :3, 3]
        mw = torch.einsum("mij,mj->mi", A, means) + b
        from utils.geometry import build_covariance
        q = rots / rots.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cw = A @ build_covariance(scales, q) @ A.transpose(1, 2)
        mses, lps = [], []
        with torch.no_grad():
            for c2w, tgt in held_frames:
                pred = render(mw, cw, rgb, opas, c2w, K,
                              tgt.shape[-2:], dev).clamp(0, 1)
                mses.append(torch.mean((pred - tgt) ** 2).item())
                lps.append(lp(pred * 2 - 1, tgt * 2 - 1).item())
        mse = sum(mses) / len(mses)
        return -10 * math.log10(max(mse, 1e-12)), sum(lps) / len(lps)

    raw_rgb = [rgb for _, rgb in worlds]
    p0, l0 = composed(raw_rgb)
    p1, l1 = composed(harm_rgb)
    print(f"\ncomposed map, raw:        psnr={p0:.4f}  lpips={l0:.4f}", flush=True)
    print(f"composed map, harmonized: psnr={p1:.4f}  lpips={l1:.4f}", flush=True)
    print(f"delta: {p1 - p0:+.4f} dB psnr, {l1 - l0:+.4f} lpips", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

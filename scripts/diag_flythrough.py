"""A metric that can see the veil: fly-through warp consistency.

Why this exists
---------------
The seams in the GUI captures are a semi-transparent veil -- a keyframe's
cluster placed at slightly wrong depth, floating in front of the true surface,
its silhouette the projected image rectangle (skill 17.4). Every attempt to fix
it moved psnr by thousandths of a dB, and 17.17 explains why with arithmetic
rather than excuses: held-out views sit 0.057 m from the nearest keyframe, and
at that baseline the residual depth error projects to 0.66 px. **The evaluation
protocol is structurally blind to depth error**, exactly as it was blind to the
dot lattice (17.2: psnr moved 0.01 dB across a sweep that changed the visible
artifact 13.6x).

Under that condition "many fixes were tried and none worked" and "the fixes
worked and the metric cannot see it" are the same observation. So the next step
for the seams is not another fix. It is a metric.

The signature, and why a warp finds it
--------------------------------------
A veil is two surfaces at different depths whose *colour* comes from the far one
and whose *geometry* comes from the near one. Move the camera and the front
layer carries the back layer's texture at its own wrong depth. So the defining
property is **colour motion inconsistent with depth motion** -- and that needs no
optical flow to detect, because the map hands us its own depth. Warp frame t
into frame t+1 using the rendered depth and the known relative pose: where the
geometry is right the warp matches, and where a veil sits it cannot.

Crucially this needs **no ground truth at all**. It is a self-consistency
measurement on the map, like the 2x self-consistency of 17.2, which is what
makes it immune to pose error, exposure, and held-out selection.

What is measured
----------------
Around each held-out pose, a small dolly (default +-5 cm along the camera's own
x axis, 5 frames). For each consecutive pair, backward-warp frame t into t+1
through t+1's rendered depth, and report the SSIM deficit on the pixels the warp
can reach. Reported alongside a STATIC control -- the same SSIM between frame
t+1 and itself shifted by the mean warp displacement -- so a scene that is simply
hard to warp does not read as a veil.

Usage:
    python3 scripts/diag_flythrough.py --ply <map>.ply \
        --traj logs/.../<seq>.txt --dataset datasets/tum/<seq> --n 12
"""
import argparse
import math
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


def ssim_map(a, b, win=7):
    """Local SSIM, returned per pixel rather than averaged, so it can be
    restricted to the pixels a warp actually reached."""
    pad = win // 2
    mu_a = F.avg_pool2d(F.pad(a, (pad,) * 4, mode="replicate"), win, 1)
    mu_b = F.avg_pool2d(F.pad(b, (pad,) * 4, mode="replicate"), win, 1)
    saa = F.avg_pool2d(F.pad(a * a, (pad,) * 4, mode="replicate"), win, 1) - mu_a ** 2
    sbb = F.avg_pool2d(F.pad(b * b, (pad,) * 4, mode="replicate"), win, 1) - mu_b ** 2
    sab = F.avg_pool2d(F.pad(a * b, (pad,) * 4, mode="replicate"), win, 1) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * sab + c2)) /
            ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))).mean(1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=12, help="held-out poses to probe")
    ap.add_argument("--dolly", type=float, default=0.05,
                    help="half-extent of the camera sweep, metres")
    ap.add_argument("--steps", type=int, default=5, help="frames per sweep")
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from eval_map_quality import decode_gaussians_from_ply, render_map

    load_config(args.config)
    dev = args.device
    ds = load_dataset(args.dataset)
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(args.traj)
    gt_ts, gt_T = load_tum_traj(os.path.join(args.dataset, "groundtruth.txt"))

    pairs = associate(est_ts, gt_ts)
    s_, R_, t_ = umeyama_sim3(np.array([est_T[i, :3, 3] for i, _ in pairs]),
                              np.array([gt_T[j, :3, 3] for _, j in pairs]))
    Rt = R_.T
    kf_idx = {j for _, j in associate(est_ts, ds_ts)}
    cand = [(i, j) for i, j in associate(ds_ts, gt_ts) if i not in kf_idx]
    held = cand[:: max(1, len(cand) // args.n)][: args.n]

    g = decode_gaussians_from_ply(args.ply, device=dev)
    K = ds.camera_intrinsics.K_frame
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    h, w = ds.get_img_shape()[0]
    from splatt3r_slam.splatt3r_utils import resize_img
    probe = resize_img(ds.get_image(held[0][0]), ds.img_size)["img"]
    h, w = probe.shape[-2:]

    def depth_render(c2w):
        """Alpha-weighted mean view-space depth, by rendering each Gaussian with
        its own z as colour. The standard 3DGS depth approximation; exact enough
        for a warp, and it is the map's OWN notion of depth, which is what the
        consistency test needs to be about."""
        w2c = np.linalg.inv(c2w)
        m = g["means"]
        z = (torch.as_tensor(w2c[:3, :3], dtype=torch.float32, device=dev)
             @ m.T).T[:, 2] + float(w2c[2, 3])
        gz = dict(g)
        gz["colors"] = (z / args.far).clamp(0, 1)[:, None].expand(-1, 3).contiguous()
        d = render_map(gz, c2w, K, (h, w), dev).reshape(1, 3, h, w)[:, :1]
        return d * args.far

    yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32),
                            indexing="ij")

    warp_def, static_def, disp_px = [], [], []
    with torch.no_grad():
        for di, gj in held:
            c2w_gt = gt_T[gj]
            base = np.eye(4)
            base[:3, :3] = Rt @ c2w_gt[:3, :3]
            base[:3, 3] = Rt @ (c2w_gt[:3, 3] - t_) / s_
            offs = np.linspace(-args.dolly, args.dolly, args.steps)
            views = []
            for o in offs:
                c = base.copy()
                c[:3, 3] = c[:3, 3] + base[:3, 0] * o   # slide along camera x
                views.append(c)
            imgs = [render_map(g, c, K, (h, w), dev).reshape(1, 3, h, w).clamp(0, 1)
                    for c in views]
            deps = [depth_render(c) for c in views]

            for i in range(len(views) - 1):
                A, B = views[i], views[i + 1]
                rel = np.linalg.inv(A) @ B          # B -> A
                Rr = torch.as_tensor(rel[:3, :3], dtype=torch.float32, device=dev)
                tr = torch.as_tensor(rel[:3, 3], dtype=torch.float32, device=dev)
                d = deps[i + 1][0, 0]
                valid = d > 1e-3
                X = (xx - cx) / fx * d
                Y = (yy - cy) / fy * d
                P = torch.stack([X, Y, d], -1) @ Rr.T + tr
                zp = P[..., 2].clamp_min(1e-6)
                u = fx * P[..., 0] / zp + cx
                v = fy * P[..., 1] / zp + cy
                inb = valid & (u > 0) & (u < w - 1) & (v > 0) & (v < h - 1)
                grid = torch.stack([u / (w - 1) * 2 - 1, v / (h - 1) * 2 - 1], -1)
                warped = F.grid_sample(imgs[i], grid[None], align_corners=True,
                                       padding_mode="border")
                m = inb[None, None].float()
                sm = ssim_map(warped, imgs[i + 1])
                n = m.sum().clamp_min(1)
                warp_def.append(float(((1 - sm) * m).sum() / n))
                # Static control: the same comparison with a rigid shift equal to
                # the mean displacement. A scene that is simply hard to match
                # scores badly here too, and only the DIFFERENCE is the veil.
                du = float(((u - xx)[inb]).mean()) if int(inb.sum()) else 0.0
                dv = float(((v - yy)[inb]).mean()) if int(inb.sum()) else 0.0
                gs = torch.stack([(xx + du) / (w - 1) * 2 - 1,
                                  (yy + dv) / (h - 1) * 2 - 1], -1)
                shifted = F.grid_sample(imgs[i], gs[None], align_corners=True,
                                        padding_mode="border")
                sm2 = ssim_map(shifted, imgs[i + 1])
                static_def.append(float(((1 - sm2) * m).sum() / n))
                disp_px.append((du ** 2 + dv ** 2) ** 0.5)

    wd = sum(warp_def) / len(warp_def)
    sd = sum(static_def) / len(static_def)
    print(f"\n{args.ply}")
    print(f"  probes {len(held)} poses x {args.steps - 1} pairs, "
          f"dolly +-{args.dolly * 100:.0f} cm, mean disparity "
          f"{sum(disp_px) / len(disp_px):.1f} px")
    print(f"  warp SSIM deficit            {wd:.4f}   <- depth-aware warp")
    print(f"  static-shift SSIM deficit    {sd:.4f}   <- rigid control")
    print(f"  ratio warp/static            {wd / max(sd, 1e-9):.4f}")
    print("\n  A map whose geometry is right warps BETTER than a rigid shift, so"
          "\n  the ratio should be well below 1. Ratio near or above 1 means the"
          "\n  rendered depth does not explain the rendered colour's motion --"
          "\n  which is the definition of a veil.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

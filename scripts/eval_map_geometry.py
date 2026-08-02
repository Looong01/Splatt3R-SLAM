"""Geometric evaluation of a persisted Gaussian map against ground-truth depth.

Why this exists
---------------
Every map-quality number in this project so far is photometric (psnr / lpips,
§9.2). That is only half the evidence for a claim about *mapping*. The
fine-tuned head produces maps with **16% fewer Gaussians and higher psnr**,
which is consistent with two very different stories:

  (a) it removes low-confidence junk primitives and keeps the geometry, or
  (b) it drops real structure and the photometric score rises anyway because
      what remains is easier to shade.

Photometric metrics cannot separate those. Depth can. TUM ships ground-truth
depth per frame, so the map's rendered depth can be compared against it
directly.

Metrics, all computed only where ground-truth depth is valid (TUM encodes
missing returns as 0) and where the map actually rendered something:

  depth L1     mean |d_pred - d_gt|, in metres
  AbsRel       mean |d_pred - d_gt| / d_gt  -- scale-aware, comparable across
               scenes with different depth ranges
  delta<1.25   fraction of pixels within a 25% relative error (the standard
               monocular-depth accuracy threshold)
  completeness fraction of valid ground-truth pixels the map rendered at all

Completeness is reported alongside accuracy on purpose: a map can win on L1 by
rendering only the easy pixels, and the pair of numbers makes that visible.

Scale: the map lives in the SLAM estimate's frame, which is metric only up to
the Sim3 fit against ground truth. The alignment scale from that fit is applied
to rendered depths before comparison -- the same alignment eval_map_quality.py
uses, so the two evaluations are consistent.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_map_geometry.py \
        --ply logs/head_ate_head/rgbd_dataset_freiburg1_room_gaussians.ply \
        --traj logs/head_ate_head/rgbd_dataset_freiburg1_room.txt \
        --dataset datasets/tum/rgbd_dataset_freiburg1_room --n 50
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, os.path.join(CORE, "src", "pixelsplat_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))

import cv2
import numpy as np
import torch

from eval_map_quality import associate, load_tum_traj, umeyama_sim3
from splatt3r_slam.gaussian_ply_codec import decode_gaussians_from_ply

TUM_DEPTH_SCALE = 5000.0   # TUM stores depth as uint16 millimetres/5
DELTA_THRESHOLD = 1.25


def resize_depth_like_rgb(depth_m, size=512):
    """Apply the SAME geometric transform `resize_img` applies to RGB, but to a
    metric depth map.

    `resize_img` cannot be reused directly: it does `PIL.Image.fromarray(
    np.uint8(img * 255))`, which assumes the input is in [0,1] and quantizes to
    8 bits. Feeding it depth in metres silently rescales 0.53-2.56 m onto
    [-1,1] and the resulting "depth" is meaningless -- an earlier version did
    exactly that and reported AbsRel = 8.80.

    Geometry replicated from splatt3r_utils.resize_img / _resize_pil_image:
    resize the long edge to `size`, then centre-crop to even multiples of 8.
    Interpolation is NEAREST on purpose -- averaging depth across an occlusion
    boundary invents a surface that exists in neither the foreground nor the
    background.
    """
    h1, w1 = depth_m.shape[:2]
    s = size / max(w1, h1)
    w = int(round(w1 * s))
    h = int(round(h1 * s))
    out = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

    cx, cy = w // 2, h // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if w == h:
        halfh = int(3 * halfw / 4)
    return out[cy - halfh:cy + halfh, cx - halfw:cx + halfw]


@torch.no_grad()
def render_depth(g, c2w, K, hw, device):
    """Render the map's expected depth by splatting each Gaussian's camera-space
    z as its colour.

    The rasterizer has no depth output, but it is a weighted compositor: feeding
    z in place of RGB and dividing by the same compositing weight yields the
    alpha-weighted mean depth. The weight image comes from splatting a constant
    1.0 the same way, so both share every other setting.
    """
    from src.pixelsplat_src.cuda_splatting import render_cuda

    h, w = hw
    ext = torch.as_tensor(c2w, dtype=torch.float32, device=device)[None]
    intr = torch.as_tensor(K, dtype=torch.float32, device=device)[None].clone()
    intr[:, 0, :] /= w
    intr[:, 1, :] /= h

    w2c = torch.inverse(ext[0])
    cam_z = (g["means"] @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]

    def splat(values):
        # values (G,) -> (1, G, 3, 1) constant across channels, no SH rotation
        sh = values[:, None].expand(-1, 3).contiguous()[None][..., None]
        return render_cuda(
            ext, intr,
            torch.full((1,), 0.1, device=device),
            torch.full((1,), 1000.0, device=device),
            (h, w), torch.zeros((1, 3), device=device),
            g["means"][None], g["covariances"][None], sh, g["opacity"][None],
            use_sh=False,
        ).reshape(3, h, w)[0]

    num = splat(cam_z)
    den = splat(torch.ones_like(cam_z))
    covered = den > 1e-3
    depth = torch.where(covered, num / den.clamp_min(1e-6), torch.zeros_like(num))
    return depth, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img

    load_config(args.config)
    dev = args.device
    dataset = load_dataset(args.dataset)
    ds_ts = np.array([float(t) for t in dataset.timestamps])

    est_ts, est_T = load_tum_traj(args.traj)
    gt_ts, gt_T = load_tum_traj(os.path.join(args.dataset, "groundtruth.txt"))

    pairs = associate(est_ts, gt_ts)
    s, R, t = umeyama_sim3(
        np.array([est_T[i, :3, 3] for i, _ in pairs]),
        np.array([gt_T[j, :3, 3] for _, j in pairs]))
    Rt = R.T

    # Depth frames, associated to rgb by timestamp (TUM ships them separately).
    dep_file = os.path.join(args.dataset, "depth.txt")
    raw = np.loadtxt(dep_file, dtype=np.str_)
    dep_ts = raw[:, 0].astype(np.float64)
    dep_paths = [os.path.join(args.dataset, p) for p in raw[:, 1]]

    kf_idx = {j for _, j in associate(est_ts, ds_ts)}
    gt_pairs = associate(ds_ts, gt_ts)
    held = [(i, j) for i, j in gt_pairs if i not in kf_idx]
    held = held[:: max(1, len(held) // args.n)][: args.n]

    g = decode_gaussians_from_ply(args.ply, device=dev)
    K = dataset.camera_intrinsics.K_frame
    print(f"map: {g['n']:,} gaussians   scoring {len(held)} held-out frames", flush=True)

    tot = {"l1": 0.0, "absrel": 0.0, "delta": 0.0, "cover": 0.0}
    n = 0
    for di, gj in held:
        k = associate(np.array([ds_ts[di]]), dep_ts)
        if not k:
            continue
        gt_depth_raw = cv2.imread(dep_paths[k[0][1]], cv2.IMREAD_UNCHANGED)
        if gt_depth_raw is None:
            continue
        # Put GT depth through the identical resize/crop the RGB goes through,
        # so pixels correspond. INTER_NEAREST: averaging depth across a
        # discontinuity invents surfaces that exist in neither frame.
        gt_m = gt_depth_raw.astype(np.float32) / TUM_DEPTH_SCALE
        rgb = resize_img(dataset.get_image(di), dataset.img_size)["img"]
        h, w = rgb.shape[-2:]
        gt_t = torch.as_tensor(resize_depth_like_rgb(gt_m, dataset.img_size),
                               device=dev)
        assert gt_t.shape == (h, w), (gt_t.shape, (h, w))

        c2w_gt = gt_T[gj]
        c2w_map = np.eye(4)
        c2w_map[:3, :3] = Rt @ c2w_gt[:3, :3]
        c2w_map[:3, 3] = Rt @ (c2w_gt[:3, 3] - t) / s

        pred, covered = render_depth(g, c2w_map, K, (h, w), dev)
        pred_metric = pred * s   # map units -> ground-truth scale

        valid = (gt_t > 1e-3) & covered
        if valid.sum() < 100:
            continue
        d, gtv = pred_metric[valid], gt_t[valid]
        tot["l1"] += (d - gtv).abs().mean().item()
        tot["absrel"] += ((d - gtv).abs() / gtv).mean().item()
        ratio = torch.maximum(d / gtv, gtv / d)
        tot["delta"] += (ratio < DELTA_THRESHOLD).float().mean().item()
        tot["cover"] += (valid.sum() / (gt_t > 1e-3).sum()).item()
        n += 1

    if n == 0:
        raise SystemExit("no frames scored -- check depth association")
    print(f"\n  geometry | L1={tot['l1']/n:.4f} m  AbsRel={tot['absrel']/n:.4f}  "
          f"delta<1.25={tot['delta']/n*100:.1f}%  completeness={tot['cover']/n*100:.1f}%  "
          f"(n={n})", flush=True)


if __name__ == "__main__":
    main()

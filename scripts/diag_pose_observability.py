"""Is the pose even observable from this map's photometric loss?

The controlled injection test showed zero recovery in every arm. Two readings
survive it: the optimizer is broken, or the loss has no gradient to follow.
This script separates them without training anything — it just walks the camera
away from its pose and watches the loss.

Direct-method observability (DTAM/LSD-SLAM lineage) needs
`photometric signal ≈ displacement × image gradient magnitude` to rise above the
residual floor. Our map renders at 19-20 dB, i.e. an RMS residual around 0.11,
and its high frequencies are smoothed away. If a 4 cm displacement produces a
loss change buried under that floor, pose is simply unobservable at our
operating point — and no learning rate, iteration count or gradient
implementation can change that.

Output is the loss as a function of displacement magnitude, together with the
loss change caused by the same displacement measured against the *residual
floor*. A curve that is flat below ~4 cm demonstrates unobservability at the
operating point directly, with no extrapolation.

Usage:
    python3 scripts/diag_pose_observability.py --ply logs/map_gtposes_desk.ply \
        --traj logs/frames_head/rgbd_dataset_freiburg1_desk.txt \
        --dataset datasets/tum/rgbd_dataset_freiburg1_desk
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(CORE, "src", "pixelsplat_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))

import numpy as np
import torch

from eval_map_quality import NEAR, FAR, associate, load_tum_traj, umeyama_sim3

DISPLACEMENTS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)


@torch.no_grad()
def render(g, c2w, K, hw, device):
    from src.pixelsplat_src.cuda_splatting import render_cuda
    h, w = hw
    ext = torch.as_tensor(c2w, dtype=torch.float32, device=device)[None]
    intr = torch.as_tensor(K, dtype=torch.float32, device=device)[None].clone()
    intr[:, 0, :] /= w
    intr[:, 1, :] /= h
    img = render_cuda(
        ext, intr,
        torch.full((1,), NEAR, device=device), torch.full((1,), FAR, device=device),
        (h, w), torch.zeros((1, 3), device=device),
        g["means"][None], g["covariances"][None],
        (g["f_dc"][:, :, None])[None], g["opacity"].reshape(-1)[None],
        use_sh=True)
    return img.reshape(1, 3, h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=20, help="views to average over")
    ap.add_argument("--dirs", type=int, default=6, help="random directions per view")
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.gaussian_ply_codec import decode_gaussians_from_ply
    from refine_gaussian_map import uniform_subsample

    load_config(args.config)
    os.chdir(CORE)
    dev = args.device
    ds = load_dataset(os.path.join(REPO_ROOT, args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(os.path.join(REPO_ROOT, args.traj))
    gt_ts, gt_T = load_tum_traj(os.path.join(REPO_ROOT, args.dataset, "groundtruth.txt"))

    pairs = associate(est_ts, gt_ts)
    s, R, t = umeyama_sim3(np.array([est_T[i, :3, 3] for i, _ in pairs]),
                           np.array([gt_T[j, :3, 3] for _, j in pairs]))
    Rt = R.T

    def to_map(c2w_gt):
        m = np.eye(4)
        m[:3, :3] = Rt @ c2w_gt[:3, :3]
        m[:3, 3] = Rt @ (c2w_gt[:3, 3] - t) / s
        return m

    kf_idx = {j for _, j in associate(est_ts, ds_ts)}
    gt_pairs = associate(ds_ts, gt_ts)
    non_kf = [(i, j) for i, j in gt_pairs if i not in kf_idx]
    held = non_kf[:: max(1, len(non_kf) // 50)][:50]
    held_set = {i for i, _ in held}
    pool = [(i, j) for i, j in gt_pairs if i not in held_set]
    views = uniform_subsample(pool, args.n)

    g = decode_gaussians_from_ply(os.path.join(REPO_ROOT, args.ply), device=dev)
    K = ds.camera_intrinsics.K_frame
    rng = np.random.default_rng(0)

    print(f"\n=== {args.ply} ===")
    print(f"  {len(views)} views x {args.dirs} random directions, no training\n")
    print(f"  {'|dt| (m)':>10} {'mse':>10} {'psnr':>8} {'Δmse vs 0':>12} "
          f"{'Δmse / mse0':>12}")

    base_mse = None
    for disp in DISPLACEMENTS:
        acc = []
        for di, gj in views:
            img = resize_img(ds.get_image(di), ds.img_size)["img"]
            tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
            c2w0 = to_map(gt_T[gj])
            n_dirs = 1 if disp == 0.0 else args.dirs
            for _ in range(n_dirs):
                c2w = c2w0.copy()
                if disp > 0:
                    v = rng.normal(size=3)
                    c2w[:3, 3] = c2w[:3, 3] + disp * v / np.linalg.norm(v)
                pred = render(g, c2w, K, tgt.shape[-2:], dev).clamp(0, 1)
                acc.append(torch.mean((pred - tgt) ** 2).item())
        mse = sum(acc) / len(acc)
        if base_mse is None:
            base_mse = mse
        print(f"  {disp:>10.3f} {mse:>10.5f} {-10*math.log10(mse):>8.3f} "
              f"{mse-base_mse:>+12.5f} {(mse-base_mse)/base_mse:>+12.2%}")

    print("\n  READING: the loss must RISE with displacement for pose to be")
    print("  recoverable. A flat region below the operating point (~0.024 m of")
    print("  real pose error, ~0.039 m injected) means the photometric loss")
    print("  carries no usable signal there, and no optimizer can find it.")


if __name__ == "__main__":
    sys.exit(main())

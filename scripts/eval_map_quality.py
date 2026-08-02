"""Score a persisted Gaussian map by re-rendering it from held-out camera poses.

Why this exists
---------------
Everything measured so far in the fine-tuning series scores a SINGLE two-view
prediction on a validation pair. That is not the deployed objective. In SLAM,
dozens to hundreds of keyframes are fused into one world map, so the map's
coverage is far better than any individual two-view prediction's, and the
question "does the +1.78 dB per-pair gain survive fusion?" cannot be answered
by more per-pair metrics. It has to be measured on the map.

ATE is the other half of the SLAM-level check (`scripts/eval_head_ate.sh`),
but it is a CONTROL here, not the measurement: head-only training leaves the
encoder bit-identical, so matching and tracking are unchanged by construction
and ATE should not move at all. Any gain must show up in map quality, which is
what this script measures.

Protocol
--------
1. Load the map written by `evaluate.save_gaussian_map` (standard 3DGS .ply).
2. Load the SLAM keyframe trajectory and the dataset ground truth.
3. Sim3-align the estimated trajectory to ground truth (Umeyama, with scale --
   monocular SLAM is scale-free, and the map lives in the estimate's frame).
   Invert that to bring ground-truth poses INTO the map frame.
4. Evaluate on frames the SLAM run never selected as keyframes, so this is
   novel-view synthesis rather than a self-consistency check.
5. Render the map from each held-out pose and score psnr/lpips against the
   real image.

Both arms (base vs. fine-tuned head) get identical treatment, so the
comparison is fair even though each arm's map lives in its own world frame.

Usage:
    python3 scripts/eval_map_quality.py \
        --ply logs/<run>/<seq>_gaussians.ply \
        --traj logs/<run>/<seq>.txt \
        --dataset datasets/tum/rgbd_dataset_freiburg1_room \
        --n 100
"""
import argparse
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, os.path.join(CORE, "src", "pixelsplat_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))

import numpy as np
import torch

from splatt3r_slam.gaussian_ply_codec import decode_gaussians_from_ply

NEAR, FAR = 0.1, 1000.0   # matches DecoderSplattingCUDA's training values


def load_tum_traj(path):
    """TUM-format trajectory file -> (timestamps, (N,4,4) c2w)."""
    from scipy.spatial.transform import Rotation

    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None]
    ts = raw[:, 0]
    t = raw[:, 1:4]
    q_xyzw = raw[:, 4:8]
    R = Rotation.from_quat(q_xyzw).as_matrix()
    T = np.tile(np.eye(4), (len(ts), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return ts, T


def umeyama_sim3(src, dst):
    """Similarity transform (s, R, t) mapping src points onto dst.

    Umeyama 1991. Scale is estimated, not assumed: monocular SLAM recovers
    geometry only up to scale, so a rigid-only alignment would fold that
    unknown into the residual and make every rendering look wrong.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = float((sig * np.diag(W)).sum() / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def associate(ts_a, ts_b, max_dt=0.02):
    """Greedy nearest-timestamp association, returns index pairs."""
    order = np.argsort(ts_b)
    sb = ts_b[order]
    pairs = []
    for i, t in enumerate(ts_a):
        j = np.searchsorted(sb, t)
        cands = [k for k in (j - 1, j) if 0 <= k < len(sb)]
        if not cands:
            continue
        k = min(cands, key=lambda k: abs(sb[k] - t))
        if abs(sb[k] - t) <= max_dt:
            pairs.append((i, int(order[k])))
    return pairs


@torch.no_grad()
def render_map(g, c2w, K, hw, device):
    """Render the whole map from one camera pose."""
    from src.pixelsplat_src.cuda_splatting import render_cuda

    h, w = hw
    extrinsics = torch.as_tensor(c2w, dtype=torch.float32, device=device)[None]
    intrinsics = torch.as_tensor(K, dtype=torch.float32, device=device)[None].clone()
    # render_cuda wants normalized intrinsics (fx/W, fy/H, cx/W, cy/H).
    intrinsics[:, 0, :] /= w
    intrinsics[:, 1, :] /= h

    sh = (g["rgb"] - 0.5) / 0.28209479177387814  # RGB -> SH band 0
    return render_cuda(
        extrinsics,
        intrinsics,
        torch.full((1,), NEAR, device=device),
        torch.full((1,), FAR, device=device),
        (h, w),
        torch.zeros((1, 3), device=device),
        g["means"][None],
        g["covariances"][None],
        sh[None][..., None],
        g["opacity"][None],
        use_sh=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--traj", required=True, help="SLAM keyframe trajectory (TUM format)")
    ap.add_argument("--dataset", required=True, help="sequence directory")
    ap.add_argument("--n", type=int, default=100, help="held-out frames to score")
    ap.add_argument("--dump-renders", default=None,
                    help="write GT/render PNG pairs here. The standing rule in "
                         "this project is that no verdict is believed on "
                         "scalars alone -- images caught the encoder-LoRA "
                         "failure when metrics did not -- yet the refined map's "
                         "19 dB had never been looked at.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default="config/eval_calib.yaml",
                    help="must match the config the SLAM run used, since it "
                         "sets img_size and use_calib and hence the intrinsics")
    args = ap.parse_args()

    import lpips as lpips_lib
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img

    # The dataset classes read the global config (use_calib, img_size), which
    # only main.py normally populates.
    load_config(args.config)  # updates the global config in place

    dev = args.device
    dataset = load_dataset(args.dataset)
    ds_ts = np.array([float(t) for t in dataset.timestamps])

    est_ts, est_T = load_tum_traj(args.traj)

    gt_path = os.path.join(args.dataset, "groundtruth.txt")
    if not os.path.exists(gt_path):
        raise SystemExit(f"no ground truth at {gt_path}")
    gt_ts, gt_T = load_tum_traj(gt_path)

    # --- align estimate -> ground truth, then invert to map frame ----------
    pairs = associate(est_ts, gt_ts)
    if len(pairs) < 3:
        raise SystemExit(f"only {len(pairs)} est/gt timestamp matches; cannot align")
    src = np.array([est_T[i, :3, 3] for i, _ in pairs])
    dst = np.array([gt_T[j, :3, 3] for _, j in pairs])
    s, R, t = umeyama_sim3(src, dst)
    print(f"aligned on {len(pairs)} poses: scale={s:.4f}", flush=True)

    # Invert  X_gt = s R X_map + t  ->  X_map = R^T (X_gt - t) / s.
    # The 1/s belongs to the TRANSLATION only. Folding it into the rotation
    # block (R.T / s) yields a non-orthonormal "rotation", which silently
    # corrupts every rendered viewpoint.
    Rt = R.T

    # --- held-out frames: dataset frames that are NOT keyframes ------------
    kf_pairs = associate(est_ts, ds_ts)
    kf_idx = {j for _, j in kf_pairs}
    gt_pairs = associate(ds_ts, gt_ts)
    candidates = [(i, j) for i, j in gt_pairs if i not in kf_idx]
    if not candidates:
        raise SystemExit("no held-out frames with ground truth")
    step = max(1, len(candidates) // args.n)
    held = candidates[::step][: args.n]
    print(f"{len(kf_idx)} keyframes, scoring {len(held)} held-out frames", flush=True)

    g = decode_gaussians_from_ply(args.ply, device=dev)
    print(f"map: {g['n']:,} gaussians", flush=True)

    K_frame = dataset.camera_intrinsics.K_frame
    lp = lpips_lib.LPIPS(net="alex").to(dev)

    tm = tl = 0.0
    n = 0
    for di, gj in held:
        # resize_img applies ImgNorm = Normalize(0.5, 0.5), i.e. it returns
        # the image in [-1,1], not [0,1]. The rasterizer outputs [0,1], so the
        # target has to be brought back before any metric is computed --
        # comparing the two spaces directly scored 3.16 dB on a map that
        # renders correctly.
        img = resize_img(dataset.get_image(di), dataset.img_size)["img"]  # (1,3,H,W)
        target = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
        h, w = target.shape[-2:]

        c2w_gt = gt_T[gj]
        c2w_map = np.eye(4)
        c2w_map[:3, :3] = Rt @ c2w_gt[:3, :3]
        c2w_map[:3, 3] = Rt @ (c2w_gt[:3, 3] - t) / s

        pred = render_map(g, c2w_map, K_frame, (h, w), dev)
        pred = pred.reshape(1, 3, h, w).clamp(0, 1)

        if args.dump_renders and n < 6:
            import cv2 as _cv
            os.makedirs(args.dump_renders, exist_ok=True)
            for nm, im in (("gt", target), ("render", pred)):
                a = (im[0].detach().float().clamp(0, 1).cpu().numpy()
                     .transpose(1, 2, 0) * 255).round().astype(np.uint8)
                _cv.imwrite(os.path.join(args.dump_renders, f"s{n}_{nm}.png"),
                            _cv.cvtColor(a, _cv.COLOR_RGB2BGR))
        tm += torch.mean((pred - target) ** 2).item()
        tl += lp(pred, target, normalize=True).mean().item()
        n += 1

    mse = tm / n
    print(f"\n  map | mse={mse:.4f}  psnr={-10 * math.log10(mse):.4f}  "
          f"lpips={tl / n:.4f}  (n={n})", flush=True)


if __name__ == "__main__":
    main()

"""Why does the offline ceiling sit at ~19-20 dB even with ground-truth poses?

Published per-scene 3DGS reaches 25-35 dB. This project's oracle -- ground-truth
poses, up to 400 supervision views, 30000 iterations with densification --
saturates at 19.13/19.95. Something caps it well below what the method is known
to do, and which thing it is changes what to build next:

  (a) the EVALUATION caps it. Held-out views are scored by mapping their
      ground-truth pose into the map's frame through a Sim3 fitted on the
      keyframe trajectory. That fit has residual error, so every held-out
      render is slightly mis-posed no matter how good the map is -- a ceiling
      that no amount of map quality can lift, and one that would also make the
      "4.5 dB pose gap" partly an artifact and the pose-recovery failure partly
      a measurement problem.

  (b) the REPRESENTATION or the optimization caps it. Then 19-20 dB is what
      this map can do, and the pose work is aimed at a real target.

The discriminator is cheap: score the map on the very views it was trained on,
with the very poses it was trained at. Training views need no Sim3 mapping if
they were supervised at mapped ground-truth poses -- they carry exactly the
same alignment error as the held-out ones -- so:

  train ≈ held-out ≈ 19       -> the cap is NOT generalization. Either the
                                 representation saturates, or the shared
                                 alignment error caps both. Distinguished by
                                 the third number below.
  train >> held-out           -> ordinary overfitting/coverage; the ceiling is
                                 about view count and the map is fine.

Third number, which separates (a) from (b) directly: render a training view at
its mapped ground-truth pose versus at the pose the optimizer would have used
had there been no Sim3 error at all. We cannot know the latter, but we can
bound the effect by measuring how much the score changes under a perturbation
the size of the Sim3 fit residual -- reported here as the alignment residual so
the two can be compared.

Usage:
    python3 scripts/diag_ceiling.py --ply logs/map_gtposes_desk.ply \
        --traj logs/frames_head/rgbd_dataset_freiburg1_desk.txt \
        --dataset datasets/tum/rgbd_dataset_freiburg1_desk --n-train 50
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
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--n-held", type=int, default=50)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # load_config resolves its `inherit:` chain relative to the CWD, so it has
    # to run from the repo root; the render path wants CORE on the CWD.
    os.chdir(REPO_ROOT)
    import lpips as lpips_lib
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
    src = np.array([est_T[i, :3, 3] for i, _ in pairs])
    dst = np.array([gt_T[j, :3, 3] for _, j in pairs])
    s, R, t = umeyama_sim3(src, dst)
    # How well does the Sim3 actually fit? This residual is the floor on how
    # accurately ANY ground-truth pose can be placed in the map's frame.
    resid = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
    Rt = R.T

    def to_map(c2w_gt):
        m = np.eye(4)
        m[:3, :3] = Rt @ c2w_gt[:3, :3]
        m[:3, 3] = Rt @ (c2w_gt[:3, 3] - t) / s
        return m

    kf_idx = {j for _, j in associate(est_ts, ds_ts)}
    gt_pairs = associate(ds_ts, gt_ts)
    non_kf = [(i, j) for i, j in gt_pairs if i not in kf_idx]
    held_c = non_kf[:: max(1, len(non_kf) // args.n_held)][: args.n_held]
    held_set = {i for i, _ in held_c}
    pool = [(i, j) for i, j in gt_pairs if i not in held_set]
    train_c = uniform_subsample(pool, args.n_train)

    g = decode_gaussians_from_ply(os.path.join(REPO_ROOT, args.ply), device=dev)
    K = ds.camera_intrinsics.K_frame
    lp = lpips_lib.LPIPS(net="alex").to(dev)

    def score(cands, label):
        mses, lps = [], []
        for di, gj in cands:
            img = resize_img(ds.get_image(di), ds.img_size)["img"]
            tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
            pred = render(g, to_map(gt_T[gj]), K, tgt.shape[-2:], dev).clamp(0, 1)
            mses.append(torch.mean((pred - tgt) ** 2).item())
            lps.append(lp(pred * 2 - 1, tgt * 2 - 1).item())
        mse = sum(mses) / len(mses)
        # -10*log10(mse), matching eval_map_quality.py:245 and
        # refine_gaussian_map.py:528. `mse` here is the mean over ALL elements
        # (channels included), so there is no /3 -- an earlier version of this
        # script carried one over from a context where mse was a per-pixel SUM,
        # and inflated every number by exactly 10*log10(3) = 4.7712 dB.
        print(f"  {label:28s} n={len(cands):3d}  mse={mse:.4f}  "
              f"psnr={-10*math.log10(mse):7.4f}  lpips={sum(lps)/len(lps):.4f}")
        return -10 * math.log10(mse)

    print(f"\n=== {args.ply} ===")
    print(f"  Sim3 fit residual over {len(pairs)} keyframes: "
          f"mean {resid.mean():.4f} m  median {np.median(resid):.4f} m  "
          f"max {resid.max():.4f} m")
    print(f"  (this is the floor on placing ANY ground-truth pose in the map "
          f"frame; compare it to the 0.024 m pose error the refinement chases)")
    ptr = score(train_c, "TRAIN views (supervised)")
    phe = score(held_c, "HELD-OUT views")
    print(f"\n  train - held-out = {ptr - phe:+.3f} dB")
    print("  READING: a large gap means the ceiling is generalization/coverage.")
    print("           a small gap means the map cannot do better even where it "
          "was fitted -- the cap is the representation or the alignment, not "
          "the view count.")


if __name__ == "__main__":
    sys.exit(main())

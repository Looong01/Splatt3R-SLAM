"""Is the dot lattice a map defect or a display artifact -- and what fixes it?

The question
------------
Every GUI capture of the 360 map (logs/gs_view_360/gs_map_*.png) is covered in
a regular dot/halftone pattern, worst on flat surfaces, with moire where the
pattern beats against the output pixel grid, and it degrades badly on the
close-up frames near the end of the sequence. None of it appears in the
held-out psnr/lpips this project optimizes.

Two readings, with opposite consequences:

  (a) display artifact -- the GUI renders above the map's native sampling rate,
      and the map is fine at the rate it was built for. Then the right response
      is to say so and change nothing.
  (b) representation defect -- Splatt3R sizes each Gaussian to its own pixel
      footprint at the source view, so the surface is only just covered THERE.
      The alpha between neighbouring Gaussians really is ~0, the map has no
      reconstruction filter, and the native-rate metric simply cannot see it
      because its sample points land on the lattice centres.

The mechanism, corrected
------------------------
The rasterizer point-samples each Gaussian at the pixel centre; there is no
area integration. With lattice pitch dx and effective std sigma, the relative
amplitude of the lattice's first harmonic in a flat region is

    exp(-2 pi^2 sigma^2 / dx^2)

At the source view and resolution the sample points sit on the lattice centres
and the harmonic is invisible. Magnify, shift, or resample and it enters the
output Nyquist band -- visible dots; at a non-integer ratio it beats against
the pixel grid -- moire. So the governing quantity is sigma/dx, which is a
property of the MAP and independent of the view; only its visibility depends
on magnification. (An earlier version of this argument had it that a closer
view covers fewer output pixels per Gaussian, which is backwards -- closer
means larger projections. The frequency argument is the right one, and it is
what predicts the sweep below.)

That also calibrates the fix: setting sigma = tau * dx gives a residual
modulation of exp(-2 pi^2 tau^2) -- 11% at tau=0.3, 0.7% at tau=0.5, 0.02% at
tau=0.65. So tau ~ 0.5-0.7 should be enough, and tau=1 buys nothing but blur.

What is measured
----------------
E1, which needs no ground truth to be decisive: render every held-out view at
1x and at 2x, average-pool the 2x back to native, and compare.

  - If (a), the two agree closely: the map has no structure between the native
    sample points, so integrating over them changes nothing.
  - If (b), they disagree, and the 2x-pooled render is DARKER and scores worse
    against ground truth -- because pooling sees the gaps that native-rate
    sampling steps over.

The self-consistency psnr (1x vs 2x-pooled) is the headline: it needs no GT at
all, so it cannot be confounded by pose error, exposure, or held-out choice.

Q7 is settled in the same run. Opacity is one scalar per Gaussian, uniform over
its whole footprint, so it can only darken uniformly -- producing a *pattern*
requires spatial modulation of alpha, which can only come from coverage. The
accumulated-alpha render separates them: a lattice in the alpha channel means
coverage; smooth alpha well below 1 means opacity.

Usage:
    python3 scripts/diag_lattice.py \
        --kfgauss logs/frames_head/rgbd_dataset_freiburg1_360_kfgauss.pt \
        --traj    logs/frames_head/rgbd_dataset_freiburg1_360.txt \
        --frames-traj logs/frames_head/rgbd_dataset_freiburg1_360_frames.txt \
        --dataset datasets/tum/rgbd_dataset_freiburg1_360 \
        --aa-sigma 0 0.3 0.5 0.7 1.0
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


def psnr(a, b):
    return -10 * math.log10(max(float(torch.mean((a - b) ** 2)), 1e-12))


def highpass_std(img, flat):
    """Std of (img - 3x3 box blur) over the pixels `flat` marks.

    The lattice lives at the highest representable frequency, so a small
    high-pass is the direct read-out of its amplitude. Restricted to flat
    regions because real texture also lives there and would swamp it.
    """
    blur = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    hp = (img - blur).abs().mean(1, keepdim=True)
    m = flat.float()
    n = m.sum().clamp_min(1)
    mu = (hp * m).sum() / n
    return float(((hp - mu) ** 2 * m).sum() / n) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames-traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-held", type=int, default=25)
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--aa-sigma", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--aa-compensate-opacity", action="store_true")
    ap.add_argument("--super", type=int, default=2, dest="ss",
                    help="supersampling factor for the E1 comparison")
    ap.add_argument("--save-crops", default=None, metavar="DIR",
                    help="write a 256x256 crop of the 1x render per tau, for "
                         "the figure")
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    import lpips as lpips_lib
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.refiner import (LocalGaussianMap, sim3_to_mat,
                                       gaussians_from_keyframe, render_map)

    load_config(args.config)
    os.chdir(CORE)
    dev = args.device
    torch.manual_seed(0)
    lp = lpips_lib.LPIPS(net="alex").to(dev)

    ds = load_dataset(os.path.join(REPO_ROOT, args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(os.path.join(REPO_ROOT, args.traj))
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

    # Same held-out construction as refine_local.py / refine_gaussian_map.py,
    # so these numbers sit on the same axis as every other result in the skill.
    kf_idx = {j for _, j in associate(est_ts, ds_ts)}
    gt_pairs = associate(ds_ts, gt_ts)
    non_kf = [(i, j) for i, j in gt_pairs if i not in kf_idx]
    held_c = non_kf[:: max(1, len(non_kf) // args.n_held)][: args.n_held]
    held = []
    for di, gj in held_c:
        img = resize_img(ds.get_image(di), ds.img_size)["img"]
        tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
        held.append((to_map(gt_T[gj]), tgt))
    print(f"held-out {len(held)} views", flush=True)

    blob = torch.load(os.path.join(REPO_ROOT, args.kfgauss), map_location="cpu")
    K = torch.as_tensor(ds.camera_intrinsics.K_frame, dtype=torch.float32, device=dev)

    print(f"\n{'tau':>5} {'n_gauss':>10} {'psnr_1x':>8} {'psnr_ss':>8} "
          f"{'self':>7} {'lpips':>7} {'hp_flat':>8} {'alpha':>7} {'hp_alpha':>9}",
          flush=True)

    for tau in args.aa_sigma:
        parts, kf_pose_data = [], []
        for k, kf in enumerate(blob["keyframes"]):
            h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
            local = {key: kf[key].to(dev) for key in
                     ("means", "scales", "rotations", "sh", "opacities", "conf")}
            got = gaussians_from_keyframe(
                local, kf["img"].to(dev), h, w, k, dev,
                min_confidence=args.min_confidence, aa_sigma_scale=tau,
                aa_compensate_opacity=args.aa_compensate_opacity)
            if got is None:
                continue
            parts.append(got)
            kf_pose_data.append(kf["T_WC"].to(dev))
        kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in kf_pose_data]))
        model = LocalGaussianMap(*[torch.cat([p[i] for p in parts])
                                   for i in range(6)]).to(dev)

        with torch.no_grad():
            mw, cw = model.world(kf_mats)
            rgb_, op_ = model.rgb(), model.opacity()
            ones = torch.ones_like(rgb_)
            p1, pss, pself, lps, hps, alphas, hpa = [], [], [], [], [], [], []
            for vi, (c2w, tgt) in enumerate(held):
                h, w = tgt.shape[-2:]
                r1 = render_map(mw, cw, rgb_, op_, c2w, K, (h, w), dev).clamp(0, 1)
                # Supersampled render of the SAME camera: doubling the
                # resolution means doubling fx, fy, cx, cy, which leaves the
                # normalized intrinsics render_map builds unchanged -- same
                # frustum, finer sampling.
                K2 = K.clone()
                K2[:2] *= args.ss
                rs = render_map(mw, cw, rgb_, op_, c2w, K2,
                                (h * args.ss, w * args.ss), dev).clamp(0, 1)
                rsd = F.avg_pool2d(rs, args.ss)
                # Flat = low gradient in the GROUND TRUTH, so the mask does not
                # depend on which arm is being scored.
                gx = (tgt[:, :, :, 1:] - tgt[:, :, :, :-1]).abs().mean(1, keepdim=True)
                gy = (tgt[:, :, 1:, :] - tgt[:, :, :-1, :]).abs().mean(1, keepdim=True)
                g = F.pad(gx, (0, 1, 0, 0)) + F.pad(gy, (0, 0, 0, 1))
                flat = g < 0.02
                a1 = render_map(mw, cw, ones, op_, c2w, K, (h, w), dev).clamp(0, 1)
                ass = render_map(mw, cw, ones, op_, c2w, K2,
                                 (h * args.ss, w * args.ss), dev).clamp(0, 1)
                covered = a1.mean(1, keepdim=True) > 0.1

                p1.append(psnr(r1, tgt))
                pss.append(psnr(rsd, tgt))
                pself.append(psnr(r1, rsd))
                lps.append(float(lp(r1 * 2 - 1, tgt * 2 - 1)))
                hps.append(highpass_std(rs, F.interpolate(
                    (flat & covered).float(), scale_factor=args.ss) > 0.5))
                alphas.append(float(a1.mean(1)[covered[:, 0]].mean()))
                hpa.append(highpass_std(ass, F.interpolate(
                    covered.float(), scale_factor=args.ss) > 0.5))
                if args.save_crops and vi == len(held) // 2:
                    import pathlib
                    from PIL import Image as PILImage
                    out = pathlib.Path(args.save_crops)
                    if not out.is_absolute():
                        out = pathlib.Path(REPO_ROOT) / out
                    out.mkdir(parents=True, exist_ok=True)
                    for nm, im in (("1x", r1), ("ss", rs), ("alpha", ass)):
                        arr = (im[0].permute(1, 2, 0).cpu().numpy() * 255)
                        PILImage.fromarray(arr.astype(np.uint8)).save(
                            out / f"tau{tau:g}_{nm}.png")

        def mean(x):
            return sum(x) / len(x)
        print(f"{tau:>5.2f} {model.n:>10,} {mean(p1):>8.4f} {mean(pss):>8.4f} "
              f"{mean(pself):>7.2f} {mean(lps):>7.4f} {mean(hps):>8.5f} "
              f"{mean(alphas):>7.4f} {mean(hpa):>9.5f}", flush=True)
        del model, mw, cw, parts
        torch.cuda.empty_cache()

    print("\npsnr_1x   native-rate render vs GT -- the metric everything else "
          "in the skill uses", flush=True)
    print(f"psnr_ss   {args.ss}x render average-pooled to native, vs GT", flush=True)
    print("self      psnr(1x, ss-pooled): no GT involved. Low = the map has "
          "structure between the native sample points.", flush=True)
    print("hp_flat   high-pass std of the supersampled render on FLAT ground-"
          "truth regions = the lattice amplitude", flush=True)
    print("alpha     mean accumulated alpha over covered pixels", flush=True)
    print("hp_alpha  high-pass std of the accumulated alpha: a lattice HERE "
          "means coverage, not opacity (Q7)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

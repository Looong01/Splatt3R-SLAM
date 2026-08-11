"""Route R: use a joint N-view model as an external per-cluster scale referee.

§17.45 measured that one joint forward reaches 1.68% adjacent-view scale
inconsistency where the production pairwise prediction sits at 5.35%, and
§17.47 showed averaging pairwise predictions cannot get there (rho = 0.35, a
floor at 3.06%). What is left is to use the joint model not as a backbone but as
a *reference*: run it once over the keyframes, read off what it thinks each
keyframe's depth scale should be relative to the others, and apply that scalar
to each cluster of an already-built map.

This is §17.16's per-cluster depth-scale correction, which measured +0.003 dB.
The difference is what drives it. That arm was driven by the photometric loss,
and §17.17 proved the loss is structurally blind here: at this trajectory's
0.057 m held-out baseline a 4.5% depth error moves a pixel by 0.66 px. The same
lever driven by a reference that scores 1.68% on the same yardstick is a
different experiment.

    s_k = median over valid pixels of ( z_vggt,k / z_map,k )
    s_k <- s_k / median_k(s_k)

The normalization is what makes this legitimate: VGGT's overall scale is an
arbitrary gauge, so only the RELATIVE corrections are used, and the map's global
scale -- which the trajectory sets and which §17.17 showed the evaluation cannot
see anyway -- is left untouched. All keyframes go into ONE forward, so there is
no window-to-window chaining and no submap alignment: this is the offline
ceiling of the idea, deliberately.

Writes a corrected blob. Evaluate it with the same instruments as the original:

    refine_local.py --iters 0     (psnr / lpips on held-out frames)
    diag_seam_step.py             (the spatial seam metric of 17.41)
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
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--chunk", type=int, default=0,
                    help="0 = all keyframes in one forward (the ceiling); "
                         ">0 = process in chunks of this many, which is what a "
                         "causal system could do and is strictly weaker")
    ap.add_argument("--max-correction", type=float, default=0.25,
                    help="refuse corrections beyond this fraction; a keyframe "
                         "whose referee ratio is wild is more likely a "
                         "reference failure than a map failure")
    ap.add_argument("--mode", choices=("slope", "ratio"), default="slope",
                    help="slope cancels the reference's global depth shift; "
                         "ratio does not and was measured harmful")
    ap.add_argument("--save-scales", default=None,
                    help="write the per-keyframe corrections as .npy so the "
                         "refiner can use them as a PRIOR rather than as an "
                         "edit -- 17.48 showed the edit form cannot work")
    ap.add_argument("--compensate-pose", action="store_true",
                    help="translate each keyframe along its optical axis so "
                         "the cluster mean surface stays put; tests whether "
                         "the trajectory had already absorbed the jitter")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    out_path = os.path.abspath(args.out)
    dev = args.device
    blob = torch.load(args.kfgauss, map_location="cpu")
    kfs = blob["keyframes"]
    n = len(kfs)
    print(f"{n} keyframes from {os.path.basename(args.kfgauss)}")

    # VGGT's own preprocessing: width 518, height to the nearest multiple of 14.
    # The map is 512x384; the referee only ever contributes a per-view median
    # ratio, so the two grids need not agree pixel for pixel -- but the depth
    # map is resampled to the map's grid before the ratio is taken, so the same
    # surface is compared with the same weighting.
    H, W = 392, 518

    imgs = []
    for kf in kfs:
        im = kf["img"].permute(1, 2, 0).numpy()          # 3,h,w -> h,w,3
        im = np.asarray(Image.fromarray(np.uint8(np.clip(im, 0, 1) * 255))
                        .resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(im).permute(2, 0, 1))
    ims = torch.stack(imgs)

    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()

    groups = ([list(range(n))] if args.chunk <= 0 else
              [list(range(i, min(i + args.chunk, n)))
               for i in range(0, n, args.chunk)])
    depth = np.zeros((n, H, W), dtype=np.float32)
    for g in groups:
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            out = model(ims[g].to(dev))
        depth[g] = out["depth"][0, ..., 0].float().cpu().numpy()
        del out
        torch.cuda.empty_cache()
    print(f"referee ran in {len(groups)} forward(s) of "
          f"{[len(g) for g in groups]} views")

    ratios = []
    for k, kf in enumerate(kfs):
        h, w = [int(v) for v in kf["img_shape"]]
        zmap = kf["means"][:, 2].numpy().reshape(h, w)
        zref = np.asarray(Image.fromarray(depth[k]).resize((w, h),
                                                           Image.BILINEAR))
        ok = (kf["conf"].numpy().reshape(h, w) > args.min_confidence)
        ok &= (zmap > 0.05) & (zref > 0.05) & np.isfinite(zref)
        if ok.sum() <= 2000:
            ratios.append(np.nan)
            continue
        a, b = zmap[ok], zref[ok]
        if args.mode == "ratio":
            ratios.append(float(np.median(b / a)))
        else:
            # SLOPE, not ratio. §17.45's probe measured that VGGT carries a
            # global shift of +29.9% of median depth: z_ref = A*z_true + B.
            # With z_map = c_k*z_true, the median ratio is
            #     median(z_ref/z_map) = (A + B*median(1/z_true)) / c_k
            # so it mixes the cluster's scale error with how far away that
            # view's content happens to be. The slope of z_ref against z_map
            # is A/c_k and the shift cancels exactly. One reweighting pass
            # against the tails, which are depth discontinuities where the two
            # grids disagree about which surface a pixel sees.
            p = np.polyfit(a, b, 1)
            res = np.abs(b - np.polyval(p, a))
            keep = res < 2.5 * np.median(res)
            if keep.sum() > 2000:
                p = np.polyfit(a[keep], b[keep], 1)
            ratios.append(float(p[0]))
    r = np.array(ratios)
    good = np.isfinite(r)
    if good.sum() < 3:
        print("referee produced too few usable keyframes")
        return 1
    # gauge-free: only the RELATIVE corrections are used
    s = r / np.median(r[good])
    s[~good] = 1.0
    clipped = int((np.abs(s - 1.0) > args.max_correction).sum())
    s = np.clip(s, 1 - args.max_correction, 1 + args.max_correction)

    print(f"\nper-cluster correction s_k")
    print(f"  spread   median |s-1| = {np.median(np.abs(s - 1)) * 100:.2f}%"
          f"   (this is what 17.45 measured as 5.35% pairwise)")
    print(f"  range    [{s.min():.4f}, {s.max():.4f}]   clipped {clipped}/{n}")
    print("  " + "  ".join(f"{v:.3f}" for v in s))

    if args.save_scales:
        np.save(os.path.abspath(args.save_scales), s)
        print(f"saved per-keyframe corrections -> {args.save_scales}")

    for k, kf in enumerate(kfs):
        kf["means"] = kf["means"] * float(s[k])
        kf["scales"] = kf["scales"] * float(s[k])

    if args.compensate_pose:
        # The hypothesis for why the uncompensated correction hurts: SLAM fitted
        # each keyframe's pose to that keyframe's own (biased) pointmap, so the
        # trajectory has already absorbed the cluster's MEAN depth error.
        # Rescaling the cluster unilaterally moves its mean surface in world
        # space, which is a relative displacement between overlapping clusters
        # -- the one thing the photometric evaluation is not blind to.
        #
        # Compensating puts the mean surface back and leaves only the
        # second-order part (the stretch of depth variation about the mean).
        # If that recovers the baseline, the referee's correction was already
        # accounted for by the trajectory and the map never carried it.
        import lietorch as _lt
        for k, kf in enumerate(kfs):
            d = kf["T_WC"].reshape(-1).clone()
            t, q, sc = d[:3], d[3:7], d[7]
            R = _lt.SE3(torch.cat([t, q]).unsqueeze(0)).matrix()[0, :3, :3]
            h, w = [int(v) for v in kf["img_shape"]]
            z = kf["means"][:, 2].numpy().reshape(h, w)
            dbar = float(np.median(z[np.isfinite(z) & (z > 0.05)]))
            # the mean surface sat at s_k*dbar after scaling; put it back
            delta = torch.tensor([0.0, 0.0, dbar * (1.0 - float(s[k]))])
            d[:3] = t + sc * (R @ delta)
            kf["T_WC"] = d.reshape(kf["T_WC"].shape)
        print("pose-compensated: each keyframe translated along its optical "
              "axis to hold the median surface fixed")

    torch.save(blob, out_path)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

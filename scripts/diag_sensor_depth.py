"""Upper bound on the seams: how much of them belongs to depth error at all?

This is a DIAGNOSTIC, not a proposed feature. The SLAM system is monocular and
this uses TUM's depth sensor, so nothing here is deployable. Its job is to
answer one question that no amount of parameter tuning can: **if every
Gaussian's depth were exactly right, how much of the veil would remain?**

That matters because §17.27 splits the defects into ones the map can fix
(disagreement between two measurements of one surface) and ones it cannot
(absent information). The seam was diagnosed morphologically as a misplaced
cluster (§17.4) and quantified as a −6% network depth bias with 9.6% per-keyframe
spread (§17.17), but the per-cluster depth scale bought +0.003 dB (§17.16) --
which proves nothing either way, because the metric is blind (§17.17).

Method: rescale every Gaussian along its own view ray so its depth matches the
sensor at that pixel, leaving direction, covariance, colour and opacity alone.
Then score both maps with the fly-through warp metric, which can see the veil.

Predicted by Kimi (round 7), pre-registered:
    per-keyframe spread   9.6%  ->  <= 2.5%
    warp deficit                    -40 .. -60%
    psnr                            +0.05 .. +0.25, lpips +-1%
Falsification: if the spread collapses but the warp deficit falls < 30%, the
seams are NOT depth-dominated and the next suspect is pose or alpha competition.
That verdict is worth having whichever way it goes.

Usage:
    python3 scripts/diag_sensor_depth.py \
        --kfgauss logs/frames_head/<seq>_kfgauss.pt \
        --traj logs/frames_head/<seq>.txt --dataset datasets/tum/<seq> \
        --out logs/sensor_depth
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

from eval_map_quality import associate, load_tum_traj


def sensor_depth_for(root, d_ts, d_paths, stamp, h, w):
    """TUM depth at the frame nearest `stamp`, resampled to the network's grid.

    resize_img scales the 640x480 long side to 512 and the subsequent centre
    crop is a no-op at 4:3, so this is a pure resize -- verified in §17.17
    before that section's ratio was trusted.
    """
    j = int(np.argmin(np.abs(d_ts - stamp)))
    if abs(d_ts[j] - stamp) > 0.02:
        return None
    d = np.asarray(Image.open(os.path.join(root, d_paths[j])),
                   dtype=np.float32) / 5000.0        # TUM png_depth_scale
    return np.asarray(Image.fromarray(d).resize((w, h), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="logs/sensor_depth")
    ap.add_argument("--mode", choices=("per-pixel", "per-keyframe", "curve"),
                    default="curve",
                    help="per-pixel REPLACES each surface's shape with the "
                         "sensor's, including its noise and its RGB-depth "
                         "registration offset -- measured at -1.29 dB, which is "
                         "evidence about the sensor, not about seams, and makes "
                         "that arm unable to answer the question it was built "
                         "for. per-keyframe applies ONE scalar per cluster, "
                         "which is the hypothesis actually under test (clusters "
                         "sit at the wrong depth relative to each other) with no "
                         "sensor shape noise injected -- but that arm is ALSO "
                         "invalid, because the per-keyframe ratio correlates "
                         "+0.736 with what the keyframe is looking at and 54%% "
                         "of its variance is explained by scene depth alone. "
                         "The spread is mostly a range-dependent bias, not "
                         "per-cluster randomness, so a per-keyframe scalar "
                         "applies a scene-content-driven scale and scrambles "
                         "the clusters (-0.70 dB). `curve` is the arm that "
                         "survives all three objections: one global d -> d\' "
                         "map fitted across every keyframe, applied per pixel, "
                         "correcting the systematic range-dependent bias with "
                         "no per-keyframe noise and no sensor shape.")
    ap.add_argument("--global-scale", action="store_true",
                    help="apply the FULL sensor/prediction ratio, including its "
                         "global component. Off by default and it should stay "
                         "off: the trajectory was estimated from the same "
                         "pointmaps, so poses and predicted depths share a "
                         "scale and are mutually consistent. Scaling depth "
                         "alone breaks that consistency and displaces every "
                         "cluster -- measured at -1.26 dB, which swamps the "
                         "effect being looked for. Seams are a RELATIVE "
                         "misplacement between clusters, so the per-keyframe "
                         "deviation from the median is the whole signal, and "
                         "evaluation's umeyama alignment absorbs the global "
                         "part anyway.")
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--aa-sigma", type=float, default=0.5)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    from splatt3r_slam.config import load_config
    from splatt3r_slam.refiner import (LocalGaussianMap, sim3_to_mat,
                                       gaussians_from_keyframe,
                                       save_refined_map)
    load_config(args.config)
    dev = args.device
    root = args.dataset

    d_ts, d_paths = [], []
    for line in open(os.path.join(root, "depth.txt")):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) == 2:
            d_ts.append(float(p[0])); d_paths.append(p[1])
    d_ts = np.array(d_ts)
    est = np.loadtxt(args.traj)
    blob = torch.load(args.kfgauss, map_location="cpu", weights_only=False)
    os.makedirs(args.out, exist_ok=True)

    # The global component, needed before the per-keyframe pass so the
    # relative-only arm can divide it out.
    GLOBAL_MED = 1.0
    pre = []
    for k, kf in enumerate(blob["keyframes"]):
        if k >= len(est):
            break
        h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
        d = sensor_depth_for(root, d_ts, d_paths, est[k, 0], h, w)
        if d is None:
            continue
        z = kf["means"].view(h, w, 3)[..., 2].float().numpy()
        ok = (d > 0.3) & (d < 6.0) & (z > 0.05)
        if ok.sum() > 1000:
            pre.append(float(np.median(d[ok] / z[ok])))
    if pre:
        GLOBAL_MED = float(np.median(pre))
    print(f"global sensor/pred depth ratio (divided out unless "
          f"--global-scale): {GLOBAL_MED:.4f}", flush=True)

    # Pooled affine fit d_sensor ~ a * d_pred + b over every keyframe, which is
    # the correction that survives all three failed arms: it is global (no
    # per-keyframe scene-content scale), low-order (no sensor shape noise), and
    # it targets the range-dependent bias that 54% of the per-keyframe spread
    # turned out to be.
    CURVE_A, CURVE_B = 1.0, 0.0
    xs, ys = [], []
    for k, kf in enumerate(blob["keyframes"]):
        if k >= len(est):
            break
        h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
        d = sensor_depth_for(root, d_ts, d_paths, est[k, 0], h, w)
        if d is None:
            continue
        z = kf["means"].view(h, w, 3)[..., 2].float().numpy()
        ok = (d > 0.3) & (d < 6.0) & (z > 0.05)
        if ok.sum() > 1000:
            xs.append(z[ok][::37]); ys.append(d[ok][::37])
    if xs:
        X = np.concatenate(xs); Y = np.concatenate(ys)
        A = np.stack([X, np.ones_like(X)], 1)
        (CURVE_A, CURVE_B), *_ = np.linalg.lstsq(A, Y, rcond=None)
        res = Y - (CURVE_A * X + CURVE_B)
        print(f"pooled depth curve: d_sensor = {CURVE_A:.4f} * d_pred "
              f"{CURVE_B:+.4f}   (n={len(X):,}, residual std {res.std():.4f} m, "
              f"R^2 {1 - res.var() / Y.var():.3f})", flush=True)

    for use_sensor in (False, True):
        parts, poses, ratios = [], [], []
        for k, kf in enumerate(blob["keyframes"]):
            h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
            local = {x: kf[x].to(dev).clone() for x in
                     ("means", "scales", "rotations", "sh", "opacities", "conf")}
            if use_sensor and k < len(est):
                d = sensor_depth_for(root, d_ts, d_paths, est[k, 0], h, w)
                if d is not None:
                    m = local["means"].view(h, w, 3)
                    z = m[..., 2]
                    dt = torch.as_tensor(d, device=dev)
                    # Valid sensor returns only; elsewhere keep the prediction.
                    # Rescale ALONG THE RAY: direction, covariance, colour and
                    # opacity are the network's and are left untouched, so the
                    # only thing this arm changes is depth.
                    ok = (dt > 0.3) & (dt < 6.0) & (z > 0.05)
                    r = torch.where(ok, dt / z.clamp_min(1e-6),
                                    torch.ones_like(z))
                    med = float(r[ok].median()) if bool(ok.any()) else 1.0
                    ratios.append(med)
                    if args.mode == "per-keyframe":
                        r = torch.full_like(z, med)
                    elif args.mode == "curve":
                        r = (CURVE_A * z + CURVE_B).clamp(0.5, 2.0) / z.clamp_min(1e-6)
                    if not args.global_scale and args.mode != "curve":
                        r = r / GLOBAL_MED
                    local["means"] = (m * r[..., None]).reshape(-1, 3)
            got = gaussians_from_keyframe(
                local, kf["img"].to(dev).clone(), h, w, k, dev,
                min_confidence=args.min_confidence, aa_sigma_scale=args.aa_sigma)
            if got is None:
                continue
            parts.append(tuple(t.clone() for t in got))
            poses.append(kf["T_WC"].to(dev))
        mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in poses]))
        model = LocalGaussianMap(*[torch.cat([p[i] for p in parts])
                                   for i in range(6)]).to(dev)
        tag = "sensor" if use_sensor else "pred"
        path = os.path.join(args.out, f"{tag}.ply")
        save_refined_map(path, model, mats)
        if ratios:
            r = np.array(ratios)
            print(f"  per-keyframe sensor/pred depth ratio: median "
                  f"{np.median(r):.4f}  std {r.std():.4f}  "
                  f"spread {r.max() - r.min():.4f}", flush=True)
        print(f"{tag}: {model.n:,} gaussians -> {path}", flush=True)
        del model, parts
        torch.cuda.empty_cache()

    print("\nNow score both with scripts/diag_flythrough.py and "
          "scripts/eval_map_quality.py. The pre-registered prediction is in this "
          "file's docstring; record whichever way it falls.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Does the streak lever help WHERE it fades? The within-sequence test.

The lever is now a default (`--refiner-streak-opacity 0.5`), justified by a
3-sequence paired A/B: lpips improves 12.7% / 6.6% / 1.3% on desk / room / 360.
The story attached to that spread is scene content -- 360 spins in place and has
few depth-discontinuity edges, desk is full of object boundaries.

That story is a cross-sequence rank agreement over three points, which agrees by
chance one time in three. Kimi's round-19 design, adopted: the test with real
power is **spatial and within one sequence**. If the lever works by hiding
ray-elongated Gaussians at depth discontinuities, then the improvement must
concentrate on the pixels those Gaussians paint. If the improvement is instead
spread evenly over the frame, the lever is doing something else (global haze
suppression) and the edge story is wrong.

Method, all from one baked map so nothing else differs:

  1. Build the map twice from the SAME keyframe blob, once with the fade off and
     once with it on. The per-Gaussian fade factor is then just the ratio of the
     two opacity vectors -- no need to recompute the criterion.
  2. Render a **fade-deficit image**: (1 - fade) as the colour channel, with the
     UNFADED opacities as the weights, so the image answers "how much opacity
     was removed from the Gaussians that paint this pixel".
  3. Score both renders against the real frame with spatial LPIPS (AlexNet,
     spatial=True gives a per-pixel map rather than one number).
  4. Bin pixels by fade deficit and report the LPIPS improvement per bin.

Prediction if the mechanism is right: improvement rises monotonically with fade
deficit, and is ~0 in the zero-fade bin.
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


def build(blob, dev, min_conf, aa_sigma, streak):
    from splatt3r_slam.refiner import gaussians_from_keyframe
    parts, poses = [], []
    for k, kf in enumerate(blob["keyframes"]):
        h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
        local = {x: kf[x].to(dev) for x in
                 ("means", "scales", "rotations", "sh", "opacities", "conf")}
        got = gaussians_from_keyframe(local, kf["img"].to(dev), h, w, k, dev,
                                      min_confidence=min_conf,
                                      aa_sigma_scale=aa_sigma,
                                      streak_opacity=streak)
        if got is None:
            continue
        parts.append(got)
        poses.append(kf["T_WC"].to(dev))
    return parts, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-held", type=int, default=25)
    ap.add_argument("--streak", type=float, default=0.5)
    ap.add_argument("--aa-sigma", type=float, default=0.5)
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--dump", default=None,
                    help="write GT / off / on / fade-deficit PNGs here. The "
                         "project rule is that no verdict is believed on "
                         "scalars alone; the bin table below is scalars.")
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.image import (normalize_exposure as _norm,
                                     reset_exposure_reference)
    from splatt3r_slam.refiner import LocalGaussianMap, sim3_to_mat, render_map
    import lpips as lpips_lib

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
    held = cand[:: max(1, len(cand) // args.n_held)][: args.n_held]

    # same colour space as the map (17.50)
    reset_exposure_reference()
    _norm(ds.get_image(0))

    lp = lpips_lib.LPIPS(net="alex", spatial=True).to(dev)
    K = torch.as_tensor(ds.camera_intrinsics.K_frame, dtype=torch.float32,
                        device=dev)

    blob = torch.load(args.kfgauss, map_location="cpu")
    os.chdir(CORE)
    p_off, poses = build(blob, dev, args.min_confidence, args.aa_sigma, 0.0)
    p_on, _ = build(blob, dev, args.min_confidence, args.aa_sigma, args.streak)
    kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in poses]))
    m_off = LocalGaussianMap(*[torch.cat([p[i] for p in p_off])
                               for i in range(6)]).to(dev)
    m_on = LocalGaussianMap(*[torch.cat([p[i] for p in p_on])
                              for i in range(6)]).to(dev)

    with torch.no_grad():
        o_off, o_on = m_off.opacity(), m_on.opacity()
        fade = (o_on / o_off.clamp_min(1e-8)).clamp(0, 1)
        deficit = (1.0 - fade).reshape(-1, 1).expand(-1, 3).contiguous()
        print(f"per-Gaussian fade: {float((fade < 0.999).float().mean())*100:.2f}%"
              f" of {fade.numel():,} Gaussians touched, "
              f"mean deficit on those {float((1-fade)[fade<0.999].mean()):.3f}")

        mw, cw = m_off.world(kf_mats)
        mw2, cw2 = m_on.world(kf_mats)
        edges = [0.0, 0.02, 0.05, 0.10, 0.20, 1.01]
        dumped = []
        num = np.zeros(len(edges) - 1)
        den = np.zeros(len(edges) - 1)
        for di, gj in held:
            raw = _norm(ds.get_image(di))
            img = resize_img(raw, ds.img_size)["img"]
            tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * .5 + .5
            h, w = tgt.shape[-2:]
            c2w = np.eye(4)
            c2w[:3, :3] = Rt @ gt_T[gj][:3, :3]
            c2w[:3, 3] = Rt @ (gt_T[gj][:3, 3] - t_) / s_
            r_off = render_map(mw, cw, m_off.rgb(), o_off, c2w, K, (h, w),
                               dev).clamp(0, 1)
            r_on = render_map(mw2, cw2, m_on.rgb(), o_on, c2w, K, (h, w),
                              dev).clamp(0, 1)
            # how much opacity the lever removed from what paints this pixel
            fd = render_map(mw, cw, deficit, o_off, c2w, K, (h, w),
                            dev).clamp(0, 1).mean(1)[0]
            l_off = lp(r_off, tgt, normalize=True)[0, 0]
            l_on = lp(r_on, tgt, normalize=True)[0, 0]
            gain = (l_off - l_on)          # positive = the lever helped here
            fd_r = F.interpolate(fd[None, None], size=gain.shape,
                                 mode="bilinear", align_corners=False)[0, 0]
            if args.dump and len(dumped) < 4:
                import cv2 as _cv
                os.makedirs(os.path.join(REPO_ROOT, args.dump), exist_ok=True)
                i = len(dumped)
                def _w(nm, t):
                    a = (t.detach().float().clamp(0, 1).cpu().numpy()
                         .transpose(1, 2, 0) * 255).round().astype(np.uint8)
                    _cv.imwrite(os.path.join(REPO_ROOT, args.dump,
                                             f"v{i}_{nm}.png"),
                                _cv.cvtColor(a, _cv.COLOR_RGB2BGR))
                _w("gt", tgt[0]); _w("off", r_off[0]); _w("on", r_on[0])
                # fade deficit as a heat image, and the signed lpips gain
                _w("fade", fd[None].expand(3, -1, -1))
                g = gain.clamp(-0.05, 0.05)
                g = torch.stack([(g.clamp_min(0) / 0.05),
                                 torch.zeros_like(g),
                                 ((-g).clamp_min(0) / 0.05)])
                _w("gain", F.interpolate(g[None], size=fd.shape,
                                         mode="nearest")[0])
                dumped.append(i)
            for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                m = (fd_r >= lo) & (fd_r < hi)
                if bool(m.any()):
                    num[b] += float(gain[m].sum())
                    den[b] += float(m.sum())

    print(f"\n{'fade deficit bin':>18} {'pixels':>10} {'mean lpips gain':>16}")
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if den[b] == 0:
            continue
        print(f"{lo:7.2f}-{hi:<8.2f} {den[b]/den.sum()*100:9.1f}% "
              f"{num[b]/den[b]:+16.5f}")
    print("\nmechanism predicts the gain rises with the deficit and is ~0 in the "
          "first bin.\nA flat profile means the lever is not working through the "
          "Gaussians it fades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

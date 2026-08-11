"""What is the per-view affine actually absorbing?

`--exposure` buys +0.45 dB fit psnr and -2.6% lpips offline (17.6), and the
learned per-frame gains have a mean of 1.04 with 11-12% std -- an order of
magnitude more than the 4.1% spread measured on the source images themselves
(17.4). So it is absorbing something real and it is NOT global exposure drift.

Two candidate explanations, and they make opposite predictions here:

  (a) COVERAGE DEFICIT. The render is darker than the target wherever alpha
      does not accumulate to 1, and how much darker depends on how much of THAT
      view the map covers. Then gain ~ 1/mean_alpha, the gains correlate with
      per-view coverage, and the exposure parameter is the photometric shadow of
      a geometric defect -- which would put the colour leg under the same root
      as the seams and the streaks (skill 17.34's low-parallax hypothesis).
  (b) GENUINE APPEARANCE VARIATION -- auto white balance, view-dependent
      illumination, the network's per-pair conditional bias. Then the gains are
      uncorrelated with coverage and the colour leg is an independent problem.

Method: freeze the entire map and optimize ONLY the per-frame affine. With the
map fixed, the fitted gain per frame IS that frame's best global affine, which
is exactly the quantity to explain. Then correlate it against that frame's mean
accumulated alpha and black fraction, both rendered from the same frozen map.

Zero new training of the map; a few hundred steps on 300 parameters.
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

from eval_map_quality import associate, load_tum_traj, umeyama_sim3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames-traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--aa-sigma", type=float, default=0.5)
    ap.add_argument("--alpha-min", type=float, default=0.99,
                    help="fit only where accumulated alpha exceeds this; 0 "
                         "reproduces the unmasked fit whose 0.93 gain may be "
                         "entirely black-pixel contamination")
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.refiner import (LocalGaussianMap, sim3_to_mat,
                                       gaussians_from_keyframe, render_map)
    from refine_gaussian_map import uniform_subsample

    load_config(args.config)
    os.chdir(CORE)
    dev = args.device
    ds = load_dataset(os.path.join(REPO_ROOT, args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    frm_ts, frm_T = load_tum_traj(os.path.join(REPO_ROOT, args.frames_traj))
    sel = uniform_subsample(associate(frm_ts, ds_ts), args.n_train)
    frames = []
    for fi, di in sel:
        img = resize_img(ds.get_image(di), ds.img_size)["img"]
        tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
        frames.append((frm_T[fi], tgt))

    blob = torch.load(os.path.join(REPO_ROOT, args.kfgauss), map_location="cpu")
    parts, poses = [], []
    for k, kf in enumerate(blob["keyframes"]):
        h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
        local = {x: kf[x].to(dev) for x in
                 ("means", "scales", "rotations", "sh", "opacities", "conf")}
        got = gaussians_from_keyframe(local, kf["img"].to(dev), h, w, k, dev,
                                      min_confidence=args.min_confidence,
                                      aa_sigma_scale=args.aa_sigma)
        if got is None:
            continue
        parts.append(got)
        poses.append(kf["T_WC"].to(dev))
    kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in poses]))
    model = LocalGaussianMap(*[torch.cat([p[i] for p in parts])
                               for i in range(6)]).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    K = torch.as_tensor(ds.camera_intrinsics.K_frame, dtype=torch.float32, device=dev)
    print(f"map {model.n:,} gaussians frozen; fitting {len(frames)} x 6 "
          f"exposure parameters only", flush=True)

    with torch.no_grad():
        mw, cw = model.world(kf_mats)
        rgb_, op_ = model.rgb(), model.opacity()
        ones = torch.ones_like(rgb_)
        preds, alphas, blacks = [], [], []
        for c2w, tgt in frames:
            h, w = tgt.shape[-2:]
            preds.append(render_map(mw, cw, rgb_, op_, c2w, K, (h, w), dev).clamp(0, 1))
            a = render_map(mw, cw, ones, op_, c2w, K, (h, w), dev).clamp(0, 1).mean(1)
            alphas.append(float(a.mean()))
            blacks.append(float((a < 0.1).float().mean()))

    # MASK to well-covered pixels. Fitting over the whole frame lets the
    # unmapped black regions (4-5% here) pull the gain down: they are dark in
    # the render and not in the target, so the fit trades real exposure for
    # covering them. Split-half cannot catch this -- the contamination is
    # present and identical in both halves, so a stably-wrong estimate still
    # earns 11 sigma. Kimi's gate, and it is the difference between measuring
    # the map and measuring the instrument.
    masks = []
    with torch.no_grad():
        for c2w, tgt in frames:
            h, w = tgt.shape[-2:]
            a = render_map(mw, cw, ones, op_, c2w, K, (h, w), dev).clamp(0, 1).mean(
                1, keepdim=True)
            masks.append((a > args.alpha_min).float())
    print(f"coverage mask keeps {float(torch.stack([m.mean() for m in masks]).mean()):.3f} "
          f"of pixels at alpha > {args.alpha_min}", flush=True)

    expo = torch.zeros((len(frames), 2, 3), device=dev, requires_grad=True)
    opt = torch.optim.Adam([expo], lr=args.lr)
    for it in range(args.iters):
        i = it % len(frames)
        p = preds[i] * expo[i, 0].exp().view(1, 3, 1, 1) + expo[i, 1].view(1, 3, 1, 1)
        m = masks[i]
        loss = ((p - frames[i][1]).abs() * m).sum() / m.sum().clamp_min(1) / 3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        gch = expo[:, 0].exp().cpu().numpy()          # (F,3) per-channel
        gain = gch.mean(1)
    a = np.array(alphas)
    b = np.array(blacks)
    from scipy.stats import spearmanr, pearsonr
    print(f"\nlearned gain: mean {gain.mean():.4f}  std {gain.std():.4f}  "
          f"range [{gain.min():.4f}, {gain.max():.4f}]")
    print(f"per-view mean alpha: {a.mean():.4f} +- {a.std():.4f}")
    print(f"per-view black frac: {b.mean():.4f} +- {b.std():.4f}")
    print(f"\n  corr(gain, 1/mean_alpha)  spearman {spearmanr(gain, 1/a).statistic:+.3f}"
          f"  pearson {pearsonr(gain, 1/a).statistic:+.3f}")
    print(f"  corr(gain, black frac)    spearman {spearmanr(gain, b).statistic:+.3f}")
    # If (a) holds, gain should not merely correlate with 1/alpha -- it should be
    # NUMERICALLY close to it, since that is the exact factor a coverage deficit
    # costs. Reporting the ratio makes the difference between "related" and
    # "explained" visible.
    print(f"\n  mean(gain * mean_alpha) = {float((gain * a).mean()):.4f} "
          f"(1.000 would mean coverage explains the gain exactly)")
    print(f"  residual std after dividing gain by 1/alpha: "
          f"{float((gain * a).std()):.4f}  vs raw gain std {gain.std():.4f}")

    # ---- three discriminators for what the affine IS absorbing (17.35) ----
    # 1. Channel decoupling. White balance moves the channel RATIOS; anything
    #    achromatic moves the channels in lock-step. Comparing the spread of the
    #    ratios against the spread of the luminance separates them.
    r_rg = gch[:, 0] / gch[:, 1]
    r_bg = gch[:, 2] / gch[:, 1]
    print(f"\n1. channel decoupling (white balance)")
    print(f"   luminance gain std {gain.std():.4f}   "
          f"R/G std {r_rg.std():.4f}   B/G std {r_bg.std():.4f}")
    print(f"   chroma/luma spread ratio "
          f"{max(r_rg.std(), r_bg.std()) / max(gain.std(), 1e-9):.2f}   "
          f"(>1 favours white balance, <<1 rules it out)")

    # 2. Per-cluster ANOVA. If the network carries a per-pair conditional bias,
    #    frames that see the same cluster share it, so between-group variance
    #    dominates within-group.
    dom = []
    with torch.no_grad():
        ones = torch.ones_like(rgb_)
        n_kf = int(model.kf_id.max()) + 1
        for c2w, tgt in frames:
            h, w = tgt.shape[-2:]
            best, bi = -1.0, -1
            for k in range(n_kf):
                m = model.kf_id == k
                if not bool(m.any()):
                    continue
                al = render_map(mw[m], cw[m], ones[m], op_[m], c2w, K, (h, w),
                                dev).clamp(0, 1).mean().item()
                if al > best:
                    best, bi = al, k
            dom.append(bi)
    dom = np.array(dom)
    grand = gain.mean()
    ssb = sum(((gain[dom == k].mean() - grand) ** 2) * (dom == k).sum()
              for k in np.unique(dom) if (dom == k).sum() > 0)
    ssw = sum(((gain[dom == k] - gain[dom == k].mean()) ** 2).sum()
              for k in np.unique(dom) if (dom == k).sum() > 1)
    k_g = len(np.unique(dom))
    msb = ssb / max(k_g - 1, 1)
    msw = ssw / max(len(gain) - k_g, 1)
    print(f"\n2. per-cluster ANOVA (network per-pair conditional bias)")
    print(f"   {k_g} dominant clusters over {len(gain)} frames   "
          f"between/within = {msb / max(msw, 1e-12):.2f}   "
          f"(>>1 confirms a per-cluster bias)")

    # 3. Temporal smoothness. Camera-side drift is smooth in time; a view- or
    #    map-side bias jumps. Compared against a shuffled null so the statistic
    #    is scale-free.
    step = np.abs(np.diff(gain)).mean()
    rngp = np.random.default_rng(0)
    null = np.mean([np.abs(np.diff(rngp.permutation(gain))).mean()
                    for _ in range(200)])
    print(f"\n3. temporal smoothness (camera-side vs view/map-side)")
    print(f"   mean |g(f+1)-g(f)| {step:.4f}   shuffled null {null:.4f}   "
          f"ratio {step / max(null, 1e-9):.3f}")
    print(f"   (<<1 smooth = camera side; ~1 = jumps per view, map/network side)")

    # 4. NOISE FLOOR, first, because 1-3 are unreadable without it (Kimi r14).
    #    Re-fit the affine on the top and bottom halves of each frame
    #    independently; the disagreement between two halves of the same frame is
    #    fit noise by construction, since the frame's true exposure is one
    #    number. Anything below this floor is not evidence.
    halves = []
    for half in (0, 1):
        e2 = torch.zeros((len(frames), 2, 3), device=dev, requires_grad=True)
        o2 = torch.optim.Adam([e2], lr=args.lr)
        for it in range(args.iters):
            i = it % len(frames)
            hh = preds[i].shape[-2] // 2
            sl = slice(0, hh) if half == 0 else slice(hh, None)
            pr = preds[i][:, :, sl] * e2[i, 0].exp().view(1, 3, 1, 1) + e2[i, 1].view(1, 3, 1, 1)
            mm = masks[i][:, :, sl]
            loss = ((pr - frames[i][1][:, :, sl]).abs() * mm).sum() / mm.sum().clamp_min(1) / 3
            o2.zero_grad(set_to_none=True); loss.backward(); o2.step()
        with torch.no_grad():
            halves.append(e2[:, 0].exp().cpu().numpy())
    floor = np.abs(halves[0] - halves[1]) / np.sqrt(2)
    fr_rg = np.std(halves[0][:, 0] / halves[0][:, 1] - halves[1][:, 0] / halves[1][:, 1]) / np.sqrt(2)
    fr_bg = np.std(halves[0][:, 2] / halves[0][:, 1] - halves[1][:, 2] / halves[1][:, 1]) / np.sqrt(2)
    print(f"\n4. per-channel NOISE FLOOR (split-half refit)")
    print(f"   R/G  observed std {r_rg.std():.4f}  floor {fr_rg:.4f}  "
          f"SNR {r_rg.std()/max(fr_rg,1e-9):.2f}")
    print(f"   B/G  observed std {r_bg.std():.4f}  floor {fr_bg:.4f}  "
          f"SNR {r_bg.std()/max(fr_bg,1e-9):.2f}")
    # The luminance gain's own floor -- the number that decides whether ANY
    # per-view appearance variation exists, or whether the only real effect is
    # the global offset (mean 0.93) with fit noise around it.
    lum_h = [h.mean(1) for h in halves]
    fl_lum = float(np.std(lum_h[0] - lum_h[1]) / np.sqrt(2))
    print(f"   LUMA observed std {gain.std():.4f}  floor {fl_lum:.4f}  "
          f"SNR {gain.std()/max(fl_lum,1e-9):.2f}")
    print(f"   (SNR ~1 means the channel carries no real variation and its "
          f"spread is fit noise)")
    print(f"   mean gain {gain.mean():.4f} vs 1.0: the global offset, which is "
          f"a separate question from the per-frame spread")

    # 5. THE WB AXIS TEST (Kimi r14, the sharpest one). Real white-balance drift
    #    is ONE-DIMENSIONAL along colour temperature: warmer pushes R/G up and
    #    B/G down together. Independent per-channel jitter has no such structure.
    from scipy.stats import pearsonr as _pr
    c_wb = _pr(r_rg, r_bg).statistic
    print(f"\n5. white-balance axis test")
    print(f"   corr(R/G, B/G) = {c_wb:+.3f}   "
          f"(clearly negative = colour-temperature axis = WB; "
          f"~0 = independent jitter = noise or content)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

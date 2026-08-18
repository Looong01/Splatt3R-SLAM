"""Do low opacity and large scale sit where the supervision is unreliable?

Kimi's round-28 decisive test, and it settles two open questions at once with a
single forward pass and no training.

The question behind it. Fine-tuning on any REAL family un-saturates opacity
(TUM median 0.66, eth3d 0.24) while Replica stays at 1.00, and no image-space
degradation reproduces the effect (17.65.1). Two candidate accounts survive:

  confidence channel  opacity is the head's "discount this Gaussian, it may be
                      wrong" output. Predicts low opacity CO-LOCATES with
                      unreliable supervision -- invalid depth, occlusion
                      boundaries, inter-view photometric disagreement.
  global re-gauge     un-saturation is a uniform shift with no spatial
                      structure, and the co-location is absent.

These differ in a directly measurable way, so this is not an argument that has
to be settled by mechanism-talk. The same design tests the other open reading --
that SCALE, not opacity, is the head's uncertainty channel (17.65.1) -- because
scale gets the identical treatment against the identical predictors.

Kimi's objection to the alternative test is why this one exists: "cap hurts the
noisy head more than the plain head" cannot confirm scale-as-hedge, because the
cap's damage runs through coverage and the noisy head simply has a bigger tail
to bite. Differential harm is predicted by both accounts. Co-location is not.

Three predictors, all per-pixel, all computed on the SLAM side so nothing here
shares the training preprocessing (17.55 / standing rule 1):

  invalid     depth <= 0 in the sensor depth, dilated. The mask candidate.
  edge        distance to a depth discontinuity (occlusion boundary).
  disagree    |I1 - I2 warped| proxy: local photometric disagreement between
              the pair, which is the non-rigid-photometry candidate.

Reported as the mean predictor value in each opacity/scale decile, plus a rank
correlation. A confidence channel shows a monotone gradient across deciles; a
global re-gauge shows a flat profile.

    python3 scripts/diag_colocation.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
        --head checkpoints/head_only_long/tum/head_best.pt --pairs 6
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO, "splatt3r_core")
sys.path.insert(0, REPO)
sys.path.insert(0, CORE)


def spearman(a, b):
    """Rank correlation without pulling in scipy."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def decile_profile(values, predictors, names):
    """Mean of each predictor within each decile of `values`."""
    order = np.argsort(values)
    chunks = np.array_split(order, 10)
    rows = []
    for d, idx in enumerate(chunks):
        rows.append([float(values[idx].mean())] + [float(p[idx].mean()) for p in predictors])
    return rows


def partial_spearman(a, b, c, nbins=8):
    """Rank correlation of a vs b within bins of c, pooled.

    Without this the headline correlation is not interpretable. TUM's
    structured-light depth drops out on FAR, glossy and dark surfaces, and
    Gaussian scale grows with distance by construction -- so `invalid` is a
    candidate proxy for range, and a raw correlation between opacity and
    `invalid` could be entirely a distance effect. Stratifying by the network's
    own predicted depth removes that path: whatever survives inside a depth
    band is not explained by depth.
    """
    edges = np.quantile(c, np.linspace(0, 1, nbins + 1))
    vals, wts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & (c <= hi)
        if m.sum() < 200 or np.ptp(b[m]) == 0:
            continue
        vals.append(spearman(a[m], b[m]))
        wts.append(m.sum())
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=wts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--head", default="")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--control", default="depth", choices=["depth", "conf", "none"],
                    help="which predictor to stratify the partial row on. "
                         "`depth` is the range confound. `conf` answers a "
                         "different question: `invalid` and `conf` are "
                         "plausibly the same signal twice, since the backbone "
                         "is unconfident where the sensor also failed and for "
                         "shared physical reasons (dark, glossy, far). If "
                         "invalid|conf collapses, there is one channel, not two.")
    args = ap.parse_args()

    os.chdir(REPO)
    from splatt3r_slam.config import load_config
    load_config(args.config)
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import load_splatt3r, splatt3r_asymmetric_inference, resize_img
    from splatt3r_slam.frame import create_frame
    import lietorch
    import cv2

    ds = load_dataset(os.path.join(REPO, args.dataset))
    n = len(ds)
    dev = "cuda"
    model = load_splatt3r(device=dev)
    if args.head:
        model.encoder.load_state_dict(
            torch.load(os.path.join(REPO, args.head), map_location=dev), strict=False)

    # The SLAM dataloader is monocular and exposes no depth, so the sensor
    # depth is read from the release layout directly -- same reader as
    # diag_nview_scale.py rather than a second one that could drift from it.
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from diag_nview_scale import load_depth_index
    root = os.path.join(REPO, args.dataset)
    try:
        _dts, dpaths, dscale = load_depth_index(root)
        has_depth = len(dpaths) > 0
    except Exception as e:
        print(f"  no sensor depth ({e}); invalid/edge predictors disabled")
        dpaths, dscale, has_depth = [], 1.0, False

    def sensor_depth(frame_idx):
        """Nearest depth frame for a colour index, in metres."""
        if not has_depth:
            return None
        k = min(int(round(frame_idx * len(dpaths) / max(n, 1))), len(dpaths) - 1)
        raw = cv2.imread(os.path.join(root, dpaths[k]), cv2.IMREAD_UNCHANGED)
        return None if raw is None else raw.astype(np.float32) / dscale

    idxs = [int(n * f) for f in np.linspace(0.05, 0.9, args.pairs)]

    op_all, sc_all = [], []
    inv_all, edge_all, dis_all, conf_all, z_all = [], [], [], [], []

    I = lietorch.Sim3(torch.tensor([0, 0, 0, 0, 0, 0, 1, 1.0],
                                   dtype=torch.float32, device=dev))
    for i in idxs:
        j = min(i + args.step, n - 1)
        im1, im2 = ds.get_image(i), ds.get_image(j)
        f1 = create_frame(i, im1, I, img_size=ds.img_size, device=dev)
        f2 = create_frame(j, im2, I, img_size=ds.img_size, device=dev)
        os.chdir(CORE)
        with torch.no_grad():
            _, _, _, _, res = splatt3r_asymmetric_inference(model, f1, f2)
        os.chdir(REPO)
        r = res[0]
        op = r["opacities"][0].float().reshape(-1).cpu().numpy()
        sc = r["scales"][0].float().reshape(-1, 3).cpu().numpy().max(1)
        # depth from the network's own pointmap, and the backbone confidence --
        # both free from the same forward pass. z is the confound control;
        # conf is Kimi's per-pixel unreliability predictor for the scale test.
        z = r["pts3d"][0].float().reshape(-1, 3).cpu().numpy()[:, 2]
        cf = r["conf"][0].float().reshape(-1).cpu().numpy()

        # the Gaussians are one-per-pixel of view 1, in row-major order, so the
        # per-pixel predictors index them directly
        h, w = resize_img(im1, ds.img_size)["img"].shape[-2:]
        npix = h * w
        if op.size != npix:
            print(f"  pair {i}: {op.size} gaussians for {npix} pixels -- not "
                  f"one-per-pixel, skipping")
            continue

        # --- predictor 1: invalid sensor depth (the mask candidate)
        d = sensor_depth(i)
        if d is not None:
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
            invalid = (d <= 0).astype(np.float32)
            # dilate: the candidate is that suppression bleeds beyond the mask
            inv = cv2.dilate(invalid, np.ones((9, 9), np.uint8)).reshape(-1)
            # --- predictor 2: occlusion boundary = depth discontinuity
            dd = d.copy()
            dd[dd <= 0] = np.nan
            gx = np.abs(np.gradient(dd, axis=1))
            gy = np.abs(np.gradient(dd, axis=0))
            g = np.nan_to_num(np.maximum(gx, gy), nan=0.0)
            edge = (g > 0.05).astype(np.float32)          # 5 cm jump
            edge = cv2.dilate(edge, np.ones((5, 5), np.uint8)).reshape(-1)
        else:
            inv = np.zeros(npix, np.float32)
            edge = np.zeros(npix, np.float32)

        # --- predictor 3: inter-view photometric disagreement.
        # No warp is available without a pose solve, so this is the honest
        # proxy: local contrast difference between the pair at the same pixel.
        # It is only meaningful for small baselines, which is why --step is
        # small; it is reported but must not be over-read for large motion.
        def gray(x):
            t = resize_img(x, ds.img_size)["img"]
            a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
            a = a.squeeze()                       # (3,H,W) or (H,W,3)
            if a.shape[0] == 3:
                a = a.transpose(1, 2, 0)
            return a.mean(-1).astype(np.float32)
        g1, g2 = gray(im1), gray(im2)
        dis = cv2.GaussianBlur(np.abs(g1 - g2), (9, 9), 0).reshape(-1)

        op_all.append(op); sc_all.append(sc)
        inv_all.append(inv); edge_all.append(edge); dis_all.append(dis)
        conf_all.append(cf); z_all.append(z)

    if not op_all:
        print("no usable pairs"); return 1
    op = np.concatenate(op_all); sc = np.concatenate(sc_all)
    zz = np.concatenate(z_all)
    cc = np.concatenate(conf_all)
    preds = [np.concatenate(inv_all), np.concatenate(edge_all),
             np.concatenate(dis_all), cc, zz]
    names = ["invalid", "edge", "disagree", "conf", "depth"]
    ctrl = {"depth": zz, "conf": cc, "none": None}[args.control]

    print(f"\n{len(op)} gaussians over {len(op_all)} pairs   "
          f"head={args.head or 'BASE'}   {args.dataset}")
    print(f"opacity  median {np.median(op):.4f}  frac>0.9 {(op>0.9).mean()*100:5.1f}%")
    for tag, v in (("OPACITY", op), ("MAXSCALE", sc)):
        # A rank correlation on a variable that is almost entirely tied is not
        # a weak measurement, it is an arbitrary one: argsort breaks ties by
        # position, so a saturated head (99.9% of opacities exactly at 1.0)
        # produces large, stable-looking, meaningless coefficients. Measure the
        # degeneracy and refuse to print rather than inviting the read.
        top = float((v >= v.max() - 1e-6).mean())
        spread = float(np.percentile(v, 90) - np.percentile(v, 10))
        print(f"\n  {tag} decile   mean{tag.lower():>10s} " +
              "".join(f"{n:>12s}" for n in names))
        if top > 0.90:
            print(f"    DEGENERATE: {top*100:.1f}% of values are at the maximum "
                  f"and the p10-p90 spread is {spread:.4g}.")
            print(f"    Rank statistics on this variable are tie-breaking "
                  f"artifacts, not measurements. Not reported.")
            continue
        for d, row in enumerate(decile_profile(v, preds, names)):
            print(f"    d{d}          " + "".join(f"{x:12.5f}" for x in row))
        print("    spearman      " + "".join(
            f"{spearman(v, p):12.4f}" for p in preds))
        if ctrl is not None:
            print(f"    partial|{args.control:<5s}" + "".join(
                f"{partial_spearman(v, p, ctrl):12.4f}" for p in preds))
    print("\nA confidence channel shows a monotone gradient down the OPACITY "
          "deciles.\nA global re-gauge shows a flat profile. Read the gradient, "
          "not the sign alone.")
    if args.control == "depth":
        print("The partial|depth row is the one that counts: TUM depth drops "
              "out on far/glossy\nsurfaces and Gaussian scale grows with range, "
              "so any raw correlation with\n`invalid` may be a distance effect. "
              "What survives inside a depth band is not.")
    elif args.control == "conf":
        print("partial|conf asks whether `invalid` is just `conf` measured "
              "twice. Whatever\nsurvives here is signal the sensor mask carries "
              "that the backbone's confidence\ndoes not. The conf column itself "
              "is ~0 by construction and is not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

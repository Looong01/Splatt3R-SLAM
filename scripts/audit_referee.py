"""Is the referee right? Score its corrections against the RGB-D sensor.

Route R applied VGGT's per-cluster scale corrections to a built map and made it
worse in every form (§17.48). Two explanations are on the table and they call
for opposite next moves:

  A. the trajectory has already absorbed the jitter, so the map is at a joint
     optimum and the referee's absolute criterion is not the map's criterion;
  B. the referee is simply wrong HERE -- §17.45 validated VGGT on windows of 16
     views at keyframe spacing (a few seconds apart), but a map's keyframes span
     the whole sequence, and joint consistency over 14 widely separated views is
     a different operating point.

The sensor decides it. For each keyframe:

    c_k  = slope of z_map against z_sensor      the map's own scale error
    v_k  = slope of z_ref against z_sensor      the referee's scale error

both gauge-normalized by their median. If the referee were perfect, v_k = 1 for
all k and its correction s_k would equal 1/c_k. So the audit is the correlation
between the correction the referee proposed and the correction the sensor says
is needed -- and the referee's own spread, which §17.45 puts at 1.68% at ITS
operating point.
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
from scipy.stats import pearsonr, spearmanr


def robust_slope(a, b):
    p = np.polyfit(a, b, 1)
    res = np.abs(b - np.polyval(p, a))
    keep = res < 2.5 * np.median(res)
    if keep.sum() > 2000:
        p = np.polyfit(a[keep], b[keep], 1)
    return float(p[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True)
    ap.add_argument("--traj", required=True, help="keyframe trajectory, one "
                                                  "line per keyframe")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min-confidence", type=float, default=1.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    dev = args.device
    root = os.path.abspath(args.dataset)
    blob = torch.load(args.kfgauss, map_location="cpu")
    kfs = blob["keyframes"]
    n = len(kfs)

    est_ts = [float(l.split()[0]) for l in open(args.traj)
              if l.strip() and not l.startswith("#")]
    assert len(est_ts) == n, f"{len(est_ts)} traj lines for {n} keyframes"

    d_ts, d_paths = [], []
    for line in open(os.path.join(root, "depth.txt")):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) == 2:
            d_ts.append(float(p[0])); d_paths.append(p[1])
    d_ts = np.array(d_ts)

    H, W = 392, 518
    imgs = []
    for kf in kfs:
        im = kf["img"].permute(1, 2, 0).numpy()
        im = np.asarray(Image.fromarray(np.uint8(np.clip(im, 0, 1) * 255))
                        .resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(im).permute(2, 0, 1))
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        out = model(torch.stack(imgs).to(dev))
    depth = out["depth"][0, ..., 0].float().cpu().numpy()
    del out

    cmap, cref = [], []
    for k, kf in enumerate(kfs):
        h, w = [int(v) for v in kf["img_shape"]]
        j = int(np.argmin(np.abs(d_ts - est_ts[k])))
        if abs(d_ts[j] - est_ts[k]) > 0.05:
            cmap.append((np.nan, np.nan)); cref.append((np.nan, np.nan)); continue
        ds = np.asarray(Image.open(os.path.join(root, d_paths[j])),
                        dtype=np.float32) / 5000.0
        ds = np.asarray(Image.fromarray(ds).resize((w, h), Image.NEAREST))
        zmap = kf["means"][:, 2].numpy().reshape(h, w)
        zref = np.asarray(Image.fromarray(depth[k]).resize((w, h),
                                                           Image.BILINEAR))
        ok = (kf["conf"].numpy().reshape(h, w) > args.min_confidence)
        ok &= (ds > 0.3) & (ds < 6.0) & (zmap > 0.05) & (zref > 0.05)
        if ok.sum() < 2000:
            cmap.append((np.nan, np.nan)); cref.append((np.nan, np.nan)); continue
        cmap.append((robust_slope(ds[ok], zmap[ok]),
                     float(np.median(zmap[ok] / ds[ok]))))
        cref.append((robust_slope(ds[ok], zref[ok]),
                     float(np.median(zref[ok] / ds[ok]))))

    cmap, cref = np.array(cmap), np.array(cref)
    g = np.isfinite(cmap).all(1) & np.isfinite(cref).all(1)
    print(f"\n{n} keyframes, {g.sum()} usable, spanning "
          f"{est_ts[-1] - est_ts[0]:.1f} s")
    # BOTH forms, because §17.45's headline number is a median-RATIO spread
    # while route R applies a SLOPE correction, and with a reference that
    # carries a +30% depth shift those are not the same quantity. Comparing
    # one against the other would be exactly the family of error this section
    # keeps recording.
    for j, form in enumerate(("slope", "ratio")):
        a = cmap[g, j] / np.median(cmap[g, j])
        b = cref[g, j] / np.median(cref[g, j])
        print(f"  [{form}] map {np.median(np.abs(a - 1)) * 100:5.2f}%"
              f"   referee {np.median(np.abs(b - 1)) * 100:5.2f}%"
              + ("    <- 17.45's form, which measured 1.68% on windows"
                 if form == "ratio" else ""))
    # The RATIO form from here: the two columns above show the slope estimator
    # is the noisy one, making the referee look 2-7x worse on the same data.
    # The shift-cancellation algebra behind the slope was right and the
    # estimator it produced was still worse -- which is also why route R's
    # slope arm did MORE damage to the rendering than its ratio arm.
    cm, cr = cmap[g, 1], cref[g, 1]
    cm, cr = cm / np.median(cm), cr / np.median(cr)
    # No correlation is reported here on purpose. The referee's proposal is
    # s_k = v_k / c_k and the ideal correction is 1 / c_k, so the two share the
    # factor 1/c_k and correlate by construction -- an earlier version of this
    # script printed that number and it meant nothing. What the composition
    # actually gives is c_k * (v_k / c_k) = v_k: applying the referee replaces
    # the map's per-keyframe scale error with the REFEREE'S. So the whole audit
    # is the two columns, and the verdict is simply whether v is below c.
    a = np.median(np.abs(cm - 1)) * 100
    b = np.median(np.abs(cr - 1)) * 100
    print(f"\n  applying the referee replaces the map's per-keyframe scale "
          f"error with the referee's:\n    {a:5.2f}%  ->  {b:5.2f}%   "
          f"({'improves' if b < a else 'WORSENS'} absolute geometry "
          f"{a / max(b, 1e-9):.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Visualise a refined map as INDIVIDUAL projected Gaussian splats.

Each Gaussian is drawn as a projected coloured ellipse (not alpha-composited
into a smooth photo), so the reader literally sees the primitives. Camera is a
fitted oblique orbit around the map. Outputs transparent and dark variants.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from splatt3r_slam.gaussian_ply_codec import decode_gaussians_from_ply
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection


def orbit_camera(center, radius, az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    eye = center + radius * np.array(
        [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    fwd = center - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    w2c = np.eye(4)
    w2c[0, :3] = right
    w2c[1, :3] = up
    w2c[2, :3] = -fwd
    w2c[:3, 3] = -(w2c[:3, :3] @ eye)
    return w2c


@torch.no_grad()
def render_gaussians(ply, out, size=(1300, 950), az=38, el=24,
                     alpha_scale=0.5, dark=False, device="cuda:0"):
    g = decode_gaussians_from_ply(ply, device=device)
    means = g["means"].double().cpu().numpy()
    cov = g["covariances"].double().cpu().numpy()
    rgb = g["rgb"].double().clamp(0, 1).cpu().numpy()
    opa = g["opacity"].double().cpu().numpy()
    n = len(means)

    center = np.median(means, axis=0)
    dist = np.linalg.norm(means - center, axis=1)
    extent = np.quantile(dist, 0.98)
    w2c = orbit_camera(center, extent * 2.45, az, el)
    R, t = w2c[:3, :3], w2c[:3, 3]
    pc = means @ R.T + t
    in_front = pc[:, 2] < -1e-6
    z = -pc[:, 2]

    W, H = size
    fy = (H / 2) / np.tan(np.radians(47) / 2)
    fx = fy
    cx, cy = W / 2, H / 2

    u = fx * pc[:, 0] / z + cx
    v = cy - fy * pc[:, 1] / z
    J = np.zeros((n, 2, 3))
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * pc[:, 0] / z**2
    J[:, 1, 1] = -fy / z
    J[:, 1, 2] = fy * pc[:, 1] / z**2
    cov2 = J @ cov @ J.transpose(0, 2, 1)

    evals, evecs = np.linalg.eigh(cov2)
    ax = np.sqrt(np.maximum(evals[:, 1], 0.5))
    bx = np.sqrt(np.maximum(evals[:, 0], 0.5))
    theta = np.degrees(np.arctan2(evecs[:, 1, 1], evecs[:, 0, 1]))

    margin = 150
    zfloor = np.quantile(z[in_front], 0.005)
    keep = (in_front & (u > -margin) & (u < W + margin)
            & (v > -margin) & (v < H + margin)
            & (ax >= 1.2) & (ax <= 320) & (opa > 0.08) & (z > zfloor))
    ki = np.nonzero(keep)[0]
    cap = 140000
    if len(ki) > cap:
        rng = np.random.default_rng(0)
        p = opa[ki] / opa[ki].sum()
        ki = rng.choice(ki, cap, replace=False, p=p)

    ki = ki[np.argsort(-z[ki])]

    # Tight screen-space framing from the CORE splats (dense, opaque surfaces),
    # ignoring long background streaks and far outliers.
    core = in_front & (opa > 0.2) & (ax > 1.5) & (ax < 90)
    cu, cvel = u[core], v[core]
    xa0, xa1 = np.quantile(cu, [0.02, 0.98])
    ya0, ya1 = np.quantile(cvel, [0.02, 0.98])
    mx, my = (xa0 + xa1) / 2, (ya0 + ya1) / 2
    bw, bh = xa1 - xa0, ya1 - ya0
    ar = W / H
    if bw / bh > ar:
        bh = bw / ar
    else:
        bw = bh * ar
    pad = 1.16
    xlim = (mx - bw * pad / 2, mx + bw * pad / 2)
    ylim = (my + bh * pad / 2, my - bh * pad / 2)

    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    axp = fig.add_axes([0, 0, 1, 1])
    axp.set_xlim(*xlim)
    axp.set_ylim(*ylim)
    axp.axis("off")
    if dark:
        fig.patch.set_facecolor((0.047, 0.063, 0.075, 1))
    else:
        fig.patch.set_alpha(0)

    nlay = 6
    for li in range(nlay):
        seg = ki[li::nlay]
        if len(seg) == 0:
            continue
        ec = EllipseCollection(
            2 * ax[seg], 2 * bx[seg], theta[seg], units="xy",
            offsets=np.c_[u[seg], v[seg]], offset_transform=axp.transData)
        ec.set_facecolors(
            np.c_[rgb[seg], np.clip(opa[seg] * alpha_scale, 0.02, 0.6)])
        ec.set_edgecolors("none")
        axp.add_collection(ec)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, transparent=not dark)
    plt.close(fig)
    print(out, "gaussians=", n, "drawn=", len(ki))


SCENES = {
    "desk": "logs/ref_onoff_tum_on/rgbd_dataset_freiburg1_desk_refined.ply",
    "fr1_360": "logs/p2seven_360/rgbd_dataset_freiburg1_360_refined.ply",
    "table_3": "logs/ho_eth3d_head/table_3_refined.ply",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", nargs="+", default=list(SCENES))
    p.add_argument("--outdir", type=Path,
                   default=ROOT / "logs/paper_real_20261010/gaussview")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    for key in args.scenes:
        ply = ROOT / SCENES[key]
        for variant, dark in [("clean", False), ("dark", True)]:
            render_gaussians(ply, args.outdir / f"{key}_{variant}.png",
                             dark=dark, device=args.device)


if __name__ == "__main__":
    main()

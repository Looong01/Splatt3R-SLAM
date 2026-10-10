"""Render OUR refined Gaussian maps on REAL datasets at held-out, GT-aligned views.

Real scenes only (TUM, ETH3D); Replica is synthetic and intentionally excluded.
No baselines and no training: this produces (real photo, our render) pairs used
by the teaser and the multi-dataset qualitative figure. Pairs reuse the exact
Sim(3)-alignment + exposure protocol from ``render_comparisons.py`` so each
render matches the shown photo.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "splatt3r_core/src/pixelsplat_src"))
sys.path.insert(0, str(ROOT / "splatt3r_core/src/mast3r_src"))
sys.path.insert(0, str(ROOT / "splatt3r_core/src/mast3r_src/dust3r"))

import numpy as np
import torch
from PIL import Image

from eval_map_quality import load_tum_traj, umeyama_sim3, associate
from render_comparisons import exported_sh, render_full
from splatt3r_slam.gaussian_ply_codec import decode_gaussians_from_ply

SCENES = {
    "desk": {
        "dataset": "datasets/tum/rgbd_dataset_freiburg1_desk",
        "ply": "logs/ref_onoff_tum_on/rgbd_dataset_freiburg1_desk_refined.ply",
        "traj": "logs/ref_onoff_tum_on/rgbd_dataset_freiburg1_desk.txt",
    },
    "fr1_360": {
        "dataset": "datasets/tum/rgbd_dataset_freiburg1_360",
        "ply": "logs/p2seven_360/rgbd_dataset_freiburg1_360_refined.ply",
        "traj": "logs/p2seven_360/rgbd_dataset_freiburg1_360.txt",
    },
    "table_3": {
        "dataset": "datasets/eth3d/train/table_3",
        "ply": "logs/ho_eth3d_head/table_3_refined.ply",
        "traj": "logs/ho_eth3d_head/table_3.txt",
    },
    "cables_1": {
        "dataset": "datasets/eth3d/train/cables_1",
        "ply": "logs/h4_eth3d4_base/cables_1_refined.ply",
        "traj": "logs/h4_eth3d4_base/cables_1.txt",
    },
}


def u8(t):
    return (t[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def render_scene(key, spec, output, n, device):
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.image import normalize_exposure, reset_exposure_reference

    load_config(str(ROOT / "config/eval_calib.yaml"))
    ds_path = ROOT / spec["dataset"]
    ply, traj = ROOT / spec["ply"], ROOT / spec["traj"]
    assert ds_path.is_dir() and ply.is_file() and traj.is_file()
    ds = load_dataset(str(ds_path))
    gt_ts, gt_T = load_tum_traj(ds_path / "groundtruth.txt")
    est_ts, est = load_tum_traj(traj)
    pairs = associate(est_ts, gt_ts)
    assert len(pairs) >= 3
    gt = np.array([gt_T[j] for _, j in pairs])
    estm = np.array([est[i] for i, _ in pairs])
    s, R, t = umeyama_sim3(estm[:, :3, 3], gt[:, :3, 3])

    excluded = set(j for _, j in associate(est_ts, np.asarray(ds.timestamps, float)))
    candidates = [(i, j) for i, j in
                  associate(np.asarray(ds.timestamps, float), gt_ts)
                  if i not in excluded]
    assert candidates
    step = max(1, len(candidates) // n)
    held = candidates[::step][:n]

    g = decode_gaussians_from_ply(ply, device=device)
    sh = exported_sh(ply, g, device)
    reset_exposure_reference()
    normalize_exposure(ds.get_image(0))

    out = output / key
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"scene": key, "dataset": spec["dataset"],
                "gaussians": g["n"], "views": []}
    for k0, (i, j) in enumerate(held):
        target_img = resize_img(normalize_exposure(ds.get_image(i)), ds.img_size)["img"]
        target = torch.as_tensor(target_img, dtype=torch.float32) * .5 + .5
        hw = tuple(target.shape[-2:])
        c2w = np.eye(4)
        c2w[:3, :3] = R.T @ gt_T[j, :3, :3]
        c2w[:3, 3] = R.T @ (gt_T[j, :3, 3] - t) / s
        pred = render_full(g, sh, c2w, ds.camera_intrinsics.K_frame, hw, device)
        Image.fromarray(u8(target.to(device))).save(out / f"cand{k0}_gt.png")
        Image.fromarray(u8(pred)).save(out / f"cand{k0}_ours.png")
        manifest["views"].append({"index": i, "timestamp": float(ds.timestamps[i])})
        print(key, k0, "rendered", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    del g, sh
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", nargs="+", default=list(SCENES))
    p.add_argument("--output", type=Path, default=ROOT / "logs/paper_real_20261010")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    for key in args.scenes:
        render_scene(key, SCENES[key], args.output.resolve(), args.n, args.device)


if __name__ == "__main__":
    main()

"""Render one real TUM view before and after map refinement."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_map_quality import associate, decode_gaussians_from_ply, load_tum_traj
from paper.render_comparisons import exported_sh, render_full, u8


@torch.no_grad()
def main():
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.image import normalize_exposure, reset_exposure_reference
    from splatt3r_slam.splatt3r_utils import resize_img

    device = "cuda:0"
    load_config(str(ROOT / "config/eval_calib.yaml"))
    scene = ROOT / "datasets/tum/rgbd_dataset_freiburg1_desk"
    record = json.loads(
        (ROOT / "logs/paper_ral_20261007/desk/metrics.json").read_text()
    )
    frame_row = record["selection"]["row_indices"][0]
    frame_index = record["frames"][frame_row]["index"]

    dataset = load_dataset(str(scene))
    gt_ts, gt_poses = load_tum_traj(scene / "groundtruth.txt")
    dataset_ts = np.asarray(dataset.timestamps, dtype=float)
    gt_lookup = dict(associate(dataset_ts, gt_ts))
    gt_pose = gt_poses[gt_lookup[frame_index]]

    alignment = record["methods"]["ours"]["alignment"]
    scale = float(alignment["s"])
    rotation = np.asarray(alignment["R"])
    translation = np.asarray(alignment["t"])
    camera = np.eye(4)
    camera[:3, :3] = rotation.T @ gt_pose[:3, :3]
    camera[:3, 3] = rotation.T @ (gt_pose[:3, 3] - translation) / scale

    reset_exposure_reference()
    normalize_exposure(dataset.get_image(0))
    target = resize_img(
        normalize_exposure(dataset.get_image(frame_index)), dataset.img_size
    )["img"]
    target = torch.as_tensor(target, device=device, dtype=torch.float32) * 0.5 + 0.5
    height_width = tuple(target.shape[-2:])

    source = ROOT / "logs/ref_onoff_tum_on"
    stem = "rgbd_dataset_freiburg1_desk"
    maps = {
        "before": source / f"{stem}_gaussians.ply",
        "after": source / f"{stem}_refined.ply",
    }
    output = ROOT / "docs/Thesis/ral/fig/overview_assets"
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8(target)).save(output / "target.png")

    for label, path in maps.items():
        gaussians = decode_gaussians_from_ply(path, device=device)
        sh = exported_sh(path, gaussians, device)
        rendered = render_full(
            gaussians,
            sh,
            camera,
            dataset.camera_intrinsics.K_frame,
            height_width,
            device,
        )
        Image.fromarray(u8(rendered)).save(output / f"{label}.png")
        del gaussians, sh, rendered
        torch.cuda.empty_cache()

    metadata = {
        "scene": "TUM freiburg1_desk",
        "evaluation_frame_index": int(frame_index),
        "selection": record["selection"],
        "camera_source": "GT camera transformed with the recorded Ours Sim(3)",
        "before_map": str(maps["before"].relative_to(ROOT)),
        "after_map": str(maps["after"].relative_to(ROOT)),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

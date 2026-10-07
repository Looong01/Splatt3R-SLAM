"""Re-render saved SLAM maps for the RA-L paper; no training or map edits.

Run in the splatt3r-slam environment from the repository root. All exported SH
coefficients are retained. DC-only scores are also saved for comparison with
the historical evaluator. Non-keyframe does NOT imply unseen by refinement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_map_quality import (associate, load_tum_traj, umeyama_sim3,
                              decode_gaussians_from_ply, render_map)
import numpy as np
import torch
from PIL import Image


def fingerprint(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
            "sha256": h.hexdigest()}


def exported_sh(path, g, device):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = sorted((n for n in v.data.dtype.names if n.startswith("f_rest_")),
                   key=lambda n: int(n[7:]))
    dc = g["f_dc"][..., None]
    if not names:
        return dc
    rest = np.stack([v[n] for n in names], axis=1).copy()
    rest = torch.as_tensor(rest, device=device).reshape(g["n"], 3, -1)
    sh = torch.cat([dc, rest], -1)
    assert math.isqrt(sh.shape[-1]) ** 2 == sh.shape[-1]
    return sh


@torch.no_grad()
def render_full(g, sh, pose, K, hw, device):
    from src.pixelsplat_src.cuda_splatting import render_cuda
    h, w = hw
    k = torch.as_tensor(K, device=device, dtype=torch.float32).clone()[None]
    k[:, 0] /= w
    k[:, 1] /= h
    return render_cuda(
        torch.as_tensor(pose, device=device, dtype=torch.float32)[None],
        k, torch.tensor([0.1], device=device), torch.tensor([1000.], device=device),
        hw, torch.zeros(1, 3, device=device), g["means"][None],
        g["covariances"][None], sh[None], g["opacity"][None],
    ).reshape(1, 3, h, w).clamp(0, 1)


def photo_paths(scene):
    d = ROOT / "tmp/Photo-SLAM/results" / scene
    ply, = d.glob("*_shutdown/ply/point_cloud/iteration_*/point_cloud.ply")
    return {"ply": ply, "traj": d / "KeyFrameTrajectory_TUM.txt"}


def scene_paths(scene):
    if scene != "desk":
        d = ROOT / f"logs/cmp_replica_{scene}"
        return ROOT / f"datasets/Replica/{scene}", {
            "photo": photo_paths(f"replica_{scene}"),
            "ours": {"ply": d / f"{scene}_refined.ply", "traj": d / f"{scene}.txt"},
        }
    seq = "rgbd_dataset_freiburg1_desk"
    d = ROOT / "logs/ref_onoff_tum_on"
    mono = ROOT / "tmp/MonoGS/results/tum_rgbd_dataset_freiburg1_desk/2026-08-17-22-47-19"
    return ROOT / "datasets/tum" / seq, {
        "photo": photo_paths("fr1_desk"),
        "mono": {"ply": mono / "point_cloud/final/point_cloud.ply",
                 "mono_json": mono / "plot/trj_final.json"},
        "ours": {"ply": d / f"{seq}_refined.ply", "traj": d / f"{seq}.txt"},
    }


def alignments(methods, dataset, gt_ts, gt_T):
    ts = np.asarray(dataset.timestamps, dtype=float)
    excluded, out = set(), {}
    for name, spec in methods.items():
        if "mono_json" in spec:
            d = json.loads(spec["mono_json"].read_text())
            est, gt = np.array(d["trj_est"]), np.array(d["trj_gt"])
            # MonoGS's parser selects RGB frames with matched RGB-D/GT at
            # 32 Hz. Recover indices by nearest GT pose, NOT raw RGB indexing.
            distances = ((gt[:, None, :3, 3] - gt_T[None, :, :3, 3]) ** 2).sum(-1)
            matched = distances.argmin(1)
            # Check orientation as well before accepting the correspondence.
            assert np.max(np.abs(gt[:, :3, :3] - gt_T[matched, :3, :3])) < 1e-3
            est_ts = gt_ts[matched]
        else:
            est_ts, est = load_tum_traj(spec["traj"])
            pairs = associate(est_ts, gt_ts)
            assert len(pairs) >= 3
            gt = np.array([gt_T[j] for _, j in pairs])
            est = np.array([est[i] for i, _ in pairs])
        excluded.update(j for _, j in associate(est_ts, ts))
        s, R, t = umeyama_sim3(est[:, :3, 3], gt[:, :3, 3])
        out[name] = (s, R, t)
    return excluded, out


def u8(t):
    return (t[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def evaluate(scene, output, n, device):
    import lpips
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.image import normalize_exposure, reset_exposure_reference

    load_config(str(ROOT / "config/eval_calib.yaml"))
    ds_path, methods = scene_paths(scene)
    for spec in methods.values():
        for p in spec.values():
            assert p.is_file(), p
    ds = load_dataset(str(ds_path))
    gt_ts, gt_T = load_tum_traj(ds_path / "groundtruth.txt")
    excluded, transforms = alignments(methods, ds, gt_ts, gt_T)
    candidates = [(i, j) for i, j in associate(np.array(ds.timestamps, float), gt_ts)
                  if i not in excluded]
    held = candidates[::max(1, len(candidates) // n)][:n]
    assert len(held) == n
    reset_exposure_reference()
    normalize_exposure(ds.get_image(0))
    targets = []
    for i, _ in held:
        a = resize_img(normalize_exposure(ds.get_image(i)), ds.img_size)["img"]
        targets.append(torch.as_tensor(a, dtype=torch.float32) * .5 + .5)
    lp = lpips.LPIPS(net="alex").to(device)
    out = output / scene
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "scene": scene, "dataset": str(ds_path.relative_to(ROOT)),
        "protocol": "shared non-keyframes; GT Sim3; normalized targets; full exported SH",
        "resolution_hw": list(targets[0].shape[-2:]),
        "exclusion_is_not_strict_training_holdout": True,
        "psnr_aggregation": "-10 log10(mean frame MSE)",
        "frames": [{"index": i, "timestamp": float(ds.timestamps[i])} for i, _ in held],
        "methods": {},
    }
    pictures = {}
    for name, spec in methods.items():
        g = decode_gaussians_from_ply(spec["ply"], device=device)
        sh = exported_sh(spec["ply"], g, device)
        s, R, t = transforms[name]
        full, dc, images = [], [], []
        for row, (_, j) in enumerate(held):
            c2w = np.eye(4)
            c2w[:3, :3] = R.T @ gt_T[j, :3, :3]
            c2w[:3, 3] = R.T @ (gt_T[j, :3, 3] - t) / s
            target = targets[row].to(device)
            hw = tuple(target.shape[-2:])
            pred = render_full(g, sh, c2w, ds.camera_intrinsics.K_frame, hw, device)
            fm = ((pred - target)**2).mean().item()
            fl = lp(pred, target, normalize=True).mean().item()
            full.append({"mse": fm, "psnr": -10*math.log10(fm), "lpips": fl})
            images.append(u8(pred))
            if sh.shape[-1] > 1:
                pdc = render_map(g, c2w, ds.camera_intrinsics.K_frame, hw, device).clamp(0, 1)
                md = ((pdc - target)**2).mean().item()
                ld = lp(pdc, target, normalize=True).mean().item()
                dc.append({"mse": md, "psnr": -10*math.log10(md), "lpips": ld})
            else:
                dc.append(full[-1].copy())
        def aggregate(rows):
            return {"psnr": -10 * math.log10(np.mean([r["mse"] for r in rows])),
                    "lpips": float(np.mean([r["lpips"] for r in rows]))}
        record["methods"][name] = {
            "artifacts": {k: fingerprint(v) for k, v in spec.items()},
            "gaussians": g["n"], "sh_degree": math.isqrt(sh.shape[-1])-1,
            "alignment": {"s": s, "R": R.tolist(), "t": t.tolist()},
            "full_sh": aggregate(full), "dc_only": aggregate(dc), "per_frame": full,
        }
        pictures[name] = images
        print(scene, name, record["methods"][name]["full_sh"], flush=True)
        del g, sh
        torch.cuda.empty_cache()
    # Pick the actual median delta (two central ranks), not a favourable tail.
    reference = "mono" if "mono" in methods else "photo"
    rows_o = record["methods"]["ours"]["per_frame"]
    rows_r = record["methods"][reference]["per_frame"]
    order = np.argsort([a["psnr"]-b["psnr"] for a, b in zip(rows_o, rows_r)])
    chosen = [int(order[n//2 - 1]), int(order[n//2])]
    record["selection"] = {"rule": f"two middle ranks of ours-minus-{reference} PSNR",
                           "row_indices": chosen}
    for rank, row in enumerate(chosen):
        Image.fromarray(u8(targets[row])).save(out / f"view{rank}_gt.png")
        for name in methods:
            Image.fromarray(pictures[name][row]).save(out / f"view{rank}_{name}.png")
    (out / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--output", type=Path, default=ROOT / "logs/paper_ral_20261007")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    for scene in args.scenes:
        evaluate(scene, args.output.resolve(), args.n, args.device)

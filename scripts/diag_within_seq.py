"""Does the parallax effect reproduce WITHIN a sequence?

§17.37 established that refinement's perceptual gain is set by the trajectory's
translation/rotation ratio (partial r = -0.975 after controlling for veils), and
framed it as an information boundary. That claim has one serious confound, named
in §17.37 and not yet broken: **across seven sequences, motion type covaries with
scene type** -- rpy and 360 are room-scale rotations, plant and teddy are
object-centric translations. A reviewer will say the correlation is about scenes.

The only way to break it is to look inside a single sequence, where the scene is
held fixed by construction. Every sequence contains both rotation-dominated and
translation-dominated stretches; if the mechanism is real, the gain must track
local parallax *within* a sequence too.

Method: score every held-out frame twice, against the baked map and against the
polished map from the same run, and attribute to each frame the local
translation/rotation ratio of the trajectory around it. Then correlate the
per-frame gain with the per-frame ratio, WITHIN each sequence.

No new runs -- both maps are already on disk.

Reading:
  within-sequence correlation holds  -> scene confound broken, boundary stands
  correlation vanishes within        -> the cross-sequence result is about
                                        scenes, and §17.37 must be demoted to a
                                        seven-point observation
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


def local_ratio(est_ts, est_T, t, half=6):
    """Translation/rotation ratio of the trajectory in a window around time t.

    The same quantity §17.34 computed globally, evaluated locally so a frame in a
    rotation-dominated stretch of a mostly-translating sequence is labelled as
    such. Median over the window, not mean: a single large step should not
    relabel a whole neighbourhood.
    """
    i = int(np.argmin(np.abs(est_ts - t)))
    lo, hi = max(0, i - half), min(len(est_ts) - 1, i + half)
    if hi - lo < 3:
        return None
    p = np.array([est_T[k][:3, 3] for k in range(lo, hi + 1)])
    R = [est_T[k][:3, :3] for k in range(lo, hi + 1)]
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    ang = []
    for a, b in zip(R[:-1], R[1:]):
        c = (np.trace(a.T @ b) - 1) / 2
        ang.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    ang = np.array(ang)
    md, ma = np.median(d), np.median(ang)
    return float(md / max(ma, 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baked", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    import lpips as lpips_lib
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from eval_map_quality import decode_gaussians_from_ply, render_map

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
    held = cand[:: max(1, len(cand) // args.n)][: args.n]

    lp = lpips_lib.LPIPS(net="alex").to(dev)
    gb = decode_gaussians_from_ply(args.baked, device=dev)
    gr = decode_gaussians_from_ply(args.refined, device=dev)
    K = ds.camera_intrinsics.K_frame
    kf_pos = np.array([est_T[i][:3, 3] for i in range(len(est_ts))])

    rows = []
    with torch.no_grad():
        for di, gj in held:
            rho = local_ratio(est_ts, est_T, ds_ts[di])
            if rho is None:
                continue
            img = resize_img(ds.get_image(di), ds.img_size)["img"]
            tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
            h, w = tgt.shape[-2:]
            c = np.eye(4)
            c[:3, :3] = Rt @ gt_T[gj][:3, :3]
            c[:3, 3] = Rt @ (gt_T[gj][:3, 3] - t_) / s_
            out = []
            for g in (gb, gr):
                p = render_map(g, c, K, (h, w), dev).reshape(1, 3, h, w).clamp(0, 1)
                out.append(float(lp(p, tgt, normalize=True).mean()))
            # MEDIATOR (Kimi r13): the trajectory ratio barely varies inside a
            # sequence -- 1.2-2x against 15x across -- so a within-sequence test
            # on it is underpowered by construction. The mechanism variable does
            # vary: how much PARALLAX the supervision had for the content this
            # frame sees. Approximated by the spread of the keyframe positions
            # near this viewpoint, normalized by depth to make it an angle.
            d = np.linalg.norm(kf_pos - c[:3, 3], axis=1)
            near = np.argsort(d)[:8]
            sup = kf_pos[near]
            sup_par = float(np.max(np.linalg.norm(
                sup[:, None] - sup[None], axis=-1))) / max(float(np.median(d)), 1e-6)
            rows.append((rho, out[0], out[1], (out[1] - out[0]) / out[0] * 100,
                         sup_par))

    a = np.array(rows)
    from scipy.stats import spearmanr
    name = os.path.basename(args.refined)
    print(f"\n{name}: {len(a)} held-out frames")
    print(f"{'local ratio bin':>18} {'n':>4} {'baked':>8} {'refined':>9} {'gain%':>8}")
    qs = np.quantile(a[:, 0], [0, .25, .5, .75, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (a[:, 0] >= lo) & (a[:, 0] <= hi)
        if m.sum():
            print(f"{lo:8.3f}-{hi:8.3f} {int(m.sum()):>4} {a[m,1].mean():>8.4f} "
                  f"{a[m,2].mean():>9.4f} {a[m,3].mean():>7.1f}%")
    print(f"\n  WITHIN-SEQUENCE spearman(local ratio,  lpips gain) = "
          f"{spearmanr(a[:,0], a[:,3]).statistic:+.3f}   "
          f"range {a[:,0].max()/max(a[:,0].min(),1e-9):.1f}x")
    print(f"  WITHIN-SEQUENCE spearman(supervision parallax, gain) = "
          f"{spearmanr(a[:,4], a[:,3]).statistic:+.3f}   "
          f"range {a[:,4].max()/max(a[:,4].min(),1e-9):.1f}x")
    print("  The second is the mediator: it has the dynamic range the first "
          "lacks, and it tests the mechanism rather than its proxy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

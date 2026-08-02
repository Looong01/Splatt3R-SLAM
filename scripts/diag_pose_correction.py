"""Is photometric pose refinement a pose refiner, or a photometric sponge?

`mean pose translation correction` cannot tell these apart. Both move the
cameras; only one moves them toward the truth. The distinction decides whether
the whole pose-optimization line (and with it the case for camera gradients in
CUDA) is worth further investment, because the measured behaviour so far is
ambiguous:

  - the pose learning rate behaves as a pure perception-distortion dial --
    higher LR buys psnr and costs lpips, monotonically, which is what a sponge
    absorbing photometric residual looks like;
  - but the LR=1e-4 setting beats its own fixed-pose control on BOTH metrics,
    which a pure sponge should not be able to do.

Ground truth makes it decidable offline. Three statistics, per supervision view:

  toward-GT rate   ||opt - GT|| vs ||est - GT||. A refiner shrinks the error on
                   most views; a sponge scatters around the initial value.
  direction cosine cos(delta, GT - est). A refiner points at the truth; a
                   sponge points wherever the residual pulls.
  stratification   keyframe vs non-keyframe. Non-keyframes start further from
                   the truth (their poses are never revisited by the pose-graph
                   backend: ATE 0.0291 m vs 0.0170 m on desk), so a real
                   refiner must correct them MORE. A sponge has no reason to
                   care which is which.

Plus one statistic that needs no ground truth, so it can be carried into the
online system as a proxy: the temporal smoothness of the correction sequence.
True pose drift is low-frequency; sponge noise is high-frequency.

Usage:
    python3 scripts/diag_pose_correction.py --poses logs/poses_lr1e-4.npz \
        --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
        --traj logs/frames_head/rgbd_dataset_freiburg1_desk.txt \
        --frames-traj logs/frames_head/rgbd_dataset_freiburg1_desk_frames.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from eval_map_quality import associate, load_tum_traj, umeyama_sim3


def rot_angle(R):
    """Geodesic angle of a rotation matrix, in degrees."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True, help="npz from --save-poses")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--traj", required=True, help="keyframe trajectory, for the Sim3 fit")
    ap.add_argument("--frames-traj", required=True)
    args = ap.parse_args()

    z = np.load(args.poses, allow_pickle=True)
    init, opt = z["init"], z["opt"]
    ref = z["ref"]

    frm_ts, _ = load_tum_traj(args.frames_traj)
    est_ts, est_T = load_tum_traj(args.traj)
    gt_ts, gt_T = load_tum_traj(os.path.join(args.dataset, "groundtruth.txt"))

    # Same Sim3 the refinement used to place ground truth in the map's frame.
    pairs = associate(est_ts, gt_ts)
    s, R, t = umeyama_sim3(np.array([est_T[i, :3, 3] for i, _ in pairs]),
                           np.array([gt_T[j, :3, 3] for _, j in pairs]))
    Rt = R.T

    def to_map(c2w_gt):
        m = np.eye(4)
        m[:3, :3] = Rt @ c2w_gt[:3, :3]
        m[:3, 3] = Rt @ (c2w_gt[:3, 3] - t) / s
        return m

    kf_ts = set(np.round(est_ts, 6).tolist())
    rows = []
    for k in range(len(init)):
        ts = frm_ts[int(ref[k])]
        g = associate(np.array([ts]), gt_ts)
        if not g:
            continue
        gt = to_map(gt_T[g[0][1]])
        e_before = np.linalg.norm(init[k][:3, 3] - gt[:3, 3])
        e_after = np.linalg.norm(opt[k][:3, 3] - gt[:3, 3])
        true_corr = gt[:3, 3] - init[k][:3, 3]
        delta = opt[k][:3, 3] - init[k][:3, 3]
        nd, nt = np.linalg.norm(delta), np.linalg.norm(true_corr)
        cos = float(delta @ true_corr / (nd * nt)) if nd > 1e-12 and nt > 1e-12 else np.nan
        rot_before = rot_angle(init[k][:3, :3].T @ gt[:3, :3])
        rot_after = rot_angle(opt[k][:3, :3].T @ gt[:3, :3])
        rows.append(dict(is_kf=round(float(ts), 6) in kf_ts,
                         e_before=e_before, e_after=e_after, cos=cos,
                         dnorm=nd, tnorm=nt,
                         rot_before=rot_before, rot_after=rot_after,
                         delta=delta))

    if not rows:
        print("no views could be associated to ground truth"); return 1

    def report(name, sel):
        if not sel:
            print(f"  {name:14s}  (none)"); return
        eb = np.array([r["e_before"] for r in sel])
        ea = np.array([r["e_after"] for r in sel])
        cs = np.array([r["cos"] for r in sel])
        cs = cs[~np.isnan(cs)]
        rb = np.array([r["rot_before"] for r in sel])
        ra = np.array([r["rot_after"] for r in sel])
        improved = float((ea < eb).mean())
        print(f"  {name:14s} n={len(sel):3d} | "
              f"trans err {eb.mean():.4f} -> {ea.mean():.4f} m "
              f"({100*(ea.mean()-eb.mean())/eb.mean():+.1f}%)  "
              f"improved on {100*improved:.0f}% of views | "
              f"rot err {rb.mean():.3f} -> {ra.mean():.3f} deg | "
              f"cos(delta, true) = {cs.mean():+.3f} ± {cs.std():.3f}")

    print(f"\n=== {args.poses} ===")
    print(f"  tag: {z['tag']}")
    report("all views", rows)
    report("keyframes", [r for r in rows if r["is_kf"]])
    report("non-keyframes", [r for r in rows if not r["is_kf"]])

    # Magnitude sanity: is the correction even the right size?
    dn = np.array([r["dnorm"] for r in rows])
    tn = np.array([r["tnorm"] for r in rows])
    print(f"\n  correction magnitude {dn.mean():.4f} m vs true error "
          f"{tn.mean():.4f} m  (ratio {dn.mean()/tn.mean():.2f})")

    # Ground-truth-free proxy, usable online: true drift is low-frequency,
    # sponge noise is high-frequency. Compare the size of the step-to-step
    # change to the size of the corrections themselves.
    d = np.stack([r["delta"] for r in rows])
    if len(d) > 2:
        rough = np.linalg.norm(np.diff(d, axis=0), axis=1).mean()
        print(f"  GT-free smoothness: mean |delta_i - delta_(i-1)| = {rough:.4f} m "
              f"vs mean |delta| = {dn.mean():.4f} m  (ratio {rough/dn.mean():.2f}; "
              f"<<1 = smooth/low-frequency, >1 = high-frequency noise)")

    print("\n  READING: a refiner shrinks the translation error, corrects "
          "non-keyframes more than keyframes, and has cos > 0.\n"
          "           a sponge leaves the error flat or worse, shows no "
          "stratification, and has cos ~ 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

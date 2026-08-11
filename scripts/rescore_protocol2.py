"""Re-score saved maps under the fixed evaluation protocol (p2).

Every psnr/lpips in this project before this was computed against held-out
targets that had NOT been exposure-normalized, while the map was built from
frames that had (§17.40, §17.50). The mismatch is per-sequence -- desk's
exposure gain is 0.957 and 360's is 1.138 -- so it distorted sequences by
different amounts (+0.126 dB and +2.36 dB), and Kimi's round-19 point follows:
**cross-sequence quantities did not cancel**. The seven-sequence pooled means
(+2.21 dB / -16.7%) and the -0.96 correlation are exactly that kind of number.

Re-running SLAM would be expensive. Re-scoring is not: the baked and polished
maps are still on disk, so this is a render + metric pass.

    logs/online_new/    desk, 360, room
    logs/online_pol9/   plant, rpy, teddy, xyz

Emits the p1 (as recorded) and p2 (re-scored) tables side by side, plus the
correlation both ways.
"""
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# The polish runs, verified by grepping each log for "polish phase". An
# earlier version of this file pointed desk/360/room at logs/online_new, which
# are the NON-polished arms of 17.18's regression-era A/B -- scoring them as
# "polished" produced a 1 dB LOSS from the polish, i.e. an artifact of picking
# the wrong file. The four sane sequences next to three absurd ones was the
# tell; the fix was to check which runs actually ran the phase.
RUNS = {
    "desk": "online_polish_new", "360": "online_polish_360",
    "room": "online_polish_room",
    "plant": "online_pol9", "rpy": "online_pol9",
    "teddy": "online_pol9", "xyz": "online_pol9",
}

# As recorded in the skill under the broken protocol, baked -> polished.
P1 = {
    "desk":  (10.7085, 0.5588, 12.8154, 0.4194),
    "360":   (10.4383, 0.5813, 12.6417, 0.4890),
    "room":  (10.4264, 0.5813, 12.6417, 0.4890),
    "plant": (None,) * 4, "rpy": (None,) * 4,
    "teddy": (None,) * 4, "xyz": (None,) * 4,
}

# translation/rotation ratio per sequence (17.34), the correlation's x-axis
RATIO = {"xyz": 15.0, "desk": 3.4, "plant": 2.6, "teddy": 2.2,
         "room": 1.9, "360": 1.1, "rpy": 0.6}


def score(ply, traj, dataset, n=100, gpu="0"):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    out = subprocess.run(
        [PY, os.path.join(REPO, "scripts", "eval_map_quality.py"),
         "--ply", ply, "--traj", traj, "--dataset", dataset, "--n", str(n)],
        capture_output=True, text=True, cwd=REPO, env=env).stdout
    for line in out.splitlines():
        if "map |" in line:
            f = line.split()
            return (float(f[f.index([x for x in f if x.startswith("psnr=")][0])]
                          .split("=")[1]),
                    float([x for x in f if x.startswith("lpips=")][0].split("=")[1]))
    return None, None


def main():
    gpu = sys.argv[1] if len(sys.argv) > 1 else "0"
    seqs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(RUNS)
    rows = {}
    for s in seqs:
        run = RUNS[s]
        full = f"rgbd_dataset_freiburg1_{s}"
        d = os.path.join(REPO, "logs", run)
        traj = os.path.join(d, f"{full}.txt")
        ds = os.path.join(REPO, "datasets", "tum", full)
        baked = os.path.join(d, f"{full}_gaussians.ply")
        ref = os.path.join(d, f"{full}_refined.ply")
        if not (os.path.exists(baked) and os.path.exists(ref)):
            print(f"{s:8s} MISSING artifacts in {run}", flush=True)
            continue
        bp, bl = score(baked, traj, ds, gpu=gpu)
        rp, rl = score(ref, traj, ds, gpu=gpu)
        if bp is None or rp is None:
            print(f"{s:8s} scoring failed", flush=True)
            continue
        rows[s] = (bp, bl, rp, rl)
        print(f"{s:8s} p2  baked {bp:7.4f}/{bl:.4f}  polished {rp:7.4f}/{rl:.4f}"
              f"  d {rp-bp:+.2f} dB / {(rl-bl)/bl*100:+.1f}%", flush=True)
    if len(rows) >= 3:
        from scipy.stats import spearmanr
        seqs_ = sorted(rows)
        dl = [(rows[s][3] - rows[s][1]) / rows[s][1] for s in seqs_]
        dp = [rows[s][2] - rows[s][0] for s in seqs_]
        r = [RATIO[s] for s in seqs_]
        print(f"\npooled  d psnr {np.mean(dp):+.2f} dB   d lpips "
              f"{np.mean(dl)*100:+.1f}%   (n={len(seqs_)})")
        print(f"spearman(ratio, d lpips) = {spearmanr(r, dl).statistic:+.3f}"
              f"   [17.34 reported -0.964 under p1]")
        print(f"spearman(ratio, d psnr)  = {spearmanr(r, dp).statistic:+.3f}"
              f"   [17.34 reported +0.07 under p1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

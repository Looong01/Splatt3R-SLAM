"""Summarize --frame-timing CSVs from the GPU-contention experiment (g).

Prints per-arm latency percentiles for the tracking call and for the whole
main-loop iteration, plus the implied sustainable FPS, against the 33.3 ms
frame budget of TUM's 30 fps stream. Keyframes are also reported separately:
they do extra work (Gaussian head + backend handoff) and are the rows where
contention would show first.

Usage: python3 scripts/summarize_frame_timing.py logs/contention/desk_base.csv [more.csv ...]
"""
import sys

import numpy as np


def summarize(path):
    rows = []
    with open(path) as f:
        header = f.readline()
        assert header.startswith("frame,"), f"{path}: unexpected header {header!r}"
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 6:
                continue
            rows.append((parts[2], float(parts[3]), float(parts[4]), float(parts[5])))
    if not rows:
        print(f"{path}: no rows")
        return
    modes = np.array([r[0] for r in rows])
    track = np.array([r[1] for r in rows])
    wait = np.array([r[2] for r in rows])
    itr = np.array([r[3] for r in rows])

    def pct(x):
        return "  ".join(f"p{q} {np.percentile(x, q):7.2f}" for q in (50, 95, 99))

    print(f"\n== {path}  ({len(rows)} frames, modes: "
          + ", ".join(f"{m}={int((modes == m).sum())}" for m in sorted(set(modes))))
    print(f"  track_ms   {pct(track)}   max {track.max():8.2f}")
    print(f"  iter_ms    {pct(itr)}   max {itr.max():8.2f}   mean {itr.mean():.2f}")
    if wait.sum() > 0:
        print(f"  wait_ms    {pct(wait)}   max {wait.max():8.2f}")
    over = (itr > 33.3).mean()
    print(f"  iter_ms > 33.3 ms (30 fps budget): {over * 100:.1f}% of frames "
          f"-> sustainable ~{1000.0 / itr.mean():.1f} fps mean, "
          f"~{1000.0 / np.percentile(itr, 95):.1f} fps at p95")


for p in sys.argv[1:]:
    summarize(p)

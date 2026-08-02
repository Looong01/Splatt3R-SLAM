#!/usr/bin/env bash
# Sequential GPU1 work queue, so a long chain of jobs can run unattended
# alongside the TUM training run on GPU0.
#
# Order matters: the batch scans build a training loader, which needs that
# family's coverage cache, so coverage must finish first. 7-scenes already has
# its cache, so its scan could run at any point; it is placed last only to keep
# the scans together in the log.
#
# Each step logs to its own file and the queue continues past a failure --
# a missing EuRoC/ETH3D coverage should not cost the 7-scenes scan.
set -u

cd "$(dirname "$0")/.." || exit 1
PY=/home/share-v5/miniconda3/envs/splatt3r-slam/bin/python
export CUDA_VISIBLE_DEVICES=1

run() {
    local log="logs/$1"; shift
    echo "=== $(date '+%H:%M:%S')  $*  -> $log"
    "$@" > "$log" 2>&1 \
        && echo "    ok  $(date '+%H:%M:%S')" \
        || echo "    FAILED (exit $?)  $(date '+%H:%M:%S')  -- see $log"
}

run precompute_coverage_euroc.log  "$PY" scripts/precompute_coverage.py euroc
run precompute_coverage_eth3d.log  "$PY" scripts/precompute_coverage.py eth3d

run batch_scan_7scenes.log "$PY" scripts/exp_batch_scan.py 7-scenes
run batch_scan_euroc.log   "$PY" scripts/exp_batch_scan.py euroc
run batch_scan_eth3d.log   "$PY" scripts/exp_batch_scan.py eth3d

echo "=== $(date '+%H:%M:%S')  queue done"

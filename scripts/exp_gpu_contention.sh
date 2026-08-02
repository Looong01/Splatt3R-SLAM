#!/bin/bash
# Experiment (g) — GPU contention: does a concurrent refiner-style load starve
# the tracker?
#
# The feared failure mode of the online refiner is not that it is slow, but
# that its GPU work preempts tracking -> per-frame latency grows past the
# frame budget (33.3 ms at TUM's 30 fps) -> a live system drops frames and ATE
# regresses. Offline the dataset feeds frames as fast as they are consumed, so
# the contention signature is measured as PER-FRAME LATENCY (--frame-timing
# CSV), plus ATE as a sanity check.
#
# Arms (all SLAM runs: desk, config/rt_calib.yaml = multithreaded, calib,
# subsample 2, --no-viz):
#   baseline    SLAM alone on GPU 1 (run with GPU 0 idle for a clean reference)
#   contended   SLAM + refiner-style optimization load, both on GPU 1
#   crossgpu    SLAM on GPU 1 + the same load on GPU 0 (the two-GPU deployment
#               answer; costs cross-process buffer copies in a real system)
#
# The load is this project's real refinement workload (scripts/refine_causal.py
# --mode posthoc with an effectively infinite iteration cap): full-map render +
# backward on the desk kfgauss dump, i.e. exactly what the online refiner's
# steady state would compute.
#
# Usage:  bash scripts/exp_gpu_contention.sh [baseline|contended|crossgpu|ate]
set -uo pipefail
cd "$(dirname "$0")/.."

SEQ=datasets/tum/rgbd_dataset_freiburg1_desk
GT=$SEQ/groundtruth.txt
KFGAUSS=logs/frames_head/rgbd_dataset_freiburg1_desk_kfgauss.pt
TRAJ=logs/frames_head/rgbd_dataset_freiburg1_desk.txt
FRAMES=logs/frames_head/rgbd_dataset_freiburg1_desk_frames.txt
OUT=logs/contention
mkdir -p "$OUT"

run_slam () {  # $1=tag
  CUDA_VISIBLE_DEVICES=1 python main.py --dataset "$SEQ" --config config/rt_calib.yaml \
    --no-viz --no-render-gaussians --save-as "contention_$1" \
    --frame-timing "$OUT/desk_$1.csv"
}

start_load () {  # $1=gpu id; echoes pid
  CUDA_VISIBLE_DEVICES=$1 python scripts/refine_causal.py --mode posthoc \
    --kfgauss "$KFGAUSS" --traj "$TRAJ" --frames-traj "$FRAMES" --dataset "$SEQ" \
    --iters 100000000 --eval-every 100000000 --tag "gpu-load$1" \
    > "$OUT/load_gpu$1.log" 2>&1 &
  echo $!
}

wait_load_steady () {  # $1=gpu id
  for _ in $(seq 1 90); do
    grep -q 'iter     0' "$OUT/load_gpu$1.log" 2>/dev/null && return 0
    sleep 2
  done
  echo "WARNING: load did not reach steady state in 180s" >&2
}

arm=${1:?usage: exp_gpu_contention.sh [baseline|contended|crossgpu|ate]}
case "$arm" in
  baseline)
    run_slam base 2>&1 | tee "$OUT/desk_base.log" | tail -5
    ;;
  contended)
    LOAD_PID=$(start_load 1)
    wait_load_steady 1
    run_slam load 2>&1 | tee "$OUT/desk_load.log" | tail -5
    kill "$LOAD_PID" 2>/dev/null; wait "$LOAD_PID" 2>/dev/null
    ;;
  crossgpu)
    LOAD_PID=$(start_load 0)
    wait_load_steady 0
    run_slam xgpu 2>&1 | tee "$OUT/desk_xgpu.log" | tail -5
    kill "$LOAD_PID" 2>/dev/null; wait "$LOAD_PID" 2>/dev/null
    ;;
  ate)
    for a in base load xgpu; do
      est="logs/contention_$a/rgbd_dataset_freiburg1_desk.txt"
      [ -f "$est" ] || continue
      rmse=$(evo_ape tum "$GT" "$est" -as 2>/dev/null | awk '/rmse/{print $2}')
      echo "$a ATE rmse = $rmse   ($est)"
    done
    ;;
esac

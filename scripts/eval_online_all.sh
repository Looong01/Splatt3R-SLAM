#!/bin/bash
# Online deliverable evaluation across all four dataset families (GUI mode).
#
# Per sequence, two arms:
#   head    -- per-family production head (checkpoints/head_only_long/<fam>/)
#   refiner -- same + online map refinement on the second GPU (the deliverable)
#
# Metrics collected per arm: ATE (evo_ape tum -as), map quality baked+refined
# (eval_map_quality.py), per-frame latency (--frame-timing CSV), VRAM peak
# (nvidia-smi sampler). The viz window opens on the physical display
# (DISPLAY :0); the run is terminated after "done" + grace, since GUI mode
# otherwise blocks on viz.join().
#
# Usage: bash scripts/eval_online_all.sh <tum|7-scenes|euroc|eth3d> <seq_dir_name>
set -uo pipefail
cd "$(dirname "$0")/.."

FAM=$1
SEQ=$2
case $FAM in
  tum)      DS=datasets/tum/$SEQ;            GT=$DS/groundtruth.txt ;;
  7-scenes) DS=datasets/7-scenes/$SEQ;       GT=groundtruths/7-scenes/$SEQ.txt ;;
  euroc)    DS=datasets/euroc/$SEQ;          GT=groundtruths/euroc/$SEQ.txt ;;
  eth3d)    DS=datasets/eth3d/train/$SEQ;    GT=$DS/groundtruth.txt ;;
  *) echo "unknown family $FAM" >&2; exit 1 ;;
esac
HEAD=checkpoints/head_only_long/$FAM/head_best.pt
OUT=logs/online_eval
SEQNAME=$(basename "$DS")
export DISPLAY=:0 XAUTHORITY=$(ls /run/user/1000/.mutter-Xwaylandauth.* | head -1)
mkdir -p "$OUT"

run_arm () {  # $1=arm tag, $2=extra args
  local arm=$1 extra=$2
  local log=$OUT/${FAM}_${SEQNAME}_${arm}.log
  local csv=$OUT/${FAM}_${SEQNAME}_${arm}.csv
  local save=online_eval/${FAM}/${SEQNAME}/${arm}
  echo "=== [$FAM/$SEQNAME/$arm] $(date +%H:%M:%S) ==="
  # VRAM sampler
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 2 > "$OUT/${FAM}_${SEQNAME}_${arm}.vram" 2>/dev/null &
  local samp=$!
  CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset "$DS" --config config/eval_calib.yaml \
      --head "$HEAD" --frame-timing "$csv" --save-as "$save" \
      $extra > "$log" 2>&1 &
  local pid=$!
  # wait for "done" (saves complete) then grace, then kill the process group
  for _ in $(seq 1 720); do
    grep -q '^done$' "$log" 2>/dev/null && break
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  sleep 8
  kill -TERM -$pid 2>/dev/null; sleep 3; kill -KILL -$pid 2>/dev/null
  wait $pid 2>/dev/null
  kill $samp 2>/dev/null; wait $samp 2>/dev/null
  # ATE
  local traj=logs/$save/${SEQNAME}.txt
  if [ -f "$traj" ]; then
    local rmse
    rmse=$(evo_ape tum "$GT" "$traj" -as 2>/dev/null | awk '/rmse/{print $2}')
    echo "ATE $FAM/$SEQNAME/$arm rmse = $rmse"
  else
    echo "ATE $FAM/$SEQNAME/$arm: NO TRAJECTORY (run failed?)" >&2
  fi
}

run_arm head ""
run_arm refiner "--refiner --refiner-gpu 1 --refiner-duty 1.0"

# Map quality: baked vs refined (refined arm only has _refined.ply)
for ply in logs/online_eval/$FAM/$SEQNAME/refiner/${SEQNAME}_gaussians.ply \
           logs/online_eval/$FAM/$SEQNAME/refiner/${SEQNAME}_refined.ply; do
  [ -f "$ply" ] || continue
  kind=$(basename "$ply" | sed "s/${SEQNAME}_//;s/\.ply//")
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_map_quality.py --ply "$ply" \
    --traj logs/online_eval/$FAM/$SEQNAME/refiner/${SEQNAME}.txt \
    --dataset "$DS" --n 100 > "$OUT/${FAM}_${SEQNAME}_${kind}.score" 2>&1 || true
  echo "MAPQUALITY $FAM/$SEQNAME/$kind: $(grep 'map |' "$OUT/${FAM}_${SEQNAME}_${kind}.score" | tail -1)"
done
echo "=== [$FAM/$SEQNAME] complete $(date +%H:%M:%S) ==="

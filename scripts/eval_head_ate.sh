#!/usr/bin/env bash
# SLAM-level ATE validation of the head-only fine-tuned Gaussian head.
#
# Render metrics (psnr/lpips on held-out novel views) improved +1.77 dB on
# TUM after 40 epochs of head-only training (see the splatt3r-finetuning-
# experiments skill), but the metric this project actually optimizes for is
# SLAM trajectory accuracy. This script runs the decisive comparison:
#
#   arm base : main.py on the untouched base checkpoint
#   arm head : main.py --head checkpoints/head_only_long/tum/head_best.pt
#              (the 40-epoch TUM head-only best; +1.77 dB / -10.2% lpips
#               in render metrics)
#
# Same protocol as scripts/eval_retrieval_ab.sh: eval_calib (single_thread,
# deterministic), --no-viz, evo_ape -as against the dataset groundtruth.
# Pre-registered verdict: head-only is adopted for SLAM use only if ATE
# improves (or at worst ties) on the loop-sensitive sequences without
# regressing on the control.
#
# Usage:  bash scripts/eval_head_ate.sh [base|head|all]
set -e
cd "$(dirname "$0")/.."

HEAD_CKPT=checkpoints/head_only_long/tum/head_best.pt
SEQS=(rgbd_dataset_freiburg1_room rgbd_dataset_freiburg1_360 rgbd_dataset_freiburg1_desk)
ARM=${1:-all}

run_one() {
    local seq=$1 arm=$2
    local extra=()
    if [ "$arm" = head ]; then
        extra=(--head "$HEAD_CKPT")
    fi
    echo "=== [arm $arm] $seq $(date '+%H:%M:%S') ==="
    python3 main.py \
        --dataset "datasets/tum/$seq" \
        --config config/eval_calib.yaml \
        --no-viz \
        --save-as "head_ate_$arm" \
        "${extra[@]}"
}

case "$ARM" in
    base|head)
        for seq in "${SEQS[@]}"; do run_one "$seq" "$ARM"; done
        ;;
    all)
        for arm in base head; do
            for seq in "${SEQS[@]}"; do run_one "$seq" "$arm"; done
        done
        ;;
    *) echo "usage: $0 [base|head|all]"; exit 1 ;;
esac

echo ""
echo "=== ATE RMSE summary (evo_ape tum, -as alignment) ==="
printf "%-35s %14s %14s\n" "sequence" "base" "head-only"
for seq in "${SEQS[@]}"; do
    gt="datasets/tum/$seq/groundtruth.txt"
    row="$seq"
    for arm in base head; do
        est="logs/head_ate_$arm/$seq.txt"
        if [ -f "$est" ]; then
            rmse=$(evo_ape tum "$gt" "$est" -as 2>/dev/null | awk '/rmse/{print $2}') || rmse=""
            rmse=${rmse:-ERR}
        else
            rmse=N/A
        fi
        row="$row $rmse"
    done
    printf "%-35s %14s %14s\n" $row
done

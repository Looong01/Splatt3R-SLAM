#!/bin/bash
# Stage-0 baseline ablation for the loop-closure retrieval rework:
#   arm a: existing retrieval assets (loop closure ON) + retrieval-feature
#          dump to logs/retrieval_features/<seq>/ (fitting corpus for the
#          whitening/codebook refit)
#   arm b: --no-loop-closure (retrieval loop-closure edges disabled in the
#          backend pose graph; relocalization is unaffected)
#
# Usage: ./scripts/eval_retrieval_ab.sh [a|b|all]   (default: all)
set -e

dataset_path="datasets/tum/"
sequences=(
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_360   # loop-closure sensitive
    rgbd_dataset_freiburg1_desk  # control
)
config="config/eval_calib.yaml"   # single_thread, calibrated, matches eval_tum.sh

arm="${1:-all}"
case "$arm" in
    a|b|all) ;;
    *) echo "Usage: $0 [a|b|all]"; exit 1 ;;
esac

run_arm_a() {
    for seq in "${sequences[@]}"; do
        echo "=== [arm a] $seq (loop closure ON, dumping retrieval features) ==="
        python main.py --dataset "$dataset_path$seq"/ --config "$config" \
            --no-viz --save-as retrieval_ab_a \
            --dump-retrieval-features logs/retrieval_features
    done
}

run_arm_b() {
    for seq in "${sequences[@]}"; do
        echo "=== [arm b] $seq (loop closure OFF, reloc kept) ==="
        python main.py --dataset "$dataset_path$seq"/ --config "$config" \
            --no-viz --save-as retrieval_ab_b \
            --no-loop-closure
    done
}

case "$arm" in
    a)   run_arm_a ;;
    b)   run_arm_b ;;
    all) run_arm_a; run_arm_b ;;
esac

# --- Evaluation ---------------------------------------------------------
# prepare_savedir() (evaluate.py:14-20) writes trajectories to
# logs/<save-as>/<seq>.txt, i.e. logs/retrieval_ab_<arm>/<seq>.txt here.
# evo_ape invocation matches scripts/eval_tum.sh.
echo
echo "=== ATE RMSE summary (evo_ape tum, -as alignment) ==="
printf "%-35s %14s %14s\n" "sequence" "a (with LC)" "b (no LC)"
for seq in "${sequences[@]}"; do
    gt="$dataset_path$seq/groundtruth.txt"
    rmse_a="N/A"
    rmse_b="N/A"
    for a in a b; do
        est="logs/retrieval_ab_$a/$seq.txt"
        if [ -f "$est" ]; then
            rmse=$(evo_ape tum "$gt" "$est" -as 2>/dev/null | awk '/rmse/{print $2}') || rmse=""
            rmse=${rmse:-ERR}
            if [ "$a" = a ]; then rmse_a="$rmse"; else rmse_b="$rmse"; fi
        fi
    done
    printf "%-35s %14s %14s\n" "$seq" "$rmse_a" "$rmse_b"
done

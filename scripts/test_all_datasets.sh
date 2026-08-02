#!/bin/bash
# Test all benchmark datasets (TUM, ETH3D, EuRoC, 7-Scenes) with GUI visualization.
#
# Usage:
#   bash scripts/test_all_datasets.sh                  # run all suites with GUI
#   bash scripts/test_all_datasets.sh --no-viz         # run all suites headless (no GUI)
#   bash scripts/test_all_datasets.sh --tum --euroc    # only selected suites
#   bash scripts/test_all_datasets.sh --no-calib       # without calibration (tum/euroc/7-scenes)
#   bash scripts/test_all_datasets.sh --print          # only re-print ATE from existing logs
#
# In GUI mode each run is killed 20s after it prints "done", because the
# visualization window otherwise keeps the process alive until closed manually.
# Missing dataset directories are skipped with a warning.

no_calib=false
print_only=false
no_viz=false
suites=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-calib)
            no_calib=true
            ;;
        --print)
            print_only=true
            ;;
        --no-viz)
            no_viz=true
            ;;
        --tum|--eth3d|--euroc|--7-scenes)
            suites+=("${1#--}")
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# No suite filter given -> run all
if [ ${#suites[@]} -eq 0 ]; then
    suites=(tum eth3d euroc 7-scenes)
fi

calib_dir=calib
if [ "$no_calib" = true ]; then
    calib_dir=no_calib
fi

run_suite() {
    local suite="$1"; shift
    local dataset_root="$1"; shift
    local datasets=("$@")

    # Config and groundtruth layout per suite
    local config gt_pattern
    case "$suite" in
        tum)
            if [ "$no_calib" = true ]; then config=config/eval_no_calib.yaml; else config=config/eval_calib.yaml; fi
            gt_pattern="DATASET_DIR/groundtruth.txt"
            ;;
        eth3d)
            config=config/eth3d.yaml
            gt_pattern="DATASET_DIR/groundtruth.txt"
            ;;
        euroc)
            if [ "$no_calib" = true ]; then config=config/eval_no_calib.yaml; else config=config/eval_calib.yaml; fi
            gt_pattern="groundtruths/euroc/NAME.txt"
            ;;
        7-scenes)
            if [ "$no_calib" = true ]; then config=config/eval_no_calib.yaml; else config=config/eval_calib.yaml; fi
            gt_pattern="groundtruths/7-scenes/NAME.txt"
            ;;
    esac

    # eth3d saves directly under logs/eth3d/<name>, others under logs/<suite>/<calib>/<name>
    local save_base="$suite/$calib_dir"
    if [ "$suite" = eth3d ]; then
        save_base="$suite"
    fi

    for name in "${datasets[@]}"; do
        local dataset_dir="$dataset_root$name/"
        local gt="${gt_pattern/DATASET_DIR/$dataset_dir}"
        gt="${gt/NAME/$name}"
        local log_file="logs/$save_base/$name/$name.txt"

        echo "================================================================"
        echo "[$suite] $name"
        echo "================================================================"

        if [ "$print_only" = false ]; then
            if [ ! -d "$dataset_dir" ]; then
                echo "[SKIP] dataset not found: $dataset_dir"
                continue
            fi
            if [ "$no_viz" = true ]; then
                python -u main.py --dataset "$dataset_dir" --save-as "$save_base/$name" --config "$config" --no-viz
            else
                # GUI 模式：序列跑完（日志出现 "done"）后等 20 秒再杀掉整个进程组，
                # 否则可视化窗口会一直挂着，需要手动关窗才能继续下一条。
                run_log="logs/$save_base/$name/run.log"
                mkdir -p "logs/$save_base/$name"
                setsid python -u main.py --dataset "$dataset_dir" --save-as "$save_base/$name" --config "$config" >"$run_log" 2>&1 &
                run_pid=$!
                # 实时显示日志；inotify 用满时退回轮询，失败也不影响评测
                tail ---disable-inotify -f "$run_log" 2>/dev/null &
                tail_pid=$!
                while kill -0 "$run_pid" 2>/dev/null; do
                    if grep -q "^done$" "$run_log" 2>/dev/null; then
                        sleep 20
                        kill -- -"$run_pid" 2>/dev/null
                        break
                    fi
                    sleep 2
                done
                wait "$run_pid" 2>/dev/null
                kill "$tail_pid" 2>/dev/null
            fi
        fi

        if [ ! -f "$log_file" ]; then
            echo "[SKIP] no trajectory log: $log_file"
            continue
        fi
        if [ ! -f "$gt" ]; then
            echo "[SKIP] no groundtruth: $gt"
            continue
        fi
        evo_ape tum "$gt" "$log_file" -as
    done
}

for suite in "${suites[@]}"; do
    case "$suite" in
        tum)
            run_suite tum datasets/tum/ \
                rgbd_dataset_freiburg1_360 \
                rgbd_dataset_freiburg1_desk \
                rgbd_dataset_freiburg1_desk2 \
                rgbd_dataset_freiburg1_floor \
                rgbd_dataset_freiburg1_plant \
                rgbd_dataset_freiburg1_room \
                rgbd_dataset_freiburg1_rpy \
                rgbd_dataset_freiburg1_teddy \
                rgbd_dataset_freiburg1_xyz
            ;;
        eth3d)
            run_suite eth3d datasets/eth3d/train/ \
                plant_1 plant_2 plant_3 plant_4 plant_5 \
                cables_1 cables_2 cables_3 \
                camera_shake_1 camera_shake_2 camera_shake_3 \
                ceiling_1 ceiling_2 desk_3 desk_changing_1 \
                einstein_1 einstein_2 einstein_flashlight \
                einstein_global_light_changes_1 einstein_global_light_changes_2 einstein_global_light_changes_3 \
                kidnap_1 large_loop_1 \
                mannequin_1 mannequin_3 mannequin_4 mannequin_5 mannequin_7 \
                mannequin_face_1 mannequin_face_2 mannequin_face_3 mannequin_head \
                motion_1 planar_2 planar_3 \
                plant_scene_1 plant_scene_2 plant_scene_3 \
                reflective_1 repetitive \
                sfm_bench sfm_garden sfm_house_loop sfm_lab_room_1 sfm_lab_room_2 \
                sofa_1 sofa_2 sofa_3 sofa_4 sofa_shake \
                table_3 table_4 table_7 \
                vicon_light_1 vicon_light_2
            ;;
        euroc)
            run_suite euroc datasets/euroc/ \
                MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
                V1_01_easy V1_02_medium V1_03_difficult \
                V2_01_easy V2_02_medium V2_03_difficult
            ;;
        7-scenes)
            run_suite 7-scenes datasets/7-scenes/ \
                chess fire heads office pumpkin redkitchen stairs
            ;;
    esac
done

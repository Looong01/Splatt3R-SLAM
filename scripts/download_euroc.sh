#!/bin/bash
# Download the EuRoC MAV dataset from the ETH Research Collection:
#   https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f
#
# The collection only offers three large archives (machine_hall / vicon_room1 /
# vicon_room2, ~23 GB total), each nesting the per-sequence zips we need:
#   machine_hall/MH_01_easy/MH_01_easy.zip  ->  mav0/...
# This script downloads each archive (resumable), extracts only the nested
# sequence zips, unpacks them to datasets/euroc/<SEQ>/, then removes the
# large archives to save disk.
set -u

dest="datasets/euroc"
tmp="$dest/.collection"
mkdir -p "$dest" "$tmp"

# Research Collection bitstream direct links
declare -A archives=(
    ["machine_hall"]="https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content"
    ["vicon_room1"]="https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content"
    ["vicon_room2"]="https://www.research-collection.ethz.ch/server/api/core/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content"
)

# Sequences needed by scripts/eval_euroc.sh, grouped by archive
declare -A wanted=(
    ["machine_hall"]="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult"
    ["vicon_room1"]="V1_01_easy V1_02_medium V1_03_difficult"
    ["vicon_room2"]="V2_01_easy V2_02_medium V2_03_difficult"
)

for group in "${!archives[@]}"; do
    big_zip="$tmp/${group}.zip"

    need_download=false
    for seq in ${wanted[$group]}; do
        [ -f "$dest/$seq/mav0/cam0/data.csv" ] || need_download=true
    done

    if [ "$need_download" = true ]; then
        echo "Downloading ${group}.zip (~several GB, resumable) ..."
        if ! wget -c "${archives[$group]}" -O "$big_zip"; then
            echo "[FAIL] download failed: $group (skipping, re-run to resume)"
            continue
        fi
        # Extract only the nested per-sequence zips (skip the large .bag files)
        echo "Extracting nested sequence zips from ${group}.zip ..."
        unzip -o -j "$big_zip" "$group/*/*.zip" -d "$tmp"
        rm -f "$big_zip"
    else
        echo "[SKIP] $group sequences already extracted"
    fi

    for seq in ${wanted[$group]}; do
        if [ -f "$dest/$seq/mav0/cam0/data.csv" ]; then
            echo "[SKIP] $seq already extracted"
            continue
        fi
        if [ ! -f "$tmp/$seq.zip" ]; then
            echo "[FAIL] nested zip missing for $seq"
            continue
        fi
        echo "Unzipping $seq ..."
        mkdir -p "$dest/$seq"
        if unzip -o "$tmp/$seq.zip" -d "$dest/$seq"; then
            rm -f "$tmp/$seq.zip"
        else
            echo "[FAIL] unzip failed: $seq"
        fi
    done
done

rmdir "$tmp" 2>/dev/null
echo "Done. Sequences are in $dest"

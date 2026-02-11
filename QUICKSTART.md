# Quick Start Guide for Splatt3R-SLAM

This guide helps you get Splatt3R-SLAM running quickly.

## Prerequisites

- Ubuntu 20.04+ (or WSL2 on Windows)
- NVIDIA GPU with CUDA 11.8+
- Conda/Miniconda
- Git

## Installation (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/Looong01/Splatt3R-SLAM.git --recursive
cd Splatt3R-SLAM
```

### 2. Create Environment
```bash
conda create -n splatt3r-slam python=3.11
conda activate splatt3r-slam
```

### 3. Install PyTorch
Choose based on your CUDA version:

```bash
# For CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# For CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 4. Install Dependencies
```bash
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
pip install lightning lpips omegaconf huggingface_hub gitpython
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

## Quick Test (2 minutes)

### Download Test Dataset
```bash
bash ./scripts/download_tum.sh
```

### Run Splatt3R-SLAM
```bash
python main_splatt3r.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml
```

On first run, it will automatically download the Splatt3R checkpoint (~150MB) from HuggingFace.

## What to Expect

You should see:
1. **Terminal Output**: FPS, relocalization status, optimization progress
2. **Visualization Window**: Real-time 3D reconstruction and camera trajectory
3. **Results**: Saved trajectory and reconstruction in `logs/` directory

## Common Issues & Solutions

### Issue: "No module named 'torch'"
**Solution**: Make sure you activated the conda environment:
```bash
conda activate splatt3r-slam
```

### Issue: "CUDA out of memory"
**Solution**: Reduce image resolution in config:
```yaml
dataset:
  img_downsample: 2  # Increase this value (1, 2, 4, ...)
```

### Issue: "Failed to download checkpoint"
**Solution**: Download manually:
```bash
mkdir -p checkpoints
# Download from: https://huggingface.co/brandonsmart/splatt3r_v1.0/blob/main/epoch%3D19-step%3D1200.ckpt
# Save as: checkpoints/splatt3r_v1.0.ckpt

python main_splatt3r.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --checkpoint checkpoints/splatt3r_v1.0.ckpt
```

### Issue: Visualization not showing
**Solution**: Run in headless mode:
```bash
python main_splatt3r.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --no-viz
```

## Next Steps

### Try Other Datasets
```bash
# Download more datasets
bash ./scripts/download_7_scenes.sh
bash ./scripts/download_euroc.sh

# Run on different sequences
python main_splatt3r.py --dataset datasets/7-scenes/chess --config config/base.yaml
```

### Run on Your Own Video
```bash
python main_splatt3r.py \
    --dataset path/to/your/video.mp4 \
    --config config/base.yaml
```

### Run with Camera Calibration
```bash
python main_splatt3r.py \
    --dataset path/to/your/data \
    --config config/base.yaml \
    --calib config/intrinsics.yaml
```

### Live Demo with RealSense
```bash
python main_splatt3r.py \
    --dataset realsense \
    --config config/base.yaml
```

## Performance Tips

### For Better Speed
1. Use lower resolution: Set `img_downsample: 2` or higher
2. Reduce retrieval candidates: Set `k: 3` (default: 5)
3. Use single thread mode for debugging: `single_thread: true`

### For Better Quality
1. Use higher resolution: Set `img_downsample: 1`
2. Increase loop closure candidates: Set `k: 10`
3. Adjust confidence thresholds in config

## Understanding Output

### Terminal Messages
- `FPS: X.XX` - Processing frames per second
- `RELOCALIZING against kf N` - Attempting loop closure
- `Success! Relocalized` - Loop closure successful

### Output Files
- `logs/*/trajectory.txt` - Camera trajectory (TUM format)
- `logs/*/reconstruction.ply` - 3D point cloud reconstruction
- `logs/*/keyframes/` - Saved keyframe images and poses

## Comparing with Original MASt3R-SLAM

Run the original MASt3R-SLAM for comparison:
```bash
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml
```

Key differences:
- **Splatt3R-SLAM** (`main_splatt3r.py`): Uses Gaussian Splatting, better for novel view synthesis
- **MASt3R-SLAM** (`main.py`): Uses point-based reconstruction, may be faster

## Getting Help

- See `INTEGRATION.md` for detailed architecture
- Check GitHub Issues: https://github.com/Looong01/Splatt3R-SLAM/issues
- Refer to original Splatt3R: https://splatt3r.active.vision
- Refer to MASt3R-SLAM: https://edexheim.github.io/mast3r-slam/

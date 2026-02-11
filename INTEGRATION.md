# Splatt3R Integration Guide

## Overview

This repository integrates [Splatt3R](https://splatt3r.active.vision) with SLAM capabilities, combining zero-shot 3D Gaussian Splatting with real-time dense SLAM.

## Architecture

### Directory Structure

```
Splatt3R-SLAM/
├── splatt3r_core/           # Core Splatt3R implementation (deep integration)
│   ├── src/                 # Splatt3R source code
│   │   ├── mast3r_src/     # MASt3R model with Gaussian head
│   │   └── pixelsplat_src/  # PixelSplat decoder for rendering
│   ├── utils/               # Splatt3R utilities
│   ├── main.py             # Original Splatt3R training script
│   ├── demo.py             # Original Splatt3R demo
│   └── splatt3r_model.py   # Model loading interface
│
├── splatt3r_slam/           # SLAM implementation with Splatt3R
│   ├── splatt3r_utils.py   # Splatt3R inference utilities
│   ├── tracker.py           # Visual tracking
│   ├── global_opt.py        # Global optimization/bundle adjustment
│   └── ...                  # Other SLAM components
│
├── mast3r_slam/             # Original MASt3R-SLAM (for comparison)
│   └── ...
│
├── main_splatt3r.py         # Main entry point for Splatt3R-SLAM
└── main.py                  # Original MASt3R-SLAM entry point
```

## Key Components

### 1. Splatt3R Model (`splatt3r_core/`)

The Splatt3R model is a modified MASt3R architecture that outputs:
- **3D Points** (`pts3d`): Initial 3D point estimates
- **Confidence** (`conf`): Confidence scores for predictions
- **Descriptors** (`desc`): Feature descriptors for matching
- **Gaussian Parameters**:
  - `means`: 3D Gaussian centers (optionally with offsets)
  - `scales`: Gaussian scales (3 per Gaussian)
  - `rotations`: Gaussian rotations (quaternions)
  - `sh`: Spherical harmonics for view-dependent appearance
  - `opacities`: Opacity/alpha values

### 2. Splatt3R Utilities (`splatt3r_slam/splatt3r_utils.py`)

Key functions:
- `load_splatt3r()`: Loads model from HuggingFace or local checkpoint
- `splatt3r_inference_mono()`: Monocular 3D prediction
- `splatt3r_match_asymmetric()`: Asymmetric stereo matching
- `splatt3r_match_symmetric()`: Symmetric stereo matching

### 3. Main SLAM Loop (`main_splatt3r.py`)

The SLAM system runs in three modes:
1. **INIT**: Initialize with first frame
2. **TRACKING**: Track camera pose against last keyframe
3. **RELOC**: Relocalize if tracking fails

## Model Loading

### Automatic Download (Recommended)

```python
python main_splatt3r.py --dataset <dataset_path> --config config/base.yaml
```

The system automatically downloads the Splatt3R checkpoint from HuggingFace:
- Model: `brandonsmart/splatt3r_v1.0`
- Checkpoint: `epoch=19-step=1200.ckpt`

### Manual Checkpoint Path

```python
python main_splatt3r.py --dataset <dataset_path> --config config/base.yaml --checkpoint path/to/checkpoint.ckpt
```

## Differences from MASt3R-SLAM

### Model Architecture
- **MASt3R**: Outputs 3D points + descriptors
- **Splatt3R**: Outputs 3D points + descriptors + Gaussian parameters

### Gaussian Splatting
Splatt3R enables:
- Direct prediction of 3D Gaussians from image pairs
- View-dependent appearance via spherical harmonics
- Better scene representation for novel view synthesis

### Inference
- Both use similar symmetric/asymmetric matching
- Splatt3R decoder includes Gaussian parameter prediction
- Additional postprocessing for Gaussian parameters

## Configuration

Use the same configuration files as MASt3R-SLAM:
- `config/base.yaml`: Base configuration
- `config/calib.yaml`: With camera calibration
- `config/intrinsics.yaml`: Custom intrinsics

## Usage Examples

### Basic Usage
```bash
python main_splatt3r.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml
```

### With Calibration
```bash
python main_splatt3r.py --dataset datasets/tum/rgbd_dataset_freiburg1_room --config config/calib.yaml
```

### With Custom Intrinsics
```bash
python main_splatt3r.py --dataset path/to/video.mp4 --config config/base.yaml --calib config/intrinsics.yaml
```

### Live Camera
```bash
python main_splatt3r.py --dataset realsense --config config/base.yaml
```

## Technical Details

### Checkpoint Format
The Splatt3R checkpoint is a PyTorch Lightning checkpoint containing:
- `model.encoder`: MASt3R encoder with Gaussian head
- `model.decoder`: PixelSplat decoder for rendering
- Training hyperparameters and optimizer state

### Integration Points
1. **Model Loading**: `splatt3r_utils.load_splatt3r()`
2. **Monocular Init**: `splatt3r_inference_mono()`
3. **Tracking**: `splatt3r_match_asymmetric()`
4. **Loop Closure**: `splatt3r_match_symmetric()`

### Memory Sharing
The model uses PyTorch's `share_memory()` for multiprocess inference:
- Main process: Frontend tracking
- Backend process: Global optimization

## Troubleshooting

### Import Errors
Ensure all paths are set correctly in `splatt3r_core/__init__.py`:
```python
sys.path.insert(0, os.path.join(_current_dir, 'src', 'mast3r_src'))
sys.path.insert(0, os.path.join(_current_dir, 'src', 'mast3r_src', 'dust3r'))
sys.path.insert(0, os.path.join(_current_dir, 'src', 'pixelsplat_src'))
```

### CUDA Errors
- Check CUDA compatibility with PyTorch version
- Ensure GPU memory is sufficient
- Try reducing batch size in config

### Download Issues
If HuggingFace download fails:
1. Download manually from https://huggingface.co/brandonsmart/splatt3r_v1.0
2. Place at `checkpoints/epoch=19-step=1200.ckpt`
3. Use `--checkpoint checkpoints/epoch=19-step=1200.ckpt`

## Development

### Adding Custom Features
1. Extend `splatt3r_utils.py` for new inference modes
2. Modify `tracker.py` for tracking improvements
3. Update `global_opt.py` for optimization changes

### Testing
```bash
# Syntax check
python -m py_compile main_splatt3r.py splatt3r_slam/splatt3r_utils.py

# Run on test dataset
python main_splatt3r.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml --no-viz
```

## References

- [Splatt3R Paper](https://arxiv.org/abs/2408.13912): Smart et al., "Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs"
- [MASt3R-SLAM Paper](https://arxiv.org/abs/2412.12392): Murai et al., "MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors"
- [Splatt3R GitHub](https://github.com/btsmart/splatt3r)
- [MASt3R GitHub](https://github.com/naver/mast3r)

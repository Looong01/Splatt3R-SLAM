<p align="center">
  <h1 align="center">Splatt3R-SLAM: Real-Time Dense SLAM with 3D Gaussian Splatting</h1>
  <p align="center">
    Built on top of <a href="https://edexheim.github.io/mast3r-slam/">MASt3R-SLAM</a> and integrated with <a href="https://splatt3r.active.vision">Splatt3R</a>
  </p>

  <h3 align="center">
    <a href="https://splatt3r.active.vision">Splatt3R Project</a> | 
    <a href="https://edexheim.github.io/mast3r-slam/">MASt3R-SLAM Project</a>
  </h3>
  <div align="center"></div>

<p align="center">
    <img src="./media/teaser.gif" alt="teaser" width="100%">
</p>
<br>

## Overview

Splatt3R-SLAM integrates [Splatt3R](https://splatt3r.active.vision) (Zero-shot Gaussian Splatting from Uncalibrated Image Pairs) into a real-time SLAM system. This combines the dense 3D reconstruction capabilities of MASt3R-SLAM with Splatt3R's 3D Gaussian Splatting for improved scene representation.

### Key Features
- **3D Gaussian Splatting**: Uses Splatt3R to predict 3D Gaussians directly from image pairs
- **Zero-shot Reconstruction**: No scene-specific training required
- **Real-time Performance**: Maintains real-time SLAM capabilities
- **Dense 3D Reconstruction**: Produces detailed 3D reconstructions with Gaussian splats
- **Per-frame PNG Export**: Saves Gaussian-rendered images for every frame by default

### Differences from MASt3R-SLAM

| Aspect | MASt3R-SLAM | Splatt3R-SLAM |
|--------|-------------|---------------|
| **Model** | MASt3R | MAST3RGaussians (Splatt3R) |
| **Output** | Points + Descriptors | Points + Descriptors + Gaussians |
| **Visualization** | OpenGL point cloud | Interactive Gaussian Splatting |
| **View Synthesis** | Limited | Excellent |
| **Entry Point** | `main.py` | `main_splatt3r.py` |

---

## Installation

### Prerequisites
- Ubuntu 20.04+ (or WSL2 on Windows)
- NVIDIA GPU with CUDA 11.8+ **or** AMD GPU with ROCm 7.1
- Conda/Miniconda
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/Looong01/Splatt3R-SLAM.git --recursive
cd Splatt3R-SLAM/

# If you cloned without --recursive:
# git submodule update --init --recursive
```

### Step 2: Create Environment
```bash
conda create -n splatt3r-slam python=3.11
conda activate splatt3r-slam
```

### Step 3: Install PyTorch
Choose one backend and install the matching PyTorch build.

**CUDA (NVIDIA)**
Check your CUDA version with `nvcc --version`, then install matching PyTorch:
```bash
# For CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# For CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

**ROCm 7.1 (AMD)**
```bash
# Use the ROCm 7.1 wheel index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1

# Optional but recommended for deterministic arch targeting
export PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx1100;gfx1101;gfx1200"
```

### Step 4: Install Dependencies (IN THIS ORDER!)

**4a. Install lietorch FIRST** (critical dependency):
```bash
pip install git+https://github.com/princeton-vl/lietorch.git
```

**4b. Install thirdparty dependencies**:
```bash
pip install -e thirdparty/in3d
pip install -e thirdparty/diff-gaussian-rasterization-modified
```

**4c. Install main package**:
```bash
# Auto-detect (default): picks ROCm when PyTorch is ROCm build, otherwise CUDA when available.
pip install --no-build-isolation -e .

# Force CUDA extension build
SPLATT3R_GPU_BACKEND=cuda pip install --no-build-isolation -e .

# Force ROCm extension build (for AMD + ROCm 7.1)
SPLATT3R_GPU_BACKEND=rocm pip install --no-build-isolation -e .
```

If you are on AMD/ROCm, choose `SPLATT3R_GPU_BACKEND=rocm` (or leave default auto-detect with ROCm PyTorch). If you are on NVIDIA/CUDA, choose `SPLATT3R_GPU_BACKEND=cuda`.

**4d. Install Splatt3R-specific dependencies**:
```bash
pip install lightning lpips omegaconf huggingface_hub gitpython einops
```

**4e. (Optional) Install torchcodec** for faster mp4 loading:
```bash
pip install torchcodec==0.1
```

### Verify Installation
```bash
python -c "import lietorch, PIL, cv2, einops, lightning, lpips, omegaconf; print('All dependencies OK')"
python -c "from splatt3r_slam.splatt3r_utils import load_splatt3r; print('splatt3r_slam OK')"
```

### Checkpoint
Download MASt3R backbone weights (required):
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

The Splatt3R checkpoint (`epoch=19-step=1200.ckpt`, ~150MB) will be **automatically loaded** from `checkpoints/` if present, or **downloaded from HuggingFace** on first run.
To download manually:
```bash
# https://huggingface.co/brandonsmart/splatt3r_v1.0/blob/main/epoch=19-step=1200.ckpt
wget 'https://huggingface.co/brandonsmart/splatt3r_v1.0/resolve/main/epoch%3D19-step%3D1200.ckpt' -O checkpoints/epoch=19-step=1200.ckpt
```

---

## Usage

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `datasets/tum/rgbd_dataset_freiburg1_desk` | Path to dataset, video, or `realsense` |
| `--config` | `config/base.yaml` | Path to config YAML |
| `--calib` | `""` | Path to camera intrinsics YAML (optional) |
| `--checkpoint` | `None` | Path to Splatt3R checkpoint (auto-downloads if not set) |
| `--no-viz` | `False` | Disable visualization window |
| `--render-gaussians` | **`True`** | Save per-frame Gaussian-rendered PNGs |
| `--no-render-gaussians` | `False` | Disable per-frame PNG saving |
| `--render-dir` | `logs/gaussian_renders` | Directory for rendered PNGs |
| `--max-gaussians` | `4194304` (4M) | Max Gaussians in shared visualization buffer |
| `--spatial-stride` | `4` | Spatial subsampling stride (1 = no subsampling, 4 = 16× fewer Gaussians per frame) |
| `--save-as` | `default` | Save tag for results |

### Quick Test
```bash
bash ./scripts/download_tum.sh
python main_splatt3r.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml
```

By default, per-frame Gaussian-rendered PNGs are saved to `logs/gaussian_renders/`.

### Custom Gaussian Parameters
```bash
# Higher density Gaussians (slower, better quality)
python main_splatt3r.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --spatial-stride 1 \
    --max-gaussians 8388608

# Lower density Gaussians (faster, less memory)
python main_splatt3r.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --spatial-stride 8 \
    --max-gaussians 2097152
```

### Disable PNG Saving
```bash
python main_splatt3r.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --no-render-gaussians
```

### With Camera Calibration
```bash
python main_splatt3r.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_room/ \
    --config config/calib.yaml

# With custom intrinsics
python main_splatt3r.py \
    --dataset path/to/data \
    --config config/base.yaml \
    --calib config/intrinsics.yaml
```

### Run on Video / Image Folder
```bash
python main_splatt3r.py --dataset path/to/video.mp4 --config config/base.yaml
python main_splatt3r.py --dataset path/to/image_folder --config config/base.yaml
```

### Live Demo (RealSense)
```bash
python main_splatt3r.py --dataset realsense --config config/base.yaml
```

### Headless Mode (No GUI)
```bash
python main_splatt3r.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --no-viz
```

### Original MASt3R-SLAM (for comparison)
```bash
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml
```

---

## Output

| Output | Location | Description |
|--------|----------|-------------|
| Trajectory | `logs/<dataset>/trajectory.txt` | Camera trajectory (TUM format) |
| Reconstruction | `logs/<dataset>/reconstruction.ply` | 3D point cloud |
| Keyframes | `logs/keyframes/` | Saved keyframe images |
| GS Renders | `logs/gaussian_renders/` | Per-frame Gaussian-rendered PNGs |

---

## Architecture

```
Splatt3R-SLAM/
├── main_splatt3r.py         # Main entry point (Splatt3R-SLAM)
├── main.py                  # Original MASt3R-SLAM entry point
├── thirdparty/
│   ├── in3d/                # OpenGL camera/visualization library
│   ├── diff-gaussian-rasterization-modified/  # CUDA Gaussian rasterizer (submodule)
│   ├── mast3r/              # MASt3R upstream (submodule)
│   └── eigen/               # Eigen headers
├── splatt3r_core/           # Core Splatt3R implementation
│   ├── main.py              # MAST3RGaussians Lightning module
│   ├── src/
│   │   ├── mast3r_src/      # MASt3R encoder with Gaussian head
│   │   └── pixelsplat_src/  # PixelSplat decoder (CUDA rasterizer)
│   └── utils/               # Geometry, SH, loss utilities
├── splatt3r_slam/           # SLAM package with Splatt3R
│   ├── splatt3r_utils.py    # Model loading, inference, Gaussian conversion
│   ├── tracker.py           # Frame tracking
│   ├── global_opt.py        # Global optimization / bundle adjustment
│   ├── frame.py             # Frame + SharedGaussians buffer
│   ├── visualization.py     # Interactive GS rendering + OpenGL
│   └── ...                  # Other SLAM components
├── mast3r_slam/             # Original MASt3R-SLAM (for comparison)
├── config/                  # YAML configuration files
├── scripts/                 # Dataset download & evaluation scripts
└── checkpoints/             # Model checkpoints
```

### Inference Pipeline

1. **Encode**: `model.encoder._encode_image()` → features + positions
2. **Decode**: `model.encoder._decoder()` → cross-attention tokens
3. **Downstream Head**: `model.encoder._downstream_head()` → 3D points, confidence, descriptors, **Gaussian params** (means, scales, rotations, SH, opacities)
4. **SH Residual**: Network outputs SH residuals; original image colour is added: `sh[..., 0] += RGB2SH(original_image)`
5. **World Transform**: Per-frame Gaussians are transformed to world coordinates via camera pose
6. **Rasterize**: `diff_gaussian_rasterization` renders from any viewpoint

### Splatt3R Model Output

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `pts3d` | (B, H, W, 3) | 3D point estimates |
| `conf` | (B, H, W) | Confidence scores |
| `desc` | (B, H, W, 24) | Feature descriptors |
| `means` | (B, H, W, 3) | Gaussian centres |
| `scales` | (B, H, W, 3) | Gaussian scales (exp-activated) |
| `rotations` | (B, H, W, 4) | Quaternion rotations (L2-normalised) |
| `sh` | (B, H, W, 3, 1) | SH residuals (degree 0 DC only) |
| `opacities` | (B, H, W, 1) | Opacity (sigmoid-activated, [0,1]) |

---

## Downloading Datasets

### TUM-RGBD Dataset
```bash
bash ./scripts/download_tum.sh
```

### 7-Scenes Dataset
```bash
bash ./scripts/download_7_scenes.sh
```

### EuRoC Dataset
```bash
bash ./scripts/download_euroc.sh
```

### ETH3D SLAM Dataset
```bash
bash ./scripts/download_eth3d.sh
```

---

## Evaluations

All evaluation scripts run in single-threaded headless mode. Can run with or without calibration:

### TUM-RGBD
```bash
bash ./scripts/eval_tum.sh
bash ./scripts/eval_tum.sh --no-calib
```

### 7-Scenes
```bash
bash ./scripts/eval_7_scenes.sh
bash ./scripts/eval_7_scenes.sh --no-calib
```

### EuRoC
```bash
bash ./scripts/eval_euroc.sh
bash ./scripts/eval_euroc.sh --no-calib
```

### ETH3D
```bash
bash ./scripts/eval_eth3d.sh
```

---

## Troubleshooting

### "No module named 'lietorch'"
lietorch must be installed **before** the main package:
```bash
pip install git+https://github.com/princeton-vl/lietorch.git
pip install --no-build-isolation -e .
```

### "No module named 'torch'" / wrong environment
```bash
conda activate splatt3r-slam
```

### "CUDA out of memory"
Reduce Gaussian density or image resolution:
```bash
# Increase spatial stride (fewer Gaussians)
python main_splatt3r.py --dataset ... --spatial-stride 8 --max-gaussians 2097152

# Or reduce image resolution in config:
# config/base.yaml → dataset.img_downsample: 2
```

### "Failed to download checkpoint"
Download manually:
```bash
mkdir -p checkpoints/
# Download from: https://huggingface.co/brandonsmart/splatt3r_v1.0/blob/main/epoch%3D19-step%3D1200.ckpt
python main_splatt3r.py --checkpoint checkpoints/epoch=19-step=1200.ckpt ...
```

### Visualization not showing
Run headless:
```bash
python main_splatt3r.py --dataset ... --no-viz
```

### WSL Users
```bash
git checkout windows
```
This disables multiprocessing which causes shared memory issues ([details](https://github.com/rmurai0610/MASt3R-SLAM/issues/21)).

### Quick Fix Commands
```bash
# Reinstall lietorch
pip uninstall lietorch -y && pip install git+https://github.com/princeton-vl/lietorch.git

# Reinstall main package
pip uninstall Splatt3R-SLAM -y && pip install --no-build-isolation -e .

# Install missing dependencies
pip install Pillow opencv-python tqdm pyyaml einops
pip install lightning lpips omegaconf huggingface_hub gitpython
```

---

## Reproducibility
There might be minor differences between the released version and results in the paper after developing this multi-processing version.
We run all experiments on an RTX 4090; performance may differ with a different GPU.

---

## Acknowledgement
We sincerely thank the developers and contributors of the many open-source projects that our code is built upon.
- [Splatt3R](https://splatt3r.active.vision) - Zero-shot Gaussian Splatting
- [MASt3R](https://github.com/naver/mast3r) - Matching and Stereo 3D Reconstruction
- [MASt3R-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm)
- [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/) - Original SLAM system
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [ModernGL](https://github.com/moderngl/moderngl)
- [PixelSplat](https://github.com/dcharatan/pixelsplat) - Gaussian Splatting components

---

## Citation

### Splatt3R
```bibtex
@article{smart2024splatt3r,
  title={Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs}, 
  author={Brandon Smart and Chuanxia Zheng and Iro Laina and Victor Adrian Prisacariu},
  year={2024},
  eprint={2408.13912},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
}
```

### MASt3R-SLAM
```bibtex
@inproceedings{murai2024_mast3rslam,
  title={{MASt3R-SLAM}: Real-Time Dense {SLAM} with {3D} Reconstruction Priors},
  author={Murai, Riku and Dexheimer, Eric and Davison, Andrew J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```

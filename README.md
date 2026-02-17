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

---

## Installation

### Prerequisites
- Ubuntu 20.04+ (or WSL2 on Windows)
- NVIDIA GPU with CUDA 11.8+
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
conda create -n splatt3r-slam python=3.11 -y
conda activate splatt3r-slam
```

### Step 3: Install PyTorch
Check your CUDA version with `nvcc --version`, then install matching PyTorch:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Dependencies (IN THIS ORDER!)

```bash
pip install -r requirements.txt
pip install -e thirdparty/in3d
pip install --no-build-isolation thirdparty/asmk
pip install --no-build-isolation thirdparty/lietorch
pip install --no-build-isolation thirdparty/diff-gaussian-rasterization-modified
pip install --no-build-isolation -e .
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

## Command-Line Arguments

`main.py` currently supports:

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | `datasets/tum/rgbd_dataset_freiburg1_desk` | Input sequence folder, video path, or `realsense` |
| `--config` | `config/base.yaml` | SLAM config YAML |
| `--save-as` | `default` | Output naming for evaluation save path |
| `--no-viz` | off | Disable interactive GUI window |
| `--calib` | `""` | Optional calibration YAML path |
| `--checkpoint` | `None` | Splatt3R checkpoint (auto-downloads if not set) |
| `--render-gaussians` | on | Deprecated compatibility flag (rendering is enabled by default) |
| `--no-render-gaussians` | off | Disable Splatt3R rendering and PNG export |
| `--render-dir` | `logs/gaussian_renders` | Directory for per-frame rendered PNGs |
| `--max-gaussians` | `4194304` | Max Gaussians in shared visualization buffer |
| `--spatial-stride` | `4` | Per-frame Gaussian subsampling stride (`1` = no subsampling) |
| `--depth-max-percentile` | `0.98` | Depth percentile cutoff for splash filtering (`1.0` = off) |
| `--max-scale` | `1.0` | Remove Gaussians with any scale axis above this value |
| `--min-confidence` | `1.5` | Remove Gaussians at low-confidence pixels (`0` = off) |
| `--keep-ratio` | `0.6` | Keep top N% per frame by quality score (`1.0` = off) |

Example with explicit rendering-related parameters:

```bash
python main.py \
  --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
  --config config/base.yaml \
  --spatial-stride 2 \
  --max-gaussians 6000000 \
  --render-dir logs/gaussian_renders \
  --max-scale 0.3 --min-confidence 2.0 --depth-max-percentile 0.95
```

## GUI Controls (Interactive Viz)

When GUI is enabled (default, without `--no-viz`), the left panel exposes runtime controls:

| GUI Item | Range / Default | Effect |
|---|---:|---|
| `pause` | bool | Pause frame stepping |
| `C_conf_threshold` | `0.0 .. 10.0` (default `1.5`) | Filters low-confidence points before rendering |
| `show all` | bool (on) | Show all point maps |
| `follow cam` | bool (on) | View follows current tracking camera |
| `spatial stride` | `1 .. 16` (default from CLI `--spatial-stride`) | Subsampling density control per frame |
| `max gaussians (k)` | `64k .. 8192k` (default from CLI `--max-gaussians`) | Cap total active Gaussians in shared buffer |
| **Splash Filter** | | |
| `depth max pct` | `0.5 .. 1.0` (default from CLI `--depth-max-percentile`) | Depth percentile cutoff — lower = more aggressive depth culling |
| `max scale` | `0.01 .. 3.0` (default from CLI `--max-scale`) | Max Gaussian scale axis — lower = remove large splash blobs |
| `min confidence` | `0.0 .. 10.0` (default from CLI `--min-confidence`) | Pointmap confidence gate — higher = keep only confident predictions |
| `keep ratio` | `0.1 .. 1.0` (default from CLI `--keep-ratio`) | Quality-percentile filter — lower = more aggressive per-frame culling |
| `GS rendering (Splatt3R)` | bool (on) | Toggle Gaussian splatting rendering overlay |
| `GS resolution` | `0.1 .. 1.0` (default `0.5`) | Rendering resolution scale in viewport |
| `surfelmap` / `trianglemap` | radio | Point-cloud shader (when GS rendering is off) |
| `show_keyframe_edges` / `show_keyframe` / `show_axis` | bool | Overlay debugging visuals |
| `show_normal` / `culling` | bool | Normal display & face culling (point-cloud mode) |
| `show_curr_pointmap` | bool (on) | Show current frame point map |
| `radius` / `slant_threshold` | drag control | Point-cloud shader params |
| `line_thickness` / `frustum_scale` | drag control | Frustum/edge visualization style |

### CLI vs GUI Priority

- `--spatial-stride`, `--max-gaussians`, `--depth-max-percentile`, `--max-scale`, `--min-confidence`, `--keep-ratio` are **startup defaults** and initialize GUI sliders.
- During GUI run, slider updates are applied live to subsequent frames.
- For PNG export in `logs/gaussian_renders/`, current GUI values of `spatial_stride` and `max_gaussians` are used; other GUI sliders are viewport-only.
- The `--max-gaussians` CLI value determines the **shared memory buffer size** allocated at startup; the GUI slider upper bound is 8M (FIFO eviction keeps memory bounded).
- In headless mode (`--no-viz`), only CLI values are used for the whole run.
- If `--no-render-gaussians` is set, Splatt3R rendering and PNG export are disabled regardless of GUI state.

### Splash Artifact Filtering

When the model predicts Gaussians for **occluded / unseen** regions (e.g. the back
of a monitor), it generates large, mis-positioned "splash" blobs.  A five-stage
pipeline inside `gaussians_to_world()` removes these **before** adding to the
shared buffer, plus a **voxel-based spatial replacement** policy ensures that
old bad Gaussians are removed when a new better view arrives:

1. **Depth filter** (`depth_max_percentile`): removes Gaussians deeper than the
   *p*-th percentile of all positive depths.  Default `0.98`.
2. **Scale filter** (`max_scale`): removes Gaussians whose max axis scale
   exceeds this threshold.  Default `1.0`.
3. **Confidence filter** (`min_confidence`): removes Gaussians at pixel
   positions with pointmap confidence below this value.  Default `1.5`.
4. **FOV cone filter** (internal, 50° half-angle): removes Gaussians predicted
   at extreme angles from the camera's optical axis — typically hallucinations.
5. **Quality percentile filter** (`keep_ratio`): computes a per-Gaussian
   quality score = conf / (scale + ε) and keeps only the top *keep_ratio*
   fraction.  Default `0.6` (keep top 60%).
6. **Voxel-based spatial replacement** (automatic): before appending new
   Gaussians, old Gaussians in the same 5 cm voxels are **replaced** if the
   new Gaussians have a higher quality score.  This prevents splash from one
   viewpoint from corrupting reconstructions built from a different viewpoint.

If splash is still visible, try stricter values:

```bash
python main.py --max-scale 0.3 --min-confidence 2.0 --depth-max-percentile 0.95 --keep-ratio 0.5
```

All filter parameters are adjustable **live** via the GUI sliders during runtime.

### Quick Test
```bash
bash ./scripts/download_tum.sh
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml
```

By default, per-frame Gaussian-rendered PNGs are saved to `logs/gaussian_renders/`.

### Custom Gaussian Parameters
```bash
# Higher density Gaussians (slower, better quality)
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --spatial-stride 1 \
    --max-gaussians 8388608

# Lower density Gaussians (faster, less memory)
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --spatial-stride 8 \
    --max-gaussians 2097152
```

### Disable PNG Saving
```bash
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --no-render-gaussians
```

### With Camera Calibration
```bash
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_room/ \
    --config config/calib.yaml

# With custom intrinsics
python main.py \
    --dataset path/to/data \
    --config config/base.yaml \
    --calib config/intrinsics.yaml
```

### Run on Video / Image Folder
```bash
python main.py --dataset path/to/video.mp4 --config config/base.yaml
python main.py --dataset path/to/image_folder --config config/base.yaml
```

If the calibration parameters are known, you can specify them in intrinsics.yaml
```bash
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml --calib config/intrinsics.yaml
python main.py --dataset <path/to/folder> --config config/base.yaml --calib config/intrinsics.yaml
```

### Live Demo (RealSense)
```bash
python main.py --dataset realsense --config config/base.yaml
```

### Headless Mode (No GUI)
```bash
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/base.yaml \
    --no-viz
```

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
├── main.py         # Main entry point (Splatt3R-SLAM)
├── thirdparty/
│   ├── in3d/                # OpenGL camera/visualization library
│   ├── diff-gaussian-rasterization-modified/  # CUDA Gaussian rasterizer (submodule)
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
python main.py --dataset ... --spatial-stride 8 --max-gaussians 2097152

# Or reduce image resolution in config:
# config/base.yaml → dataset.img_downsample: 2
```

### "Failed to download checkpoint"
Download manually:
```bash
mkdir -p checkpoints/
# Download from: https://huggingface.co/brandonsmart/splatt3r_v1.0/blob/main/epoch%3D19-step%3D1200.ckpt
python main.py --checkpoint checkpoints/epoch=19-step=1200.ckpt ...
```

### Visualization not showing
Run headless:
```bash
python main.py --dataset ... --no-viz
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

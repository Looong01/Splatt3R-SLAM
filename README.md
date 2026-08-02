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
- NVIDIA GPU (compute capability >= 7.5; developed on RTX A6000, sm_86)
- CUDA 13.x toolkit (`nvcc --version`; developed with 13.3)
- Conda/Miniconda
- Git

> **Note:** All third-party code (faiss, lietorch, asmk, in3d, pyimgui, eigen, glm,
> diff-gaussian-rasterization) is **vendored** under `thirdparty/` with the CUDA 13
> adaptations described below already applied. No git submodules are used anymore.

### Step 1: Clone Repository
```bash
git clone https://github.com/Looong01/Splatt3R-SLAM.git
cd Splatt3R-SLAM/
```

### Step 2: Create Environment
```bash
conda create -n splatt3r-slam python=3.11 -y
conda activate splatt3r-slam
```

### Step 3: Install PyTorch (CUDA 13.2 build)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```
Developed with `torch 2.13.0+cu132` / `torchvision 0.28.0+cu132`.

For the original CUDA 12.4 configuration instead:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Build Tools
cmake, ninja and swig are needed to compile faiss and the torch CUDA extensions;
MKL provides the BLAS backend for faiss (all via pip, no system packages required):
```bash
pip install cmake ninja swig mkl-devel
```

### Step 5: Install Python Dependencies (IN THIS ORDER!)

```bash
pip install -r requirements.txt
pip install -e thirdparty/in3d
pip install --no-build-isolation thirdparty/asmk
```

### Step 6: Build and Install faiss (GPU)
faiss is built from the vendored source in `thirdparty/faiss` (v1.14.3) with GPU
support — prebuilt PyPI packages (`faiss-cpu`, `faiss-gpu-cu12`) must NOT be
installed, they would shadow this build and pin `numpy<2`.
All required CMake options are pre-seeded in `thirdparty/faiss/CMakeLists.txt`
(GPU on, Release, CUDA arch auto-detected from the local GPU via
`CMAKE_CUDA_ARCHITECTURES=native`, BLAS auto-detected), so a plain configure
works:

```bash
cd thirdparty/faiss
cmake -B build .
make -C build -j$(nproc) faiss swigfaiss
pip install --no-build-isolation --no-deps ./build/faiss/python
cd ../..
```
The defaults prefer the active conda env (`CONDA_PREFIX`) for python/swig, and
auto-detect MKL (e.g. from `pip install mkl-devel`). If MKL is not found, faiss
automatically falls back to threaded OpenBLAS / system BLAS — install OpenBLAS
instead on AMD CPUs for better performance (MKL runs on AMD but slower). To
force a specific GPU architecture, e.g. for cross-compiling:
`cmake -B build . -DCMAKE_CUDA_ARCHITECTURES=89`.

### Step 7: Build the torch CUDA Extensions
```bash
export CUDA_HOME=/usr/local/cuda   # CUDA 13.x toolkit
export MAX_JOBS=$(nproc)           # parallel compilation

pip install --no-build-isolation thirdparty/lietorch
pip install --no-build-isolation thirdparty/diff-gaussian-rasterization-modified
pip install --no-build-isolation -e .

# Optional but recommended: CUDA RoPE2D kernel for MASt3R (otherwise a slower
# PyTorch fallback is used and a warning is printed at startup)
cd splatt3r_core/src/mast3r_src/dust3r/croco/models/curope
python setup.py build_ext --inplace
cd ../../../../../../..
```

### CUDA 13.x / torch 2.13 Adaptations (already applied in this repo)
If you build from this repository you do not need to change anything — the
following fixes are part of the vendored code:

- `setup.py`: dropped `compute_60/61/70` gencode flags (removed in CUDA 13;
  minimum is `sm_75`)
- `splatt3r_slam/backend/src/gn_kernels.cu`: `torch::linalg::linalg_norm` →
  `at::linalg_norm` (the `torch::linalg` C++ namespace was removed in torch 2.13)
- `splatt3r_slam/backend/src/matching_kernels.cu`: `D11.type()` →
  `D11.scalar_type()` in the dispatch macro
- `splatt3r_core/src/mast3r_src/dust3r/croco/models/curope/kernels.cu`:
  `tokens.type()` → `tokens.scalar_type()` (same torch 2.13 change; required to
  build the optional CUDA RoPE2D kernel)
- `splatt3r_slam/dataloader.py`: `np.unicode_` → `np.str_` (removed in NumPy 2.0)
- `main.py`, `splatt3r_core/main.py`: set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
  because torch >= 2.6 defaults `torch.load` to `weights_only=True`, which rejects
  the pickled objects (omegaconf `DictConfig`, `argparse.Namespace`, faiss
  indices) inside the MASt3R/Splatt3R checkpoints
- `thirdparty/asmk/pyproject.toml`: dropped the `faiss-cpu` dependency (faiss is
  provided by the vendored GPU build)
- `requirements.txt`: unpinned `numpy`, replaced `faiss-gpu-cu12` with the
  vendored source build, made `torchcodec` optional
- `thirdparty/faiss/CMakeLists.txt`: pre-seeded defaults (GPU on, Release,
  CUDA arch auto-detected via `native`, MKL auto-detected with OpenBLAS
  fallback, conda python/swig) so a plain `cmake -B build .` works
- submodule `.git` metadata removed; everything is plain source in one repo

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
| `--lora` | `None` | Path to a trained LoRA adapter dir to hot-swap onto the base checkpoint (see `scripts/train_lora_per_scene.py`) |
| `--max-gaussians` | `16777216` | Target Gaussian **render budget** across all keyframes; the baker coarsens its stride to stay under it |
| `--spatial-stride` | `1` | Per-frame Gaussian subsampling stride (`1` = no subsampling). See the warning below |
| `--depth-max-percentile` | `0.98` | Upper depth percentile kept when baking Gaussians |
| `--max-scale` | `1.0` | Clamp on predicted Gaussian scale (guards the rasterizer) |
| `--min-confidence` | `1.5` | Minimum pointmap confidence for a Gaussian to be kept |

> **`--spatial-stride` stability note.** The default — and the only
> recommended value — is `1` (full per-pixel density). The vendored CUDA
> rasterizer has been observed to hit `illegal memory access` once enough
> Gaussians accumulate, and this happens with `stride=2` as well, just
> less reliably; higher strides (4, 8) thin the map further and may dodge
> the crash, but nothing above `1` is guaranteed stable. The GUI slider
> changes it live.

Example with explicit rendering-related parameters:

```bash
python main.py \
  --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
  --config config/base.yaml \
  --spatial-stride 1 \
  --max-gaussians 16777216 \
  --render-dir logs/gaussian_renders
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
| `max gaussians (k)` | `64k .. CLI upper bound` (default from CLI `--max-gaussians`) | Live cap on the total Gaussian render budget |
| `GS rendering (Splatt3R)` | bool (on) | Toggle Gaussian splatting rendering overlay |
| `GS resolution` | `0.1 .. 1.0` (default `1.0`) | Rendering resolution scale in viewport |
| `surfelmap` / `trianglemap` | radio | Point-cloud shader (when GS rendering is off) |
| `show_keyframe_edges` / `show_keyframe` / `show_axis` | bool | Overlay debugging visuals |
| `show_normal` / `culling` | bool | Normal display & face culling (point-cloud mode) |
| `show_curr_pointmap` | bool (on) | Show current frame point map |
| `radius` / `slant_threshold` | drag control | Point-cloud shader params |
| `line_thickness` / `frustum_scale` | drag control | Frustum/edge visualization style |

### CLI vs GUI Priority

- `--spatial-stride` and `--max-gaussians` are **startup defaults** and initialize GUI sliders.
- During GUI run, slider updates are applied live to subsequent frames.
- For PNG export in `logs/gaussian_renders/`, current GUI values of `spatial_stride` and `max_gaussians` are used; other GUI sliders are viewport-only.
- `--max-gaussians` is a **render budget**, not a preallocated buffer: Gaussians are re-baked from each keyframe every frame (there is no persistent `SharedGaussians` store -- that design was removed), and the baker raises its stride to stay under the budget. The CLI value sets the GUI slider's upper bound.
- In headless mode (`--no-viz`), only CLI values are used for the whole run.
- If `--no-render-gaussians` is set, Splatt3R rendering and PNG export are disabled regardless of GUI state.

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

Paths below are what `splatt3r_slam/evaluate.py: prepare_savedir()` actually
produces: everything lands directly under `logs/` (or `logs/<--save-as>/` when
`--save-as` is given), and files are named after the **sequence**, i.e.
`<seq_name>` = the dataset directory's own name, e.g.
`rgbd_dataset_freiburg1_desk`.

| Output | Location | Description |
|--------|----------|-------------|
| Trajectory | `logs/<seq_name>.txt` | Camera trajectory (TUM format) |
| Reconstruction | `logs/<seq_name>.ply` | 3D point cloud (positions + colours only) |
| **Gaussian map** | `logs/<seq_name>_gaussians.ply` | **The Gaussian splatting map** in standard 3DGS `.ply` form (means, covariance as quaternion+scale, SH DC, opacity) -- openable in any 3DGS viewer and re-renderable from new viewpoints. Written automatically together with the trajectory/reconstruction saves at the end of a run |
| Keyframes | `logs/keyframes/<seq_name>/` | Saved keyframe images |
| GS Renders | `logs/gaussian_renders/` | Per-frame Gaussian-rendered PNGs (`--render-dir`) |

> With `--save-as NAME`, these become `logs/NAME/<seq_name>.txt`,
> `logs/NAME/<seq_name>.ply`, `logs/NAME/<seq_name>_gaussians.ply`,
> `logs/NAME/keyframes/<seq_name>/`.
>
> The Gaussian map honours `--spatial-stride`, `--depth-max-percentile`,
> `--max-scale` and `--min-confidence`, but **not** the stride-proportional
> scale inflation the live viewport applies -- that is a display-only
> compensation and would write oversized splats into a persisted file.

---

## Architecture

```
Splatt3R-SLAM/
├── main.py         # Main entry point (Splatt3R-SLAM)
├── thirdparty/
│   ├── in3d/                # OpenGL camera/visualization library
│   ├── faiss/               # FAISS v1.14.3 source (built with GPU support)
│   ├── lietorch/            # Lie groups for PyTorch (CUDA extension)
│   ├── asmk/                # ASMK image retrieval
│   ├── diff-gaussian-rasterization-modified/  # CUDA Gaussian rasterizer
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
│   ├── frame.py             # Frame + SharedKeyframes (per-keyframe camera-space Gaussians)
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
Downloads from the ETH Research Collection (three large archives, ~23 GB total;
only the required per-sequence zips are kept afterwards):
```bash
bash ./scripts/download_euroc.sh
```

### ETH3D SLAM Dataset
Downloads all training sequences from https://www.eth3d.net/slam_datasets
(resumable, already-downloaded sequences are skipped):
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
lietorch must be installed from the vendored source **before** the main package:
```bash
pip install --no-build-isolation thirdparty/lietorch
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
pip uninstall lietorch -y && pip install --no-build-isolation thirdparty/lietorch

# Reinstall main package
pip uninstall Splatt3R-SLAM -y && pip install --no-build-isolation -e .

# Install missing dependencies
pip install Pillow opencv-python tqdm pyyaml einops
pip install lightning lpips omegaconf huggingface_hub gitpython
```

---

## Reproducibility
There might be minor differences between the released version and results in the paper after developing this multi-processing version.

The upstream MASt3R-SLAM numbers this project builds on were produced on an
RTX 4090. **This** fork is developed and run on dual RTX A6000 (sm_86), which
is also what the Prerequisites section above and the LoRA training scripts
assume -- performance and memory headroom will differ on other GPUs.

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

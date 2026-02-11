# Splatt3R-SLAM: Real-Time Dense SLAM with 3D Gaussian Splatting

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

## ⚠️ Important: Installation Required

**Before running Splatt3R-SLAM**, you MUST install all dependencies following the instructions below. The system will check for missing dependencies and provide installation commands if needed.

If you get import errors, see:
- [Installation Instructions](#installation) (below)
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions

## Overview

Splatt3R-SLAM integrates [Splatt3R](https://splatt3r.active.vision) (Zero-shot Gaussian Splatting from Uncalibrated Image Pairs) into a real-time SLAM system. This combines the dense 3D reconstruction capabilities of MASt3R-SLAM with Splatt3R's 3D Gaussian Splatting for improved scene representation.

### Key Features
- **3D Gaussian Splatting**: Uses Splatt3R to predict 3D Gaussians directly from image pairs
- **Zero-shot Reconstruction**: No scene-specific training required  
- **Real-time Performance**: Maintains real-time SLAM capabilities
- **Dense 3D Reconstruction**: Produces detailed 3D reconstructions with Gaussian splats

# Getting Started
## Installation

**⚠️ IMPORTANT: Follow these steps IN ORDER**

### 1. Create Environment
```bash
conda create -n splatt3r-slam python=3.11
conda activate splatt3r-slam
```
### 2. Install PyTorch

Check your system's CUDA version:
```bash
nvcc --version
```

Install PyTorch with **matching** CUDA version:
```bash
# For CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# For CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 3. Clone Repository

```bash
git clone https://github.com/Looong01/Splatt3R-SLAM.git --recursive
cd Splatt3R-SLAM/

# If you cloned without --recursive, run:
# git submodule update --init --recursive
```

### 4. Install Dependencies (IN THIS ORDER!)

**Step 4a: Install lietorch FIRST** (critical dependency):
```bash
pip install git+https://github.com/princeton-vl/lietorch.git
```

**Step 4b: Install thirdparty dependencies**:
```bash
pip install -e thirdparty/in3d
```

**Step 4c: Install main package**:
```bash
pip install --no-build-isolation -e .
```

**Step 4d: Install Splatt3R-specific dependencies**:
```bash
pip install lightning lpips omegaconf huggingface_hub gitpython
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

**Step 4e (Optional): Install torchcodec for faster mp4 loading**:
```bash
pip install torchcodec==0.1
```

### 5. Verify Installation

After installation, verify dependencies are correctly installed:
```bash
python -c "import lietorch, PIL, cv2, einops, lightning, lpips, omegaconf; print('✓ All dependencies OK')"
```

If you see any import errors, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Quick Test

Download a test dataset and run:
```bash
bash ./scripts/download_tum.sh
python main_splatt3r.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml
```

If you get dependency errors, the script will tell you exactly what to install.

---

## Checkpoint Download

The Splatt3R checkpoint will be automatically downloaded from HuggingFace when you first run the system.
You can also manually download it:
```bash
mkdir -p checkpoints/
# The system will download this automatically:
# https://huggingface.co/brandonsmart/splatt3r_v1.0/blob/main/epoch=19-step=1200.ckpt
```

## WSL Users
We have primarily tested on Ubuntu. If you are using WSL, please checkout to the windows branch and follow the above installation.
```
git checkout windows
```
This disables multiprocessing which causes an issue with shared memory as discussed [here](https://github.com/rmurai0610/MASt3R-SLAM/issues/21).

## Examples
Run Splatt3R-SLAM on TUM dataset:
```
bash ./scripts/download_tum.sh
python main_splatt3r.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```

For the original MASt3R-SLAM version (without Gaussian Splatting):
```
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```
## Live Demo
Connect a realsense camera to the PC and run
```
python main_splatt3r.py --dataset realsense --config config/base.yaml
```
## Running on a video
Our system can process either MP4 videos or folders containing RGB images.
```
python main_splatt3r.py --dataset <path/to/video>.mp4 --config config/base.yaml
python main_splatt3r.py --dataset <path/to/folder> --config config/base.yaml
```
If the calibration parameters are known, you can specify them in intrinsics.yaml
```
python main_splatt3r.py --dataset <path/to/video>.mp4 --config config/base.yaml --calib config/intrinsics.yaml
python main_splatt3r.py --dataset <path/to/folder> --config config/base.yaml --calib config/intrinsics.yaml
```

## Downloading Dataset
### TUM-RGBD Dataset
```
bash ./scripts/download_tum.sh
```

### 7-Scenes Dataset
```
bash ./scripts/download_7_scenes.sh
```

### EuRoC Dataset
```
bash ./scripts/download_euroc.sh
```
### ETH3D SLAM Dataset
```
bash ./scripts/download_eth3d.sh
```

## Running Evaluations
All evaluation script will run our system in a single-threaded, headless mode.
We can run evaluations with/without calibration:
### TUM-RGBD Dataset
```
bash ./scripts/eval_tum.sh 
bash ./scripts/eval_tum.sh --no-calib
```

### 7-Scenes Dataset
```
bash ./scripts/eval_7_scenes.sh 
bash ./scripts/eval_7_scenes.sh --no-calib
```

### EuRoC Dataset
```
bash ./scripts/eval_euroc.sh 
bash ./scripts/eval_euroc.sh --no-calib
```
### ETH3D SLAM Dataset
```
bash ./scripts/eval_eth3d.sh 
```

## Reproducibility
There might be minor differences between the released version and the results in the paper after developing this multi-processing version. 
We run all our experiments on an RTX 4090, and the performance may differ when running with a different GPU.

## Acknowledgement
We sincerely thank the developers and contributors of the many open-source projects that our code is built upon.
- [Splatt3R](https://splatt3r.active.vision) - Zero-shot Gaussian Splatting
- [MASt3R](https://github.com/naver/mast3r) - Matching and Stereo 3D Reconstruction
- [MASt3R-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm)
- [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/) - Original SLAM system
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [ModernGL](https://github.com/moderngl/moderngl)
- [PixelSplat](https://github.com/dcharatan/pixelsplat) - Gaussian Splatting components

# Citation
If you found this code/work to be useful in your own research, please considering citing the following:

## Splatt3R
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

## MASt3R-SLAM
```bibtex
@inproceedings{murai2024_mast3rslam,
  title={{MASt3R-SLAM}: Real-Time Dense {SLAM} with {3D} Reconstruction Priors},
  author={Murai, Riku and Dexheimer, Eric and Davison, Andrew J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```

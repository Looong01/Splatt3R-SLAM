# Installation Fix Summary

## Date: 2026-02-11

## Problem Reported

User followed the installation instructions and got an import error:
```
Traceback (most recent call last):
  File "/home/loong/桌面/Codes/3R/Splatt3R-SLAM/main_splatt3r.py", line 15, in 
    from splatt3r_slam.global_opt import FactorGraph
ModuleNotFoundError: No module named 'lietorch'
```

Additional missing modules: PIL, cv2, lightning, lpips, omegaconf, etc.

## Root Causes Identified

1. **pyproject.toml Issues**:
   - Only included `mast3r_slam` package
   - Missing `splatt3r_slam` and `splatt3r_core` packages
   - Missing many dependencies (Pillow, opencv-python, lightning, etc.)

2. **Installation Order**:
   - lietorch needs to be installed BEFORE the main package
   - Original instructions didn't specify this critical order

3. **setup.py Issues**:
   - Hardcoded to only check `mast3r_slam/backend`
   - Didn't check for `splatt3r_slam/backend`

## Fixes Applied

### 1. Updated pyproject.toml
```toml
[project]
name = "Splatt3R-SLAM"
version = "0.1.0"

# Added all missing dependencies
dependencies = [
    "numpy==1.26.4",
    "einops",
    "Pillow",              # NEW
    "opencv-python",       # NEW
    "tqdm",                # NEW
    "pyyaml",              # NEW
    "plyfile",
    "natsort",
    # Splatt3R-specific
    "lightning",           # NEW
    "lpips",               # NEW
    "omegaconf",           # NEW
    "huggingface_hub",     # NEW
    "gitpython",           # NEW
    # ... etc
]

[tool.setuptools.packages.find]
include = ["mast3r_slam", "splatt3r_slam", "splatt3r_core"]  # UPDATED
```

### 2. Updated setup.py
```python
# Now checks for backend in both locations
backend_dir = None
if os.path.exists(os.path.join(ROOT, "splatt3r_slam/backend")):
    backend_dir = "splatt3r_slam"
elif os.path.exists(os.path.join(ROOT, "mast3r_slam/backend")):
    backend_dir = "mast3r_slam"
```

### 3. Created requirements.txt
- Lists all Python dependencies
- Includes installation notes for git-based packages
- Makes dependency tracking easier

### 4. Updated Documentation

**README.md**:
- Added lietorch as FIRST installation step
- Clarified installation order
- Better formatting with bash code blocks

**QUICKSTART.md**:
- Added lietorch installation step
- Corrected dependency installation order
- Clearer step-by-step instructions

**TROUBLESHOOTING.md** (NEW):
- Common error messages and solutions
- Correct installation order
- Verification commands
- CUDA compilation troubleshooting
- Submodule issues

## Correct Installation Procedure

```bash
# 1. Create environment
conda create -n splatt3r-slam python=3.11
conda activate splatt3r-slam

# 2. Install PyTorch (match your CUDA version)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Clone repository
git clone https://github.com/Looong01/Splatt3R-SLAM.git --recursive
cd Splatt3R-SLAM/

# 4. Install lietorch FIRST (Critical!)
pip install git+https://github.com/princeton-vl/lietorch.git

# 5. Install thirdparty
pip install -e thirdparty/in3d

# 6. Install main package
pip install --no-build-isolation -e .

# 7. Install Splatt3R dependencies
pip install lightning lpips omegaconf huggingface_hub gitpython
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified

# 8. Optional: torchcodec
pip install torchcodec==0.1
```

## Verification

After installation, all these should work:

```bash
python -c "import lietorch; print('lietorch OK')"
python -c "import PIL; print('PIL OK')"
python -c "import cv2; print('cv2 OK')"
python -c "import lightning; print('lightning OK')"
python -c "from splatt3r_slam.splatt3r_utils import load_splatt3r; print('splatt3r_slam OK')"
python -c "from splatt3r_slam.global_opt import FactorGraph; print('global_opt OK')"
```

And the main command should work:
```bash
python main_splatt3r.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml
```

## Key Takeaways

1. **Installation order matters**: lietorch MUST be installed before the main package
2. **All packages must be declared**: pyproject.toml must include all Python packages in the repo
3. **Dependencies must be complete**: All import requirements must be in pyproject.toml
4. **Documentation is critical**: Clear installation steps prevent user issues

## Files Changed

1. ✅ `pyproject.toml` - Added packages and dependencies
2. ✅ `setup.py` - Fixed backend path detection
3. ✅ `requirements.txt` - Created for dependency tracking
4. ✅ `README.md` - Updated installation instructions
5. ✅ `QUICKSTART.md` - Updated installation order
6. ✅ `TROUBLESHOOTING.md` - Created comprehensive guide

## Result

✅ **Users can now successfully install and run Splatt3R-SLAM**

The import errors are fixed and the installation procedure is properly documented.

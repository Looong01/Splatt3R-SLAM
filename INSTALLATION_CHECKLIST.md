# Installation Verification Checklist

Use this checklist to verify your Splatt3R-SLAM installation is complete and working.

## Pre-Installation Checks

- [ ] Python 3.11 installed
  ```bash
  python --version  # Should show Python 3.11.x
  ```

- [ ] CUDA available (if using GPU)
  ```bash
  nvcc --version
  ```

- [ ] Conda environment created and activated
  ```bash
  conda env list  # Should show (splatt3r-slam) active
  ```

## Installation Steps Checklist

Follow these steps **in order**:

- [ ] **Step 1**: PyTorch installed with correct CUDA version
  ```bash
  python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
  # Should print: PyTorch 2.5.1+cu121, CUDA: True (or False if CPU-only)
  ```

- [ ] **Step 2**: Repository cloned with submodules
  ```bash
  ls thirdparty/  # Should show: eigen  in3d  mast3r
  ```

- [ ] **Step 3**: lietorch installed (CRITICAL - must be first!)
  ```bash
  python -c "import lietorch; print('lietorch OK')"
  # Should print: lietorch OK
  ```

- [ ] **Step 4**: thirdparty/in3d installed
  ```bash
  python -c "import in3d; print('in3d OK')"
  # Should print: in3d OK
  ```

- [ ] **Step 5**: Main package installed
  ```bash
  pip show Splatt3R-SLAM
  # Should show package info
  ```

- [ ] **Step 6**: Splatt3R dependencies installed
  ```bash
  python -c "import lightning, lpips, omegaconf; print('Dependencies OK')"
  # Should print: Dependencies OK
  ```

- [ ] **Step 7**: diff-gaussian-rasterization installed
  ```bash
  python -c "import diff_gaussian_rasterization; print('Gaussian rasterization OK')"
  # Should print: Gaussian rasterization OK (or may not be needed if not using rendering)
  ```

## Import Verification

Run each of these commands. All should complete without errors:

### Core Dependencies
- [ ] `python -c "import numpy; print('numpy OK')"`
- [ ] `python -c "import torch; print('torch OK')"`
- [ ] `python -c "import cv2; print('cv2 OK')"`
- [ ] `python -c "import PIL; print('PIL OK')"`
- [ ] `python -c "import yaml; print('yaml OK')"`
- [ ] `python -c "import tqdm; print('tqdm OK')"`
- [ ] `python -c "import einops; print('einops OK')"`

### SLAM Dependencies
- [ ] `python -c "import lietorch; print('lietorch OK')"`
- [ ] `python -c "import plyfile; print('plyfile OK')"`

### Splatt3R Dependencies
- [ ] `python -c "import lightning; print('lightning OK')"`
- [ ] `python -c "import lpips; print('lpips OK')"`
- [ ] `python -c "import omegaconf; print('omegaconf OK')"`
- [ ] `python -c "from huggingface_hub import hf_hub_download; print('huggingface_hub OK')"`

### Package Imports
- [ ] `python -c "import mast3r_slam; print('mast3r_slam OK')"`
- [ ] `python -c "import splatt3r_slam; print('splatt3r_slam OK')"`
- [ ] `python -c "import splatt3r_core; print('splatt3r_core OK')"`

### Specific Module Imports
- [ ] `python -c "from splatt3r_slam.config import load_config; print('config OK')"`
- [ ] `python -c "from splatt3r_slam.global_opt import FactorGraph; print('global_opt OK')"`
- [ ] `python -c "from splatt3r_slam.splatt3r_utils import load_splatt3r; print('splatt3r_utils OK')"`
- [ ] `python -c "from splatt3r_slam.tracker import FrameTracker; print('tracker OK')"`
- [ ] `python -c "from splatt3r_core.main import MAST3RGaussians; print('MAST3RGaussians OK')"`

## Functional Tests

### Test Dataset Download (Optional)
- [ ] Download TUM dataset
  ```bash
  bash ./scripts/download_tum.sh
  ls datasets/tum/  # Should show downloaded datasets
  ```

### Test Main Script Loading
- [ ] Test script loads without errors
  ```bash
  python -c "import main_splatt3r; print('main_splatt3r loads OK')"
  # Should print: main_splatt3r loads OK
  ```

### Test Model Loading (Requires Internet)
- [ ] Test Splatt3R checkpoint download
  ```bash
  python -c "from splatt3r_slam.splatt3r_utils import load_splatt3r; model = load_splatt3r(device='cpu'); print('Model loaded successfully')"
  # Should download checkpoint and print: Model loaded successfully
  # This may take a few minutes on first run (downloads ~150MB)
  ```

## Full System Test

If all above checks pass, try running the full system:

- [ ] Run Splatt3R-SLAM on test dataset
  ```bash
  python main_splatt3r.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml --no-viz
  ```
  
  Expected output:
  - "Loading configuration..."
  - "Loading dataset..."
  - "Downloading Splatt3R checkpoint..." (first time only)
  - "Loading Splatt3R model..."
  - "FPS: ..." (processing frames)
  - "done"

## Troubleshooting

If any check fails, see:
- **TROUBLESHOOTING.md** - Common issues and solutions
- **INSTALLATION_FIX.md** - Complete installation fix details
- **README.md** - Official installation instructions

## Quick Fix Commands

If something is missing:

```bash
# Reinstall lietorch
pip uninstall lietorch -y
pip install git+https://github.com/princeton-vl/lietorch.git

# Reinstall main package
pip uninstall Splatt3R-SLAM -y
pip install --no-build-isolation -e .

# Install missing dependencies
pip install Pillow opencv-python tqdm pyyaml
pip install lightning lpips omegaconf huggingface_hub gitpython
```

## Success Criteria

✅ **Installation is successful when**:
1. All import verification commands work
2. `python -c "from splatt3r_slam.splatt3r_utils import load_splatt3r"` works
3. Main script loads: `python -c "import main_splatt3r"`
4. (Optional) Full system runs on test dataset

## Need Help?

If you've followed all steps and still have issues:
1. Check **TROUBLESHOOTING.md** for your specific error
2. Verify installation order (lietorch must be first!)
3. Create a GitHub issue with:
   - Your full error message
   - Output of `pip list`
   - Output of `python --version`
   - Output of `conda env list`

# Installation Troubleshooting Guide

This document helps resolve common installation issues with Splatt3R-SLAM.

## Common Import Errors

### Error: `ModuleNotFoundError: No module named 'lietorch'`

**Cause**: lietorch is not installed or was not installed before the main package.

**Solution**:
```bash
pip install git+https://github.com/princeton-vl/lietorch.git
```

lietorch must be installed **before** installing the main Splatt3R-SLAM package.

### Error: `ModuleNotFoundError: No module named 'PIL'`

**Cause**: Pillow is not installed.

**Solution**:
```bash
pip install Pillow
```

### Error: `ModuleNotFoundError: No module named 'cv2'`

**Cause**: OpenCV is not installed.

**Solution**:
```bash
pip install opencv-python
```

### Error: `ModuleNotFoundError: No module named 'lightning'`

**Cause**: PyTorch Lightning is not installed (needed for Splatt3R).

**Solution**:
```bash
pip install lightning lpips omegaconf huggingface_hub gitpython
```

## Correct Installation Order

The installation order is critical. Follow these steps **in order**:

### 1. Create and Activate Environment
```bash
conda create -n splatt3r-slam python=3.11
conda activate splatt3r-slam
```

### 2. Install PyTorch (matching your CUDA version)
```bash
# Example for CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Clone Repository
```bash
git clone https://github.com/Looong01/Splatt3R-SLAM.git --recursive
cd Splatt3R-SLAM/
```

### 4. Install lietorch FIRST (Critical!)
```bash
pip install git+https://github.com/princeton-vl/lietorch.git
```

### 5. Install thirdparty dependencies
```bash
pip install -e thirdparty/in3d
```

### 6. Install main package
```bash
pip install --no-build-isolation -e .
```

### 7. Install Splatt3R dependencies
```bash
pip install lightning lpips omegaconf huggingface_hub gitpython
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

### 8. (Optional) Install torchcodec
```bash
pip install torchcodec==0.1
```

## Verification

After installation, verify all imports work:

```bash
python -c "import lietorch; print('lietorch OK')"
python -c "import PIL; print('PIL OK')"
python -c "import cv2; print('cv2 OK')"
python -c "import lightning; print('lightning OK')"
python -c "from splatt3r_slam.splatt3r_utils import load_splatt3r; print('splatt3r_slam OK')"
```

All should print "OK" without errors.

## CUDA Backend Compilation Issues

### Error during `pip install --no-build-isolation -e .`

If you get CUDA compilation errors:

1. **Check CUDA is available**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True`.

2. **Check CUDA version matches**:
   ```bash
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```
   These should match (or be compatible).

3. **If no GPU available**: The backend will skip CUDA compilation, which is fine for CPU-only testing.

## Submodule Issues

### Error: Missing files in `thirdparty/`

**Cause**: Repository not cloned with `--recursive` flag.

**Solution**:
```bash
git submodule update --init --recursive
```

## Package Not Found: mast3r_slam_backends

This is a compiled C++/CUDA extension. If installation failed:

1. Check that you have a C++ compiler installed
2. For CUDA features, ensure CUDA toolkit is installed
3. Try reinstalling:
   ```bash
   pip uninstall mast3r-slam -y
   pip install --no-build-isolation -e .
   ```

## Still Having Issues?

1. **Check Python version**: Must be Python 3.11
   ```bash
   python --version
   ```

2. **Check you're in the right environment**:
   ```bash
   conda env list
   # Should show (splatt3r-slam) as active
   ```

3. **Reinstall from scratch**:
   ```bash
   conda deactivate
   conda remove -n splatt3r-slam --all
   # Then start from step 1 above
   ```

4. **Create an issue on GitHub**: Include:
   - Full error traceback
   - Output of `python --version`
   - Output of `pip list`
   - Output of `nvcc --version`
   - Output of `python -c "import torch; print(torch.version.cuda)"`

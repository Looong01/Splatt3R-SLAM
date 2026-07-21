#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, ROCM_HOME
import os
import torch

_dir = os.path.dirname(os.path.abspath(__file__))


def _select_backend() -> str:
    requested = os.environ.get("DIFFGAUSS_GPU_BACKEND", "auto").strip().lower()
    if requested not in {"auto", "cuda", "rocm"}:
        raise RuntimeError(
            f"Invalid DIFFGAUSS_GPU_BACKEND='{requested}', expected one of ['auto', 'cuda', 'rocm']"
        )

    has_rocm = bool(getattr(torch.version, "hip", None)) and bool(ROCM_HOME)
    has_cuda = bool(getattr(torch.version, "cuda", None))

    if requested == "auto":
        if has_rocm:
            return "rocm"
        if has_cuda:
            return "cuda"
        raise RuntimeError(
            "Neither ROCm nor CUDA PyTorch build detected; cannot compile diff_gaussian_rasterization."
        )

    if requested == "rocm" and not has_rocm:
        raise RuntimeError(
            "DIFFGAUSS_GPU_BACKEND=rocm requires ROCm-enabled PyTorch and ROCM_HOME."
        )
    if requested == "cuda" and not has_cuda:
        raise RuntimeError("DIFFGAUSS_GPU_BACKEND=cuda requires CUDA-enabled PyTorch.")
    return requested


backend = _select_backend()

glm_include = os.path.join(_dir, "third_party/glm/")

if backend == "cuda":
    extra_nvcc = ["-I" + glm_include]
else:
    # ROCm: PyTorch hipify translates the "nvcc" key for hipcc
    rocm_arch = os.environ.get(
        "PYTORCH_ROCM_ARCH", "gfx90a;gfx942;gfx1100;gfx1101;gfx1200"
    )
    os.environ.setdefault("PYTORCH_ROCM_ARCH", rocm_arch)
    extra_nvcc = ["-I" + glm_include, "-std=c++17"]

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={"nvcc": extra_nvcc, "cxx": ["-O3"]},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

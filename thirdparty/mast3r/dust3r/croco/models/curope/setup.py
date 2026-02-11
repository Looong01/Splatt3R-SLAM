from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME

import os
import torch


def _select_backend() -> str:
    requested = os.environ.get("CUROPE_GPU_BACKEND", "auto").strip().lower()
    if requested not in {"auto", "cuda", "rocm"}:
        raise RuntimeError(
            f"Invalid CUROPE_GPU_BACKEND='{requested}', expected one of ['auto', 'cuda', 'rocm']"
        )

    has_rocm = bool(getattr(torch.version, "hip", None)) and bool(ROCM_HOME)
    has_cuda = bool(getattr(torch.version, "cuda", None))

    if requested == "auto":
        if has_rocm:
            return "rocm"
        if has_cuda:
            return "cuda"
        raise RuntimeError(
            "Neither ROCm nor CUDA PyTorch build detected; cannot compile curope extension."
        )

    if requested == "rocm" and not has_rocm:
        raise RuntimeError(
            "CUROPE_GPU_BACKEND=rocm requires ROCm-enabled PyTorch and ROCM_HOME."
        )
    if requested == "cuda" and not has_cuda:
        raise RuntimeError("CUROPE_GPU_BACKEND=cuda requires CUDA-enabled PyTorch.")
    return requested


backend = _select_backend()

if backend == "cuda":
    all_gpu_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()
    extra_nvcc = ["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_gpu_archs
else:
    rocm_arch = os.environ.get(
        "PYTORCH_ROCM_ARCH", "gfx90a;gfx942;gfx1100;gfx1101;gfx1200"
    )
    os.environ.setdefault("PYTORCH_ROCM_ARCH", rocm_arch)
    extra_nvcc = ["-O3", "--use_fast_math", "-std=c++17"]

setup(
    name="curope",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=["curope.cpp", "kernels.cu"],
            extra_compile_args={"nvcc": extra_nvcc, "cxx": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

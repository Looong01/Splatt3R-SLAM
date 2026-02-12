from setuptools import setup

import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME

ROOT = os.path.dirname(os.path.abspath(__file__))


SUPPORTED_GPU_BACKENDS = {"auto", "cuda", "rocm", "cpu"}


def _can_build_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None)) and bool(ROCM_HOME)


def _can_build_cuda() -> bool:
    return bool(getattr(torch.version, "cuda", None))


def _detect_gpu_backend() -> str:
    requested = os.environ.get("SPLATT3R_GPU_BACKEND", "auto").strip().lower()
    if requested not in SUPPORTED_GPU_BACKENDS:
        raise RuntimeError(
            "Invalid SPLATT3R_GPU_BACKEND='{}', expected one of {}".format(
                requested, sorted(SUPPORTED_GPU_BACKENDS)
            )
        )

    has_rocm_torch = bool(getattr(torch.version, "hip", None))
    has_cuda_torch = bool(getattr(torch.version, "cuda", None))

    if requested == "auto":
        if has_rocm_torch and _can_build_rocm():
            return "rocm"
        if has_cuda_torch and _can_build_cuda():
            return "cuda"
        return "cpu"

    return requested


backend = _detect_gpu_backend()

# Backend paths
backend_dir = None
if os.path.exists(os.path.join(ROOT, "splatt3r_slam/backend")):
    backend_dir = "splatt3r_slam"

if backend_dir is None:
    print("Warning: No backend directory found in splatt3r_slam")
    ext_modules = []
else:
    include_dirs = [
        os.path.join(ROOT, f"{backend_dir}/backend/include"),
        os.path.join(ROOT, "thirdparty/eigen"),
    ]

    cpp_sources = [f"{backend_dir}/backend/src/gn.cpp"]
    gpu_sources = [
        f"{backend_dir}/backend/src/gn_kernels.cu",
        f"{backend_dir}/backend/src/matching_kernels.cu",
    ]

    extra_compile_args = {
        "cxx": ["-O3"],
    }

    ext_modules = []
    if backend == "cuda":
        if not _can_build_cuda():
            raise RuntimeError(
                "SPLATT3R_GPU_BACKEND=cuda requires a CUDA-enabled PyTorch build."
            )
        extra_compile_args["nvcc"] = [
            "-O3",
            "-gencode=arch=compute_60,code=sm_60",
            "-gencode=arch=compute_61,code=sm_61",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
        ]
        ext_modules = [
            CUDAExtension(
                "mast3r_slam_backends",
                include_dirs=include_dirs,
                sources=cpp_sources + gpu_sources,
                extra_compile_args=extra_compile_args,
            )
        ]
        print("Building mast3r_slam_backends with CUDA backend")
    elif backend == "rocm":
        if not bool(getattr(torch.version, "hip", None)):
            raise RuntimeError(
                "SPLATT3R_GPU_BACKEND=rocm requires a ROCm-enabled PyTorch build."
            )
        if not ROCM_HOME:
            raise RuntimeError(
                "ROCm toolkit not found (ROCM_HOME is unset). Install ROCm 7.1 and export ROCM_HOME, "
                "or set SPLATT3R_GPU_BACKEND=cpu to skip the extension build."
            )

        # PyTorch hipify pipeline maps 'nvcc' flags to hipcc flags.
        rocm_arch = os.environ.get(
            "PYTORCH_ROCM_ARCH", "gfx90a;gfx942;gfx1100;gfx1101;gfx1200"
        )
        os.environ.setdefault("PYTORCH_ROCM_ARCH", rocm_arch)

        extra_compile_args["nvcc"] = ["-O3", "-std=c++17"]
        ext_modules = [
            CUDAExtension(
                "mast3r_slam_backends",
                include_dirs=include_dirs,
                sources=cpp_sources + gpu_sources,
                extra_compile_args=extra_compile_args,
            )
        ]
        print(
            "Building mast3r_slam_backends with ROCm backend "
            f"(ROCm {torch.version.hip}, PYTORCH_ROCM_ARCH={os.environ['PYTORCH_ROCM_ARCH']})"
        )
    elif backend == "cpu":
        # No CPU implementation for gauss-newton CUDA kernels in this project.
        # Keep installation possible by skipping extension build explicitly.
        ext_modules = []
        print("Skipping mast3r_slam_backends build (SPLATT3R_GPU_BACKEND=cpu)")
    else:
        raise AssertionError(f"Unhandled backend: {backend}")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
)

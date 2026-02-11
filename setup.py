from pathlib import Path
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
has_cuda = torch.cuda.is_available()

# Backend paths - check both mast3r_slam and splatt3r_slam
backend_dir = None
if os.path.exists(os.path.join(ROOT, "splatt3r_slam/backend")):
    backend_dir = "splatt3r_slam"
elif os.path.exists(os.path.join(ROOT, "mast3r_slam/backend")):
    backend_dir = "mast3r_slam"

if backend_dir is None:
    print("Warning: No backend directory found in mast3r_slam or splatt3r_slam")
    ext_modules = []
else:
    include_dirs = [
        os.path.join(ROOT, f"{backend_dir}/backend/include"),
        os.path.join(ROOT, "thirdparty/eigen"),
    ]

    sources = [
        f"{backend_dir}/backend/src/gn.cpp",
    ]
    extra_compile_args = {
        "cores": ["j8"],
        "cxx": ["-O3"],
    }

    if has_cuda:
        from torch.utils.cpp_extension import CUDAExtension

        sources.append(f"{backend_dir}/backend/src/gn_kernels.cu")
        sources.append(f"{backend_dir}/backend/src/matching_kernels.cu")
        extra_compile_args["nvcc"] = [
            "-O3",
            "-gencode=arch=compute_60,code=sm_60",
            "-gencode=arch=compute_61,code=sm_61",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ]
        ext_modules = [
            CUDAExtension(
                "mast3r_slam_backends",
                include_dirs=include_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
            )
        ]
    else:
        print("CUDA not found, cannot compile backend!")
        ext_modules = []

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
)

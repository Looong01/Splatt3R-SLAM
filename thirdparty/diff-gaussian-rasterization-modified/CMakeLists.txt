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

cmake_minimum_required(VERSION 3.20)

# Detect ROCm/HIP vs CUDA
if(DEFINED ENV{ROCM_PATH} OR DEFINED ENV{HIP_PATH})
    # ROCm / HIP build
    if(NOT DEFINED HIP_PATH)
        if(DEFINED ENV{HIP_PATH})
            set(HIP_PATH $ENV{HIP_PATH})
        elseif(DEFINED ENV{ROCM_PATH})
            set(HIP_PATH "$ENV{ROCM_PATH}/hip")
        else()
            set(HIP_PATH "/opt/rocm/hip")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH})
    find_package(HIP QUIET)
endif()

if(HIP_FOUND)
    project(DiffRast LANGUAGES CXX HIP)
    set(CMAKE_HIP_STANDARD 17)
    add_compile_definitions(USE_ROCM __HIP_PLATFORM_AMD__)
else()
    project(DiffRast LANGUAGES CUDA CXX)
    set(CMAKE_CUDA_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
)

if(HIP_FOUND)
    set_source_files_properties(
        cuda_rasterizer/backward.cu
        cuda_rasterizer/forward.cu
        cuda_rasterizer/rasterizer_impl.cu
        PROPERTIES LANGUAGE HIP
    )
    target_compile_definitions(CudaRasterizer PRIVATE USE_ROCM __HIP_PLATFORM_AMD__)
    find_package(hipcub REQUIRED)
    target_link_libraries(CudaRasterizer PRIVATE hip::hipcub)
else()
    set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")
endif()

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm
    $<$<NOT:$<BOOL:${HIP_FOUND}>>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
)

cmake_minimum_required(VERSION 3.20)

project(cuda_image_processing LANGUAGES CXX CUDA)

# Language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target NVIDIA GeForce RTX 3050 Laptop GPU (Ampere - Compute Capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Smoke test executable
add_executable(smoke_test src/smoke_test.cu)

set_target_properties(smoke_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

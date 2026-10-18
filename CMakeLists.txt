cmake_minimum_required(VERSION 3.20)

project(cuda_image_processing LANGUAGES CXX CUDA)

# Language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configurable CUDA Architecture (default to sm_86 for NVIDIA GeForce RTX 3050 Laptop GPU)
if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "Target CUDA Architecture (default: 86)" FORCE)
endif()
message(STATUS "Configured CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core Library: cuda_image_proc
add_library(cuda_image_proc STATIC
    src/cpu/image_io.cpp
    src/cpu/cpu_ops.cpp
    src/cuda/cuda_ops.cu
)

target_include_directories(cuda_image_proc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(cuda_image_proc PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# Main Executable: RGB to Grayscale Conversion (CLI options: --mode, --input, --output)
add_executable(cuda_image_app
    src/main.cpp
)

target_link_libraries(cuda_image_app PRIVATE
    cuda_image_proc
)

# Test Executable: RGB-to-Grayscale CPU Reference vs. CUDA Validation Suite
add_executable(test_grayscale
    tests/test_grayscale.cpp
)

target_link_libraries(test_grayscale PRIVATE
    cuda_image_proc
)

# Test Executable: Mean Blur (Box Filter) CPU Reference vs. Naive CUDA Validation Suite
add_executable(test_mean_blur
    tests/test_mean_blur.cpp
)

target_link_libraries(test_mean_blur PRIVATE
    cuda_image_proc
)

# Benchmark Executable: Profiling Naive vs. Tiled Shared Memory (8x8, 16x16, 32x32)
add_executable(benchmark_mean_blur
    benchmarks/benchmark_mean_blur.cpp
)

target_link_libraries(benchmark_mean_blur PRIVATE
    cuda_image_proc
)

# Test Executable: 3x3 Gaussian Blur CPU Reference vs. Naive CUDA vs. Tiled Suite
add_executable(test_gaussian_blur
    tests/test_gaussian_blur.cpp
)

target_link_libraries(test_gaussian_blur PRIVATE
    cuda_image_proc
)

# Benchmark Executable: Profiling 3x3 Gaussian Blur Naive vs. Tiled Shared Memory
add_executable(benchmark_gaussian_blur
    benchmarks/benchmark_gaussian_blur.cpp
)

target_link_libraries(benchmark_gaussian_blur PRIVATE
    cuda_image_proc
)

# Test Executable: Sobel Edge Detection CPU Reference vs. Naive CUDA vs. Tiled Suite
add_executable(test_sobel
    tests/test_sobel.cpp
)

target_link_libraries(test_sobel PRIVATE
    cuda_image_proc
)

# Benchmark Executable: Profiling Sobel Edge Detection Naive vs. Tiled Shared Memory
add_executable(benchmark_sobel
    benchmarks/benchmark_sobel.cpp
)

target_link_libraries(benchmark_sobel PRIVATE
    cuda_image_proc
)

# Unified Reusable Benchmark Runner
add_executable(benchmark_runner
    benchmarks/benchmark_runner.cpp
)

target_link_libraries(benchmark_runner PRIVATE
    cuda_image_proc
)

# Kernel Fusion Benchmark Executable
add_executable(benchmark_fusion
    benchmarks/benchmark_fusion.cpp
)

target_link_libraries(benchmark_fusion PRIVATE
    cuda_image_proc
)


# Profiler Target Executable for Nsight Systems and Nsight Compute
add_executable(profiler_target
    benchmarks/profiler_target.cu
)

target_link_libraries(profiler_target PRIVATE
    cuda_image_proc
)

# Preserved Smoke Test Executable
add_executable(smoke_test
    src/smoke_test.cu
)

set_target_properties(smoke_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Robustness & Correctness Test Suite Executable
add_executable(test_robustness
    tests/test_robustness.cpp
)

target_link_libraries(test_robustness PRIVATE
    cuda_image_proc
)

# Kernel Fusion Test Suite Executable
add_executable(test_fusion
    tests/test_fusion.cpp
)

target_link_libraries(test_fusion PRIVATE
    cuda_image_proc
)

# Enable CTest
enable_testing()
add_test(NAME TestGrayscaleSuite COMMAND test_grayscale)
add_test(NAME TestMeanBlurSuite COMMAND test_mean_blur)
add_test(NAME TestGaussianBlurSuite COMMAND test_gaussian_blur)
add_test(NAME TestSobelSuite COMMAND test_sobel)
add_test(NAME TestRobustnessSuite COMMAND test_robustness)
add_test(NAME TestFusionSuite COMMAND test_fusion)





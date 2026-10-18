#include <iostream>
#include <cuda_runtime.h>
#include "cuda_image_proc/image.hpp"
#include "cuda_image_proc/device_image.cuh"
#include "cuda_image_proc/memory_transfer.cuh"
#include "cuda_image_proc/cuda_ops.cuh"

using namespace cuda_image_proc;
using namespace cuda_image_proc::cuda;

int main() {
    std::cout << "Profiling Target: Initializing 1024x1024 test image...\n";
    const int w = 1024;
    const int h = 1024;

    Image h_gray(w, h, 1);
    for (int y = 0; y < h; ++y) {
        uint8_t* row = h_gray.row_ptr(y);
        for (int x = 0; x < w; ++x) {
            row[x] = static_cast<uint8_t>((x * 3 + y * 7) % 256);
        }
    }

    CudaStream stream;
    DeviceImage d_src = to_device(h_gray, stream);

    std::cout << "Running kernels for Nsight profiling...\n";

    // 1. Mean Blur Naive
    DeviceImage d_mb_naive = cuda::mean_blur_naive(d_src, 3, stream);

    // 2. Mean Blur Tiled (16x16)
    DeviceImage d_mb_tiled = cuda::mean_blur_tiled_3x3(d_src, TileConfig::TILE_16x16, stream);

    // 3. Gaussian Blur Naive
    DeviceImage d_gb_naive = cuda::gaussian_blur_naive_3x3(d_src, stream);

    // 4. Gaussian Blur Tiled (16x16)
    DeviceImage d_gb_tiled = cuda::gaussian_blur_tiled_3x3(d_src, TileConfig::TILE_16x16, stream);

    // 5. Sobel Naive
    DeviceImage d_sobel_naive = cuda::sobel_edge_detect_naive(d_src, stream);

    // 6. Sobel Tiled (8x8)
    DeviceImage d_sobel_t8 = cuda::sobel_edge_detect_tiled(d_src, TileConfig::TILE_8x8, stream);

    // 7. Sobel Tiled (16x16)
    DeviceImage d_sobel_t16 = cuda::sobel_edge_detect_tiled(d_src, TileConfig::TILE_16x16, stream);

    // 8. Sobel Tiled (32x32)
    DeviceImage d_sobel_t32 = cuda::sobel_edge_detect_tiled(d_src, TileConfig::TILE_32x32, stream);

    // 9. Kernel Fusion Exploration: RGB Source
    Image h_rgb(w, h, 3);
    for (int y = 0; y < h; ++y) {
        uint8_t* row = h_rgb.row_ptr(y);
        for (int x = 0; x < w * 3; ++x) {
            row[x] = static_cast<uint8_t>((x * 5 + y * 11) % 256);
        }
    }
    DeviceImage d_rgb = to_device(h_rgb, stream);

    // 10. Baseline Pipeline (Separate rgbToGrayscale + meanBlurTiled)
    DeviceImage d_base_pipe = cuda::rgb_to_mean_blur_baseline(d_rgb, 3, stream);

    // 11. Fused Tiled Shared Memory (16x16)
    DeviceImage d_fused_t16 = cuda::rgb_to_mean_blur_fused(d_rgb, TileConfig::TILE_16x16, stream);

    // 12. Fused Naive (Global Memory only)
    DeviceImage d_fused_naive = cuda::rgb_to_mean_blur_fused_naive(d_rgb, 3, stream);

    stream.synchronize();
    std::cout << "All kernels executed successfully.\n";

    return 0;
}

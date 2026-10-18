#pragma once

#include <cuda_runtime.h>
#include <string>
#include "cuda_image_proc/image.hpp"
#include "cuda_image_proc/device_image.cuh"

namespace cuda_image_proc {
namespace cuda {

struct DeviceInfo {
    std::string name;
    int major{0};
    int minor{0};
    size_t total_memory_bytes{0};
    int sm_count{0};
    int max_threads_per_block{0};
    int warp_size{0};
};

/**
 * @brief Retrieves hardware information about a specified CUDA device.
 */
DeviceInfo get_device_info(int device_id = 0);

/**
 * @brief Computes standard 2D grid dimensions from block dimensions and image resolution.
 */
inline dim3 calculate_grid_dim(dim3 block, int width, int height) {
    return dim3(
        (static_cast<unsigned int>(width) + block.x - 1) / block.x,
        (static_cast<unsigned int>(height) + block.y - 1) / block.y,
        1
    );
}

/**
 * @brief Converts an RGB DeviceImage (3 channels) to Grayscale DeviceImage (1 channel) using a naive CUDA kernel.
 *
 * Uses standard luminance formula: Y = 0.299 R + 0.587 G + 0.114 B
 * Rounding: roundf (nearest, half away from zero)
 * Clamping: [0, 255]
 *
 * @param d_rgb Input RGB device image.
 * @param stream Optional CUDA stream for asynchronous execution.
 * @return DeviceImage Output 1-channel Grayscale device image.
 * @throws std::invalid_argument If image is empty or not 3 channels.
 */
DeviceImage rgb_to_grayscale(const DeviceImage& d_rgb, cudaStream_t stream = 0);

/**
 * @brief Convenience helper: copies host RGB image to GPU, converts to grayscale via CUDA,
 * and copies back to a HostImage.
 */
HostImage rgb_to_grayscale(const HostImage& h_rgb, cudaStream_t stream = 0);

/**
 * @brief Applies mean blur (box filter) using a naive 2D CUDA kernel with global-memory accesses.
 *
 * Border policy: Clamp-to-edge (coordinates outside [0, width-1] or [0, height-1]
 * are clamped to the nearest boundary pixel).
 *
 * Accumulation: 32-bit unsigned integer accumulation (sum of uint8 values)
 * followed by floating-point division by filter area (filter_size * filter_size).
 *
 * Rounding and clamping: roundf (half away from zero / nearest), clamped to [0, 255].
 *
 * Thread Indexing: 2D block (16x16) and 2D grid covering the entire image with
 * safe boundary guards for irregular dimensions.
 *
 * @param d_src Input device image.
 * @param filter_size Odd filter dimension K (e.g. 3 for 3x3, 5 for 5x5).
 * @param stream Optional CUDA stream for asynchronous execution.
 * @return DeviceImage Output blurred device image with matching dimensions and channels.
 * @throws std::invalid_argument If image is empty or filter_size is not a positive odd integer.
 */
DeviceImage mean_blur_naive(const DeviceImage& d_src, int filter_size = 3, cudaStream_t stream = 0);

/**
 * @brief Convenience helper: copies host image to GPU, applies naive CUDA mean blur,
 * and copies the result back to a HostImage.
 */
HostImage mean_blur_naive(const HostImage& h_src, int filter_size = 3, cudaStream_t stream = 0);

enum class TileConfig {
    TILE_8x8,
    TILE_16x16,
    TILE_32x32
};

struct TiledKernelStats {
    size_t shared_mem_bytes{0};
    int block_dim_x{0};
    int block_dim_y{0};
};

/**
 * @brief Retrieves static shared-memory usage and block dimensions for a tile configuration.
 */
TiledKernelStats get_tiled_3x3_kernel_stats(TileConfig config);

/**
 * @brief Applies 3x3 mean blur using a shared-memory tiled CUDA kernel.
 *
 * Loads an image tile plus 1-pixel halo into shared memory with clamp-to-edge border policy.
 * Cooperatively loads all tile and halo elements using all threads in the block.
 * Synchronizes all threads via __syncthreads() before computing the 3x3 box average from shared memory.
 * Preserves the exact same border, rounding, and output clamping policies as the CPU reference.
 *
 * @param d_src Input 1-channel grayscale device image.
 * @param config Tile dimension (TILE_8x8, TILE_16x16, or TILE_32x32).
 * @param stream Optional CUDA stream.
 * @return DeviceImage Blurred 1-channel grayscale device image.
 * @throws std::invalid_argument If image is empty or channels != 1.
 */
DeviceImage mean_blur_tiled_3x3(const DeviceImage& d_src,
                                TileConfig config = TileConfig::TILE_16x16,
                                cudaStream_t stream = 0);

/**
 * @brief Convenience helper: copies host image to GPU, applies tiled 3x3 CUDA mean blur,
 * and copies back to HostImage.
 */
HostImage mean_blur_tiled_3x3(const HostImage& h_src,
                              TileConfig config = TileConfig::TILE_16x16,
                              cudaStream_t stream = 0);

/**
 * @brief Retrieves static shared-memory usage and block dimensions for 3x3 Gaussian tiled kernel.
 */
TiledKernelStats get_tiled_gaussian_3x3_kernel_stats(TileConfig config);

/**
 * @brief Applies 3x3 Gaussian blur using a naive CUDA kernel with global-memory accesses.
 *
 * Kernel weights:
 * [ 1  2  1 ]
 * [ 2  4  2 ] / 16.0f
 * [ 1  2  1 ]
 *
 * Border policy: Clamp-to-edge.
 * Accumulation: Floating-point accumulation with normalized weights.
 * Rounding & clamping: roundf(sum), clamped to [0, 255].
 *
 * @param d_src Input device image.
 * @param stream Optional CUDA stream.
 * @return DeviceImage Output blurred device image.
 * @throws std::invalid_argument If image is empty.
 */
DeviceImage gaussian_blur_naive_3x3(const DeviceImage& d_src, cudaStream_t stream = 0);

HostImage gaussian_blur_naive_3x3(const HostImage& h_src, cudaStream_t stream = 0);

/**
 * @brief Applies Gaussian blur (3x3 or 5x5) using a naive CUDA kernel with global-memory accesses.
 *
 * @param d_src Input device image.
 * @param filter_size Filter dimension (3 or 5, default: 3).
 * @param stream Optional CUDA stream.
 * @return DeviceImage Output blurred device image.
 * @throws std::invalid_argument If image is empty or filter_size is not 3 or 5.
 */
DeviceImage gaussian_blur_naive(const DeviceImage& d_src, int filter_size = 3, cudaStream_t stream = 0);

HostImage gaussian_blur_naive(const HostImage& h_src, int filter_size = 3, cudaStream_t stream = 0);


/**
 * @brief Applies 3x3 Gaussian blur using a shared-memory tiled CUDA kernel.
 *
 * Loads an image tile plus 1-pixel halo into shared memory with clamp-to-edge border policy.
 * Cooperatively loads all tile and halo elements using all threads in the block.
 * Synchronizes via __syncthreads() before computing the weighted 3x3 Gaussian blur from shared memory.
 * Preserves the exact same clamp-to-edge, floating-point accumulation, rounding, and clamping.
 *
 * @param d_src Input 1-channel grayscale device image.
 * @param config Tile dimension (TILE_8x8, TILE_16x16, or TILE_32x32).
 * @param stream Optional CUDA stream.
 * @return DeviceImage Output blurred device image.
 * @throws std::invalid_argument If image is empty or channels != 1.
 */
DeviceImage gaussian_blur_tiled_3x3(const DeviceImage& d_src,
                                    TileConfig config = TileConfig::TILE_16x16,
                                    cudaStream_t stream = 0);

/**
 * @brief Convenience helper: copies host image to GPU, applies tiled 3x3 Gaussian blur,
 * and copies back to HostImage.
 */
HostImage gaussian_blur_tiled_3x3(const HostImage& h_src,
                                  TileConfig config = TileConfig::TILE_16x16,
                                  cudaStream_t stream = 0);

/**
 * @brief Retrieves static shared-memory usage and block dimensions for Sobel tiled kernel.
 */
TiledKernelStats get_tiled_sobel_kernel_stats(TileConfig config);

/**
 * @brief Applies Sobel edge detection using a naive CUDA kernel with global-memory accesses.
 *
 * Gx = [-1 0 1; -2 0 2; -1 0 1]
 * Gy = [-1 -2 -1; 0 0 0; 1 2 1]
 * magnitude = sqrt(Gx*Gx + Gy*Gy)
 * Rounded using roundf and clamped to [0, 255].
 *
 * @param d_src Input device image (1-channel grayscale or multi-channel).
 * @param stream Optional CUDA stream.
 * @return DeviceImage Output 1-channel edge magnitude device image.
 * @throws std::invalid_argument If image is empty.
 */
DeviceImage sobel_edge_detect_naive(const DeviceImage& d_src, cudaStream_t stream = 0);

/**
 * @brief Convenience helper: copies host image to GPU, applies naive Sobel edge detection,
 * and copies back to HostImage.
 */
HostImage sobel_edge_detect_naive(const HostImage& h_src, cudaStream_t stream = 0);

/**
 * @brief Applies Sobel edge detection using a shared-memory tiled CUDA kernel.
 *
 * Loads an image tile plus 1-pixel halo into shared memory with clamp-to-edge border policy.
 * Cooperatively loads all tile and halo elements using all threads in the block.
 * Synchronizes via __syncthreads() before computing the 3x3 Sobel gradients and magnitude from shared memory.
 * Preserves the exact same clamp-to-edge border policy, floating-point accumulation, rounding, and clamping.
 *
 * @param d_src Input 1-channel grayscale device image.
 * @param config Tile dimension (TILE_8x8, TILE_16x16, or TILE_32x32).
 * @param stream Optional CUDA stream.
 * @return DeviceImage Output 1-channel edge magnitude device image.
 * @throws std::invalid_argument If image is empty or channels != 1.
 */
DeviceImage sobel_edge_detect_tiled(const DeviceImage& d_src,
                                    TileConfig config = TileConfig::TILE_16x16,
                                    cudaStream_t stream = 0);

/**
 * @brief Convenience helper: copies host image to GPU, applies tiled Sobel edge detection,
 * and copies back to HostImage.
 */
HostImage sobel_edge_detect_tiled(const HostImage& h_src,
                                  TileConfig config = TileConfig::TILE_16x16,
                                  cudaStream_t stream = 0);

// ===========================================================================
// Kernel Fusion Exploration: RGB-to-Grayscale + Mean Blur
// ===========================================================================

/**
 * @brief Baseline pipeline: executes separate RGB-to-Grayscale and Mean Blur kernels.
 *
 * Incurs intermediate global-memory allocation and two separate kernel launches.
 *
 * @param d_rgb Input RGB device image (3 channels).
 * @param filter_size Filter size (default: 3 for 3x3).
 * @param stream Optional CUDA stream.
 * @return DeviceImage Output 1-channel blurred grayscale device image.
 */
DeviceImage rgb_to_mean_blur_baseline(const DeviceImage& d_rgb,
                                      int filter_size = 3,
                                      cudaStream_t stream = 0);

HostImage rgb_to_mean_blur_baseline(const HostImage& h_rgb,
                                    int filter_size = 3,
                                    cudaStream_t stream = 0);

/**
 * @brief Fused CUDA implementation: computes grayscale values into shared memory on-the-fly
 * and produces blurred grayscale output in a single kernel launch.
 *
 * Avoids all intermediate global-memory allocations, writes, and reads.
 * Reuses grayscale values across threads via shared memory.
 *
 * @param d_rgb Input RGB device image (3 channels).
 * @param config Tile dimension (TILE_8x8, TILE_16x16, or TILE_32x32). Default: TILE_16x16.
 * @param stream Optional CUDA stream.
 * @return DeviceImage Output 1-channel blurred grayscale device image.
 */
DeviceImage rgb_to_mean_blur_fused(const DeviceImage& d_rgb,
                                   TileConfig config = TileConfig::TILE_16x16,
                                   cudaStream_t stream = 0);

HostImage rgb_to_mean_blur_fused(const HostImage& h_rgb,
                                 TileConfig config = TileConfig::TILE_16x16,
                                 cudaStream_t stream = 0);

/**
 * @brief Fused Naive CUDA implementation: computes grayscale values for the filter neighborhood
 * directly from global memory without shared memory.
 *
 * Demonstrates the trade-off of redundant global-memory recomputation vs. shared-memory staging.
 *
 * @param d_rgb Input RGB device image (3 channels).
 * @param filter_size Filter dimension (e.g. 3 for 3x3).
 * @param stream Optional CUDA stream.
 * @return DeviceImage Output 1-channel blurred grayscale device image.
 */
DeviceImage rgb_to_mean_blur_fused_naive(const DeviceImage& d_rgb,
                                         int filter_size = 3,
                                         cudaStream_t stream = 0);

HostImage rgb_to_mean_blur_fused_naive(const HostImage& h_rgb,
                                       int filter_size = 3,
                                       cudaStream_t stream = 0);

} // namespace cuda
} // namespace cuda_image_proc





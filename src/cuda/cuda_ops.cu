#include "cuda_image_proc/cuda_ops.cuh"
#include "cuda_image_proc/cuda_utils.hpp"
#include "cuda_image_proc/memory_transfer.cuh"
#include <stdexcept>
#include <iostream>

namespace cuda_image_proc {
namespace cuda {

// ---------------------------------------------------------------------------
// Naive CUDA Kernel for RGB-to-Grayscale Conversion
// ---------------------------------------------------------------------------
__global__ void rgbToGrayscaleNaiveKernel(
    const uint8_t* __restrict__ d_rgb,
    size_t rgb_pitch,
    uint8_t* __restrict__ d_gray,
    size_t gray_pitch,
    int width,
    int height) {

    // 2D CUDA thread indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard against threads outside image boundaries
    if (x >= width || y >= height) {
        return;
    }

    // Row pointer computation respecting device pitch
    const uint8_t* in_row = d_rgb + static_cast<size_t>(y) * rgb_pitch;
    uint8_t* out_row = d_gray + static_cast<size_t>(y) * gray_pitch;

    // Interleaved RGB data: [R, G, B]
    int rgb_idx = x * 3;
    float r = static_cast<float>(in_row[rgb_idx + 0]);
    float g = static_cast<float>(in_row[rgb_idx + 1]);
    float b = static_cast<float>(in_row[rgb_idx + 2]);

    // Standard Luminance Approximation: Y = 0.299 R + 0.587 G + 0.114 B
    float gray_val = 0.299f * r + 0.587f * g + 0.114f * b;

    // Rounding and Clamping Policy
    float rounded = roundf(gray_val);
    if (rounded < 0.0f) rounded = 0.0f;
    if (rounded > 255.0f) rounded = 255.0f;

    // Coalesced write: adjacent threads write adjacent bytes in out_row
    out_row[x] = static_cast<uint8_t>(rounded);
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
DeviceInfo get_device_info(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    DeviceInfo info;
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.total_memory_bytes = prop.totalGlobalMem;
    info.sm_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;

    return info;
}

DeviceImage rgb_to_grayscale(const DeviceImage& d_rgb, cudaStream_t stream) {
    if (d_rgb.is_empty()) {
        throw std::invalid_argument("DeviceImage is empty");
    }
    if (d_rgb.channels() != 3) {
        throw std::invalid_argument("rgb_to_grayscale expects 3 channels (RGB), got " +
                                    std::to_string(d_rgb.channels()));
    }

    int width = d_rgb.width();
    int height = d_rgb.height();

    // Allocate 1-channel output on GPU with pitched allocation
    DeviceImage d_gray(width, height, 1);

    // 2D Block and Grid setup (16x16 threads = 256 threads per block)
    dim3 blockDim(16, 16);
    dim3 gridDim = calculate_grid_dim(blockDim, width, height);

    if (stream == 0) {
        rgbToGrayscaleNaiveKernel<<<gridDim, blockDim>>>(
            d_rgb.data(), d_rgb.stride_bytes(),
            d_gray.data(), d_gray.stride_bytes(),
            width, height
        );
    } else {
        rgbToGrayscaleNaiveKernel<<<gridDim, blockDim, 0, stream>>>(
            d_rgb.data(), d_rgb.stride_bytes(),
            d_gray.data(), d_gray.stride_bytes(),
            width, height
        );
    }

    // Check for kernel launch errors
    CUDA_CHECK_LAST();

    if (stream == 0) {
        // Synchronize and check for execution errors
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return d_gray;
}

HostImage rgb_to_grayscale(const HostImage& h_rgb, cudaStream_t stream) {
    // Transfer to GPU
    DeviceImage d_rgb = to_device(h_rgb, stream);

    // Execute kernel
    DeviceImage d_gray = rgb_to_grayscale(d_rgb, stream);

    // Transfer back to Host
    return to_host(d_gray, stream);
}

// ---------------------------------------------------------------------------
// Naive CUDA Kernel for Mean Blur (Box Filter) with Global-Memory Accesses
// ---------------------------------------------------------------------------
__global__ void meanBlurNaiveKernel(
    const uint8_t* __restrict__ d_src,
    size_t src_pitch,
    uint8_t* __restrict__ d_dst,
    size_t dst_pitch,
    int width,
    int height,
    int channels,
    int filter_size) {

    // 2D CUDA thread indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard against threads outside image boundaries
    if (x >= width || y >= height) {
        return;
    }

    int radius = filter_size / 2;
    float num_samples = static_cast<float>(filter_size * filter_size);
    uint8_t* out_row = d_dst + static_cast<size_t>(y) * dst_pitch;

    for (int c = 0; c < channels; ++c) {
        // Sufficiently wide 32-bit unsigned accumulation to prevent integer overflow
        uint32_t sum = 0;

        for (int dy = -radius; dy <= radius; ++dy) {
            // Clamp-to-edge border policy in Y dimension
            int sy = y + dy;
            if (sy < 0) sy = 0;
            else if (sy >= height) sy = height - 1;

            const uint8_t* in_row = d_src + static_cast<size_t>(sy) * src_pitch;

            for (int dx = -radius; dx <= radius; ++dx) {
                // Clamp-to-edge border policy in X dimension
                int sx = x + dx;
                if (sx < 0) sx = 0;
                else if (sx >= width) sx = width - 1;

                // Naive un-cached global-memory read per sample
                sum += in_row[static_cast<size_t>(sx) * channels + c];
            }
        }

        // Float division by total samples in neighborhood
        float avg = static_cast<float>(sum) / num_samples;

        // Output rounding (nearest, half away from zero) and clamping to [0, 255]
        float rounded = roundf(avg);
        if (rounded < 0.0f) rounded = 0.0f;
        else if (rounded > 255.0f) rounded = 255.0f;

        out_row[static_cast<size_t>(x) * channels + c] = static_cast<uint8_t>(rounded);
    }
}

DeviceImage mean_blur_naive(const DeviceImage& d_src, int filter_size, cudaStream_t stream) {
    if (d_src.is_empty()) {
        throw std::invalid_argument("Cannot apply mean_blur_naive to an empty DeviceImage");
    }
    if (filter_size <= 0 || (filter_size % 2 == 0)) {
        throw std::invalid_argument("filter_size must be a positive odd integer (got " +
                                    std::to_string(filter_size) + ")");
    }

    int width = d_src.width();
    int height = d_src.height();
    int channels = d_src.channels();

    DeviceImage d_dst(width, height, channels);

    dim3 blockDim(16, 16);
    dim3 gridDim = calculate_grid_dim(blockDim, width, height);

    if (stream == 0) {
        meanBlurNaiveKernel<<<gridDim, blockDim>>>(
            d_src.data(), d_src.stride_bytes(),
            d_dst.data(), d_dst.stride_bytes(),
            width, height, channels, filter_size
        );
    } else {
        meanBlurNaiveKernel<<<gridDim, blockDim, 0, stream>>>(
            d_src.data(), d_src.stride_bytes(),
            d_dst.data(), d_dst.stride_bytes(),
            width, height, channels, filter_size
        );
    }

    CUDA_CHECK_LAST();

    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return d_dst;
}

HostImage mean_blur_naive(const HostImage& h_src, int filter_size, cudaStream_t stream) {
    DeviceImage d_src = to_device(h_src, stream);
    DeviceImage d_dst = mean_blur_naive(d_src, filter_size, stream);
    return to_host(d_dst, stream);
}

// ---------------------------------------------------------------------------
// Tiled Shared-Memory CUDA Kernel for 3x3 Mean Blur
// ---------------------------------------------------------------------------
template <int TILE_W, int TILE_H>
__global__ void meanBlurTiled3x3Kernel(
    const uint8_t* __restrict__ d_src,
    size_t src_pitch,
    uint8_t* __restrict__ d_dst,
    size_t dst_pitch,
    int width,
    int height) {

    // Shared memory dimensions including 1-pixel halo on all sides
    constexpr int SMEM_W = TILE_W + 2;
    constexpr int SMEM_H = TILE_H + 2;
    constexpr int SMEM_SIZE = SMEM_W * SMEM_H;
    constexpr int NUM_THREADS = TILE_W * TILE_H;

    __shared__ uint8_t s_tile[SMEM_H][SMEM_W];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Tile origin in the output image space
    int tile_origin_x = blockIdx.x * TILE_W;
    int tile_origin_y = blockIdx.y * TILE_H;

    // 1. Cooperative loading of tile + halo into shared memory.
    // Every thread in the block participates regardless of whether its
    // corresponding output coordinate is inside or outside the image bounds.
    for (int i = tid; i < SMEM_SIZE; i += NUM_THREADS) {
        int smem_y = i / SMEM_W;
        int smem_x = i % SMEM_W;

        // Global coordinate for this sample (offset by -1 for halo)
        int gx = tile_origin_x + smem_x - 1;
        int gy = tile_origin_y + smem_y - 1;

        // Clamp-to-edge border policy
        if (gx < 0) gx = 0;
        else if (gx >= width) gx = width - 1;

        if (gy < 0) gy = 0;
        else if (gy >= height) gy = height - 1;

        const uint8_t* in_row = d_src + static_cast<size_t>(gy) * src_pitch;
        s_tile[smem_y][smem_x] = in_row[gx];
    }

    // 2. Barrier synchronization: ensure all shared-memory loading has completed.
    // Every thread in the block reaches this barrier to prevent deadlocks.
    __syncthreads();

    // 3. Compute output pixel from shared memory
    int out_x = tile_origin_x + threadIdx.x;
    int out_y = tile_origin_y + threadIdx.y;

    // Only compute and write output for threads within valid image boundaries
    if (out_x < width && out_y < height) {
        // In s_tile, center pixel is at (threadIdx.x + 1, threadIdx.y + 1)
        int cx = threadIdx.x + 1;
        int cy = threadIdx.y + 1;

        uint32_t sum = 0;
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                sum += s_tile[cy + dy][cx + dx];
            }
        }

        float avg = static_cast<float>(sum) / 9.0f;
        float rounded = roundf(avg);
        if (rounded < 0.0f) rounded = 0.0f;
        else if (rounded > 255.0f) rounded = 255.0f;

        uint8_t* out_row = d_dst + static_cast<size_t>(out_y) * dst_pitch;
        out_row[out_x] = static_cast<uint8_t>(rounded);
    }
}

TiledKernelStats get_tiled_3x3_kernel_stats(TileConfig config) {
    TiledKernelStats stats;
    cudaFuncAttributes attr;
    if (config == TileConfig::TILE_8x8) {
        stats.block_dim_x = 8;
        stats.block_dim_y = 8;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, meanBlurTiled3x3Kernel<8, 8>));
        stats.shared_mem_bytes = attr.sharedSizeBytes;
    } else if (config == TileConfig::TILE_16x16) {
        stats.block_dim_x = 16;
        stats.block_dim_y = 16;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, meanBlurTiled3x3Kernel<16, 16>));
        stats.shared_mem_bytes = attr.sharedSizeBytes;
    } else if (config == TileConfig::TILE_32x32) {
        stats.block_dim_x = 32;
        stats.block_dim_y = 32;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, meanBlurTiled3x3Kernel<32, 32>));
        stats.shared_mem_bytes = attr.sharedSizeBytes;
    }
    return stats;
}

DeviceImage mean_blur_tiled_3x3(const DeviceImage& d_src, TileConfig config, cudaStream_t stream) {
    if (d_src.is_empty()) {
        throw std::invalid_argument("Cannot apply mean_blur_tiled_3x3 to an empty DeviceImage");
    }
    if (d_src.channels() != 1) {
        throw std::invalid_argument("mean_blur_tiled_3x3 currently expects 1-channel (grayscale) image, got " +
                                    std::to_string(d_src.channels()));
    }

    int width = d_src.width();
    int height = d_src.height();
    DeviceImage d_dst(width, height, 1);

    if (config == TileConfig::TILE_8x8) {
        dim3 block(8, 8);
        dim3 grid((width + 8 - 1) / 8, (height + 8 - 1) / 8);
        if (stream == 0) {
            meanBlurTiled3x3Kernel<8, 8><<<grid, block>>>(
                d_src.data(), d_src.stride_bytes(),
                d_dst.data(), d_dst.stride_bytes(),
                width, height
            );
        } else {
            meanBlurTiled3x3Kernel<8, 8><<<grid, block, 0, stream>>>(
                d_src.data(), d_src.stride_bytes(),
                d_dst.data(), d_dst.stride_bytes(),
                width, height
            );
        }
    } else if (config == TileConfig::TILE_16x16) {
        dim3 block(16, 16);
        dim3 grid((width + 16 - 1) / 16, (height + 16 - 1) / 16);
        if (stream == 0) {
            meanBlurTiled3x3Kernel<16, 16><<<grid, block>>>(
                d_src.data(), d_src.stride_bytes(),
                d_dst.data(), d_dst.stride_bytes(),
                width, height
            );
        } else {
            meanBlurTiled3x3Kernel<16, 16><<<grid, block, 0, stream>>>(
                d_src.data(), d_src.stride_bytes(),
                d_dst.data(), d_dst.stride_bytes(),
                width, height
            );
        }
    } else if (config == TileConfig::TILE_32x32) {
        dim3 block(32, 32);
        dim3 grid((width + 32 - 1) / 32, (height + 32 - 1) / 32);
        if (stream == 0) {
            meanBlurTiled3x3Kernel<32, 32><<<grid, block>>>(
                d_src.data(), d_src.stride_bytes(),
                d_dst.data(), d_dst.stride_bytes(),
                width, height
            );
        } else {
            meanBlurTiled3x3Kernel<32, 32><<<grid, block, 0, stream>>>(
                d_src.data(), d_src.stride_bytes(),
                d_dst.data(), d_dst.stride_bytes(),
                width, height
            );
        }
    }

    CUDA_CHECK_LAST();

    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return d_dst;
}

HostImage mean_blur_tiled_3x3(const HostImage& h_src, TileConfig config, cudaStream_t stream) {
    DeviceImage d_src = to_device(h_src, stream);
    DeviceImage d_dst = mean_blur_tiled_3x3(d_src, config, stream);
    return to_host(d_dst, stream);
}

} // namespace cuda
} // namespace cuda_image_proc



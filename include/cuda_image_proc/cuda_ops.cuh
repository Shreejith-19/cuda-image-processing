#pragma once

#include <cuda_runtime.h>
#include <string>
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
 * @brief Retrieves information about a specified CUDA device.
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
 * @brief Future CUDA image processing kernels interface.
 * (Grayscale, Blur, Sobel kernels to be implemented in subsequent phases)
 */

} // namespace cuda
} // namespace cuda_image_proc

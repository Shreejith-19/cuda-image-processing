#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include "cuda_image_proc/image.hpp"
#include "cuda_image_proc/device_image.cuh"
#include "cuda_image_proc/cuda_utils.hpp"

namespace cuda_image_proc {

/**
 * @brief Copies an Image from host memory to a new DeviceImage on GPU.
 * 
 * Handles row stride differences between host and pitched device buffers
 * using 2D memory copy operations.
 */
inline DeviceImage to_device(const HostImage& host_image, cudaStream_t stream = 0) {
    if (host_image.is_empty()) {
        throw std::invalid_argument("Cannot copy empty HostImage to device");
    }

    DeviceImage device_img(host_image.width, host_image.height, host_image.channels);
    size_t width_bytes = host_image.row_bytes();

    if (stream == 0) {
        CUDA_CHECK(cudaMemcpy2D(
            device_img.data(),
            device_img.stride_bytes(),
            host_image.raw_data(),
            host_image.stride_bytes,
            width_bytes,
            static_cast<size_t>(host_image.height),
            cudaMemcpyHostToDevice
        ));
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(
            device_img.data(),
            device_img.stride_bytes(),
            host_image.raw_data(),
            host_image.stride_bytes,
            width_bytes,
            static_cast<size_t>(host_image.height),
            cudaMemcpyHostToDevice,
            stream
        ));
    }

    return device_img;
}

/**
 * @brief Copies an Image from GPU DeviceImage back to a HostImage.
 * 
 * Accurately extracts pixel rows respecting device stride into a tightly packed
 * or custom-strided HostImage buffer.
 */
inline HostImage to_host(const DeviceImage& device_image, cudaStream_t stream = 0) {
    if (device_image.is_empty()) {
        throw std::invalid_argument("Cannot copy empty DeviceImage to host");
    }

    HostImage host_img(device_image.width(), device_image.height(), device_image.channels());
    size_t width_bytes = device_image.row_bytes();

    if (stream == 0) {
        CUDA_CHECK(cudaMemcpy2D(
            host_img.raw_data(),
            host_img.stride_bytes,
            device_image.data(),
            device_image.stride_bytes(),
            width_bytes,
            static_cast<size_t>(device_image.height()),
            cudaMemcpyDeviceToHost
        ));
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(
            host_img.raw_data(),
            host_img.stride_bytes,
            device_image.data(),
            device_image.stride_bytes(),
            width_bytes,
            static_cast<size_t>(device_image.height()),
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return host_img;
}

} // namespace cuda_image_proc

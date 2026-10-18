#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <stdexcept>
#include "cuda_image_proc/cuda_utils.hpp"

namespace cuda_image_proc {

/**
 * @brief RAII-managed 2D device image allocation.
 *
 * Prevents accidental copying of device pointers by deleting copy constructor
 * and copy assignment. Supports move semantics and explicit deep cloning.
 * Allocates pitched memory via cudaMallocPitch for optimal GPU coalescing.
 */
class DeviceImage {
public:
    DeviceImage() = default;

    DeviceImage(int width, int height, int channels)
        : width_(width), height_(height), channels_(channels) {
        if (width <= 0 || height <= 0 || channels <= 0) {
            throw std::invalid_argument("DeviceImage dimensions and channels must be positive");
        }
        size_t row_width_bytes = static_cast<size_t>(width_) * static_cast<size_t>(channels_);
        CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_data_), &stride_bytes_, row_width_bytes, height_));
    }

    ~DeviceImage() noexcept {
        release();
    }

    // Delete copy operations to prevent accidental shallow copies of device pointers
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;

    // Move constructor
    DeviceImage(DeviceImage&& other) noexcept
        : width_(other.width_),
          height_(other.height_),
          channels_(other.channels_),
          stride_bytes_(other.stride_bytes_),
          d_data_(other.d_data_) {
        other.width_ = 0;
        other.height_ = 0;
        other.channels_ = 0;
        other.stride_bytes_ = 0;
        other.d_data_ = nullptr;
    }

    // Move assignment
    DeviceImage& operator=(DeviceImage&& other) noexcept {
        if (this != &other) {
            release();
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            stride_bytes_ = other.stride_bytes_;
            d_data_ = other.d_data_;

            other.width_ = 0;
            other.height_ = 0;
            other.channels_ = 0;
            other.stride_bytes_ = 0;
            other.d_data_ = nullptr;
        }
        return *this;
    }

    // Explicit deep clone on device
    DeviceImage clone(cudaStream_t stream = 0) const {
        if (is_empty()) {
            return DeviceImage();
        }
        DeviceImage copy(width_, height_, channels_);
        size_t width_bytes = static_cast<size_t>(width_) * static_cast<size_t>(channels_);
        if (stream == 0) {
            CUDA_CHECK(cudaMemcpy2D(copy.d_data_, copy.stride_bytes_,
                                    d_data_, stride_bytes_,
                                    width_bytes, height_,
                                    cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(copy.d_data_, copy.stride_bytes_,
                                         d_data_, stride_bytes_,
                                         width_bytes, height_,
                                         cudaMemcpyDeviceToDevice, stream));
        }
        return copy;
    }

    // Device and Host row pointer computation
    __host__ __device__ inline uint8_t* row_ptr(int y) noexcept {
        return d_data_ + static_cast<size_t>(y) * stride_bytes_;
    }

    __host__ __device__ inline const uint8_t* row_ptr(int y) const noexcept {
        return d_data_ + static_cast<size_t>(y) * stride_bytes_;
    }

    // Accessors
    __host__ __device__ inline int width() const noexcept { return width_; }
    __host__ __device__ inline int height() const noexcept { return height_; }
    __host__ __device__ inline int channels() const noexcept { return channels_; }
    __host__ __device__ inline size_t stride_bytes() const noexcept { return stride_bytes_; }
    __host__ __device__ inline size_t row_bytes() const noexcept {
        return static_cast<size_t>(width_) * static_cast<size_t>(channels_);
    }
    __host__ __device__ inline uint8_t* data() noexcept { return d_data_; }
    __host__ __device__ inline const uint8_t* data() const noexcept { return d_data_; }

    inline bool is_empty() const noexcept {
        return d_data_ == nullptr || width_ <= 0 || height_ <= 0 || channels_ <= 0;
    }

private:
    void release() noexcept {
        if (d_data_ != nullptr) {
            cudaFree(d_data_);
            d_data_ = nullptr;
        }
        width_ = 0;
        height_ = 0;
        channels_ = 0;
        stride_bytes_ = 0;
    }

    int width_{0};
    int height_{0};
    int channels_{0};
    size_t stride_bytes_{0};
    uint8_t* d_data_{nullptr};
};

} // namespace cuda_image_proc

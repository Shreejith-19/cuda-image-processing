#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            std::string errMsg = std::string("CUDA Error: ") + cudaGetErrorString(err__) +  \
                                 " at " + __FILE__ + ":" + std::to_string(__LINE__) +        \
                                 " (" #call ")";                                             \
            std::cerr << errMsg << std::endl;                                                \
            throw std::runtime_error(errMsg);                                                \
        }                                                                                    \
    } while (0)

#define CUDA_CHECK_LAST()                                                                    \
    do {                                                                                     \
        cudaError_t err__ = cudaGetLastError();                                              \
        if (err__ != cudaSuccess) {                                                          \
            std::string errMsg = std::string("CUDA Kernel Error: ") +                        \
                                 cudaGetErrorString(err__) + " at " + __FILE__ + ":" +       \
                                 std::to_string(__LINE__);                                   \
            std::cerr << errMsg << std::endl;                                                \
            throw std::runtime_error(errMsg);                                                \
        }                                                                                    \
    } while (0)

namespace cuda_image_proc {

/**
 * @brief RAII wrapper for a CUDA stream.
 */
class CudaStream {
public:
    CudaStream(unsigned int flags = cudaStreamDefault) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~CudaStream() noexcept {
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamSynchronize(stream_);
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }

    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    cudaStream_t stream_{nullptr};
};

} // namespace cuda_image_proc

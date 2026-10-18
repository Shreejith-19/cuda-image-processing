#include "cuda_image_proc/cuda_ops.cuh"
#include "cuda_image_proc/cuda_utils.hpp"

namespace cuda_image_proc {
namespace cuda {

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

} // namespace cuda
} // namespace cuda_image_proc

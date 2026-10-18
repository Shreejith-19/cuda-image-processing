#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            return 1;                                                         \
        }                                                                     \
    } while (0)

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " CUDA Smoke Test - Environment Check    " << std::endl;
    std::cout << "========================================" << std::endl;

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable devices found." << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA-capable device(s):" << std::endl;
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::cout << "  [" << dev << "] " << prop.name << std::endl;
        std::cout << "      Compute Capability : " << prop.major << "." << prop.minor << std::endl;
        std::cout << "      Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "      Multiprocessors (SMs): " << prop.multiProcessorCount << std::endl;
        std::cout << "      Warp Size          : " << prop.warpSize << std::endl;
        std::cout << "      Max Threads / Block: " << prop.maxThreadsPerBlock << std::endl;
    }

    // Set active device
    CUDA_CHECK(cudaSetDevice(0));

    // Vector Addition Test
    const int N = 1024 * 64; // 65,536 elements
    const size_t bytes = N * sizeof(float);

    std::cout << "\nRunning sample kernel (Vector Addition of " << N << " elements)..." << std::endl;

    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    std::cout << "Launching kernel with grid size " << gridSize 
              << ", block size " << blockSize << "..." << std::endl;

    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify result
    bool verified = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            std::cerr << "Verification mismatch at index " << i << ": got " 
                      << h_c[i] << ", expected " << expected << std::endl;
            verified = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    if (verified) {
        std::cout << "\n>>> SMOKE TEST PASSED: GPU execution and memory transfers verified successfully! <<<" << std::endl;
        return 0;
    } else {
        std::cerr << "\n>>> SMOKE TEST FAILED: Output verification failed. <<<" << std::endl;
        return 1;
    }
}

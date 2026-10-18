#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <random>
#include <cuda_runtime.h>
#include "cuda_image_proc/image.hpp"
#include "cuda_image_proc/image_io.hpp"
#include "cuda_image_proc/cpu_ops.hpp"
#include "cuda_image_proc/cuda_utils.hpp"
#include "cuda_image_proc/device_image.cuh"
#include "cuda_image_proc/memory_transfer.cuh"
#include "cuda_image_proc/cuda_ops.cuh"

namespace fs = std::filesystem;
using namespace cuda_image_proc;

std::string resolve_image_path(const std::string& rel_path) {
    if (fs::exists(rel_path)) return rel_path;
    if (fs::exists("../" + rel_path)) return "../" + rel_path;
    return rel_path;
}

Image create_synthetic_grayscale(int width, int height, unsigned int seed = 42) {
    Image img(width, height, 1);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = img.row_ptr(y);
        for (int x = 0; x < width; ++x) {
            row[x] = static_cast<uint8_t>(dist(rng));
        }
    }
    return img;
}

struct BenchResult {
    std::string name;
    size_t shared_mem_bytes;
    int block_x;
    int block_y;
    float mean_ms;
    float min_ms;
    float speedup;
    bool bit_exact;
};

template <typename Func>
BenchResult benchmark_gaussian_kernel(
    const std::string& name,
    const DeviceImage& d_src,
    const Image& cpu_ref,
    size_t smem_bytes,
    int block_x,
    int block_y,
    float baseline_mean_ms,
    Func kernel_invoker,
    int warmup_runs = 10,
    int timed_runs = 100) {

    CudaStream stream;

    // Warm-up runs
    for (int w = 0; w < warmup_runs; ++w) {
        DeviceImage d_out = kernel_invoker(d_src, stream.get());
    }
    stream.synchronize();

    // Timed runs using CUDA events
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    std::vector<float> run_times(timed_runs);
    DeviceImage last_out;

    for (int i = 0; i < timed_runs; ++i) {
        CUDA_CHECK(cudaEventRecord(start_event, stream.get()));
        last_out = kernel_invoker(d_src, stream.get());
        CUDA_CHECK(cudaEventRecord(stop_event, stream.get()));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        run_times[i] = ms;
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    // Correctness check against CPU reference
    Image h_out = to_host(last_out, stream.get());
    cpu::ImageDiffStats stats = cpu::compute_diff_stats(cpu_ref, h_out);
    bool exact = (stats.mismatched_pixels == 0) && (stats.max_diff == 0.0);

    float sum_time = std::accumulate(run_times.begin(), run_times.end(), 0.0f);
    float mean_time = sum_time / static_cast<float>(timed_runs);
    float min_time = *std::min_element(run_times.begin(), run_times.end());
    float speedup = (baseline_mean_ms > 0.0f) ? (baseline_mean_ms / mean_time) : 1.0f;

    return BenchResult{name, smem_bytes, block_x, block_y, mean_time, min_time, speedup, exact};
}

void run_gaussian_benchmark_for_image(const std::string& title, const Image& img, int warmup_runs = 10, int timed_runs = 100) {
    std::cout << "\n========================================================================================================\n";
    std::cout << " Benchmark: " << title << " (" << img.width << " x " << img.height 
              << ", " << (img.width * img.height) << " pixels)\n";
    std::cout << " Iterations: " << timed_runs << " timed runs (" << warmup_runs << " warmup runs)\n";
    std::cout << "========================================================================================================\n";

    // Compute CPU Reference
    Image cpu_ref = cpu::gaussian_blur_3x3(img);

    // Transfer input to GPU once
    DeviceImage d_src = to_device(img);

    // 1. Naive Baseline
    BenchResult naive_res = benchmark_gaussian_kernel(
        "Naive CUDA (Global Mem)",
        d_src,
        cpu_ref,
        0,
        16, 16,
        0.0f,
        [](const DeviceImage& in, cudaStream_t s) {
            return cuda::gaussian_blur_naive_3x3(in, s);
        },
        warmup_runs,
        timed_runs
    );
    naive_res.speedup = 1.0f;

    // 2. Tiled 8x8
    cuda::TiledKernelStats stats8 = cuda::get_tiled_gaussian_3x3_kernel_stats(cuda::TileConfig::TILE_8x8);
    BenchResult tiled8_res = benchmark_gaussian_kernel(
        "Tiled Shared Mem 8x8",
        d_src,
        cpu_ref,
        stats8.shared_mem_bytes,
        8, 8,
        naive_res.mean_ms,
        [](const DeviceImage& in, cudaStream_t s) {
            return cuda::gaussian_blur_tiled_3x3(in, cuda::TileConfig::TILE_8x8, s);
        },
        warmup_runs,
        timed_runs
    );

    // 3. Tiled 16x16
    cuda::TiledKernelStats stats16 = cuda::get_tiled_gaussian_3x3_kernel_stats(cuda::TileConfig::TILE_16x16);
    BenchResult tiled16_res = benchmark_gaussian_kernel(
        "Tiled Shared Mem 16x16",
        d_src,
        cpu_ref,
        stats16.shared_mem_bytes,
        16, 16,
        naive_res.mean_ms,
        [](const DeviceImage& in, cudaStream_t s) {
            return cuda::gaussian_blur_tiled_3x3(in, cuda::TileConfig::TILE_16x16, s);
        },
        warmup_runs,
        timed_runs
    );

    // 4. Tiled 32x32
    cuda::TiledKernelStats stats32 = cuda::get_tiled_gaussian_3x3_kernel_stats(cuda::TileConfig::TILE_32x32);
    BenchResult tiled32_res = benchmark_gaussian_kernel(
        "Tiled Shared Mem 32x32",
        d_src,
        cpu_ref,
        stats32.shared_mem_bytes,
        32, 32,
        naive_res.mean_ms,
        [](const DeviceImage& in, cudaStream_t s) {
            return cuda::gaussian_blur_tiled_3x3(in, cuda::TileConfig::TILE_32x32, s);
        },
        warmup_runs,
        timed_runs
    );

    std::vector<BenchResult> results = {naive_res, tiled8_res, tiled16_res, tiled32_res};

    std::cout << std::left << std::setw(28) << "Kernel Implementation"
              << std::setw(14) << "Block Size"
              << std::setw(16) << "Shared Mem/Blk"
              << std::setw(16) << "Mean Time (ms)"
              << std::setw(16) << "Min Time (ms)"
              << std::setw(14) << "Speedup"
              << "Bit-Exact?\n";
    std::cout << std::string(112, '-') << "\n";

    for (const auto& r : results) {
        std::string blk_str = std::to_string(r.block_x) + "x" + std::to_string(r.block_y);
        std::string smem_str = std::to_string(r.shared_mem_bytes) + " B";
        std::string speedup_str = (r.speedup == 1.0f) ? "1.00x (base)" : (std::to_string(r.speedup).substr(0, 4) + "x");

        std::cout << std::left << std::setw(28) << r.name
                  << std::setw(14) << blk_str
                  << std::setw(16) << smem_str
                  << std::fixed << std::setprecision(5) << std::setw(16) << r.mean_ms
                  << std::fixed << std::setprecision(5) << std::setw(16) << r.min_ms
                  << std::setw(14) << speedup_str
                  << (r.bit_exact ? "Yes (100%)" : "NO (DIFF)") << "\n";
    }
}

int main() {
    std::cout << "========================================================================================================\n";
    std::cout << " CUDA 3x3 Gaussian Blur: Naive vs. Shared-Memory Tiling Benchmark Suite                                 \n";
    std::cout << "========================================================================================================\n";

    cuda::DeviceInfo dev = cuda::get_device_info(0);
    std::cout << "GPU Target: " << dev.name << " (Compute " << dev.major << "." << dev.minor << ")\n";
    std::cout << "VRAM: " << (dev.total_memory_bytes / (1024 * 1024)) << " MB, SMs: " << dev.sm_count 
              << ", Max Threads/Block: " << dev.max_threads_per_block << "\n";

    // 1. Real Sample Image (256x256)
    std::string sample_path = resolve_image_path("images/input/sample.png");
    try {
        Image sample = load_image(sample_path, 1);
        run_gaussian_benchmark_for_image("Real Sample Image (sample.png)", sample, 15, 100);
    } catch (const std::exception& e) {
        std::cerr << "Could not load sample.png: " << e.what() << "\n";
    }

    // 2. High Resolution Image (1024x1024)
    Image img_1k = create_synthetic_grayscale(1024, 1024, 101);
    run_gaussian_benchmark_for_image("1024x1024 High-Resolution Synthetic", img_1k, 10, 50);

    // 3. Very Large Image (2048x2048, ~4.2M pixels)
    Image img_2k = create_synthetic_grayscale(2048, 2048, 202);
    run_gaussian_benchmark_for_image("2048x2048 Ultra-Resolution Synthetic (4.19M Pixels)", img_2k, 10, 30);

    // 4. Irregular Dimensions Non-Block Multiple (831x517)
    Image img_irregular = create_synthetic_grayscale(831, 517, 303);
    run_gaussian_benchmark_for_image("831x517 Irregular Non-Multiple Dimensions", img_irregular, 10, 50);

    return 0;
}

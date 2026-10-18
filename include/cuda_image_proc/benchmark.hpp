#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <random>
#include <cuda_runtime.h>

#include "cuda_image_proc/image.hpp"
#include "cuda_image_proc/cpu_ops.hpp"
#include "cuda_image_proc/cuda_utils.hpp"
#include "cuda_image_proc/device_image.cuh"
#include "cuda_image_proc/memory_transfer.cuh"
#include "cuda_image_proc/cuda_ops.cuh"

namespace cuda_image_proc {
namespace benchmark {

namespace fs = std::filesystem;
using cuda::TileConfig;

/**
 * @brief Benchmark timing measurement policy documentation:
 *
 * 1. Kernel Time (kernel_time_ms):
 *    - Measured strictly on the GPU using CUDA events (cudaEvent_t).
 *    - Excludes: Host memory allocation, GPU memory allocation, Host-to-Device transfer,
 *      Device-to-Host transfer, disk file loading, disk output saving.
 *
 * 2. End-to-End Latency (end_to_end_time_ms):
 *    - Measured using high-resolution host clock (std::chrono::high_resolution_clock).
 *    - Includes: Host-to-Device (H2D) memory transfer, GPU kernel execution, Device-to-Host (D2H) memory transfer.
 *    - Excludes: Disk file loading, disk output saving, initial memory allocation (pre-allocated).
 *
 * 3. CPU Reference Time (mean_time_ms):
 *    - Measured using high-resolution host clock (std::chrono::high_resolution_clock).
 *    - Includes: Pure CPU compute algorithm execution on in-memory buffers.
 *    - Excludes: Disk file loading and disk output saving.
 */
struct BenchmarkRecord {
    std::string operation;
    int image_width{0};
    int image_height{0};
    int filter_size{0};
    std::string implementation;
    int tile_width{0};
    int tile_height{0};
    double kernel_time_ms{0.0};
    double end_to_end_time_ms{0.0};
    double median_time_ms{0.0};
    double mean_time_ms{0.0};
    double min_time_ms{0.0};
    double max_time_ms{0.0};
    double std_dev_ms{0.0};
    double speedup_vs_cpu{1.0};
    double speedup_vs_naive{1.0};
};

struct Statistics {
    double mean{0.0};
    double median{0.0};
    double min{0.0};
    double max{0.0};
    double std_dev{0.0};
};

inline Statistics calculate_statistics(std::vector<double> samples) {
    Statistics stats;
    if (samples.empty()) return stats;

    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();

    stats.min = samples.front();
    stats.max = samples.back();

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean = sum / static_cast<double>(n);

    if (n % 2 == 1) {
        stats.median = samples[n / 2];
    } else {
        stats.median = 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    }

    double sq_sum = 0.0;
    for (double v : samples) {
        sq_sum += (v - stats.mean) * (v - stats.mean);
    }
    stats.std_dev = (n > 1) ? std::sqrt(sq_sum / static_cast<double>(n - 1)) : 0.0;

    return stats;
}

inline Image generate_synthetic_image(int width, int height, int channels, unsigned int seed = 42) {
    Image img(width, height, channels);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);

    for (int y = 0; y < height; ++y) {
        uint8_t* row = img.row_ptr(y);
        for (int x = 0; x < width * channels; ++x) {
            row[x] = static_cast<uint8_t>(dist(rng));
        }
    }
    return img;
}

inline void write_csv_header(std::ostream& os) {
    os << "operation,image_width,image_height,filter_size,implementation,tile_width,tile_height,"
       << "kernel_time_ms,end_to_end_time_ms,median_time_ms,mean_time_ms,min_time_ms,max_time_ms,"
       << "std_dev_ms,speedup_vs_cpu,speedup_vs_naive\n";
}

inline void write_csv_record(std::ostream& os, const BenchmarkRecord& r) {
    os << r.operation << ","
       << r.image_width << ","
       << r.image_height << ","
       << r.filter_size << ","
       << r.implementation << ","
       << r.tile_width << ","
       << r.tile_height << ","
       << std::fixed << std::setprecision(5)
       << r.kernel_time_ms << ","
       << r.end_to_end_time_ms << ","
       << r.median_time_ms << ","
       << r.mean_time_ms << ","
       << r.min_time_ms << ","
       << r.max_time_ms << ","
       << r.std_dev_ms << ","
       << std::setprecision(2)
       << r.speedup_vs_cpu << ","
       << r.speedup_vs_naive << "\n";
}

inline void export_results_to_csv(const std::string& filepath, const std::vector<BenchmarkRecord>& records) {
    fs::path p(filepath);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output CSV file for writing: " + filepath);
    }

    write_csv_header(file);
    for (const auto& r : records) {
        write_csv_record(file, r);
    }
}

/**
 * @brief Benchmark harness for a specific operation and image size.
 */
class OperationBenchmark {
public:
    OperationBenchmark(int width, int height, int iterations = 30, int warmup = 5)
        : width_(width), height_(height), iterations_(iterations), warmup_(warmup) {}

    // --- 1. RGB TO GRAYSCALE ---
    std::vector<BenchmarkRecord> run_grayscale() {
        std::vector<BenchmarkRecord> results;
        Image h_src = generate_synthetic_image(width_, height_, 3, 101);
        Image h_dst(width_, height_, 1);

        // A. CPU Reference
        std::vector<double> cpu_times;
        cpu_times.reserve(iterations_);

        // Warmup
        for (int i = 0; i < warmup_; ++i) {
            Image tmp = cpu::rgb_to_grayscale(h_src);
            (void)tmp;
        }

        for (int i = 0; i < iterations_; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Image tmp = cpu::rgb_to_grayscale(h_src);
            auto t1 = std::chrono::high_resolution_clock::now();
            (void)tmp;
            cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        Statistics cpu_stats = calculate_statistics(cpu_times);

        BenchmarkRecord cpu_rec;
        cpu_rec.operation = "rgb_to_grayscale";
        cpu_rec.image_width = width_;
        cpu_rec.image_height = height_;
        cpu_rec.filter_size = 1;
        cpu_rec.implementation = "CPU Reference";
        cpu_rec.tile_width = 0;
        cpu_rec.tile_height = 0;
        cpu_rec.kernel_time_ms = cpu_stats.mean;
        cpu_rec.end_to_end_time_ms = cpu_stats.mean;
        cpu_rec.median_time_ms = cpu_stats.median;
        cpu_rec.mean_time_ms = cpu_stats.mean;
        cpu_rec.min_time_ms = cpu_stats.min;
        cpu_rec.max_time_ms = cpu_stats.max;
        cpu_rec.std_dev_ms = cpu_stats.std_dev;
        cpu_rec.speedup_vs_cpu = 1.0;
        cpu_rec.speedup_vs_naive = 1.0;
        results.push_back(cpu_rec);

        // B. Naive CUDA
        CudaStream stream;
        DeviceImage d_src(width_, height_, 3);
        DeviceImage d_dst(width_, height_, 1);
        size_t src_bytes = static_cast<size_t>(width_) * 3;
        size_t dst_bytes = static_cast<size_t>(width_) * 1;

        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        // Initial copy for kernel-only timing
        CUDA_CHECK(cudaMemcpy2D(d_src.data(), d_src.stride_bytes(),
                                h_src.raw_data(), h_src.stride_bytes,
                                src_bytes, height_, cudaMemcpyHostToDevice));

        // Warmup
        for (int i = 0; i < warmup_; ++i) {
            DeviceImage tmp = cuda::rgb_to_grayscale(d_src, stream);
            stream.synchronize();
            (void)tmp;
        }

        std::vector<double> gpu_kernel_times;
        std::vector<double> gpu_e2e_times;
        gpu_kernel_times.reserve(iterations_);
        gpu_e2e_times.reserve(iterations_);

        for (int i = 0; i < iterations_; ++i) {
            // E2E Timing: H2D -> Kernel -> D2H
            auto t_e2e_0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemcpy2DAsync(d_src.data(), d_src.stride_bytes(),
                                         h_src.raw_data(), h_src.stride_bytes,
                                         src_bytes, height_, cudaMemcpyHostToDevice, stream));

            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            DeviceImage tmp = cuda::rgb_to_grayscale(d_src, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));

            CUDA_CHECK(cudaMemcpy2DAsync(h_dst.raw_data(), h_dst.stride_bytes,
                                         tmp.data(), tmp.stride_bytes(),
                                         dst_bytes, height_, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t_e2e_1 = std::chrono::high_resolution_clock::now();

            float k_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_stop));
            gpu_kernel_times.push_back(static_cast<double>(k_ms));
            gpu_e2e_times.push_back(std::chrono::duration<double, std::milli>(t_e2e_1 - t_e2e_0).count());
        }

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        Statistics k_stats = calculate_statistics(gpu_kernel_times);
        Statistics e2e_stats = calculate_statistics(gpu_e2e_times);

        BenchmarkRecord gpu_rec;
        gpu_rec.operation = "rgb_to_grayscale";
        gpu_rec.image_width = width_;
        gpu_rec.image_height = height_;
        gpu_rec.filter_size = 1;
        gpu_rec.implementation = "Naive CUDA";
        gpu_rec.tile_width = 16;
        gpu_rec.tile_height = 16;
        gpu_rec.kernel_time_ms = k_stats.mean;
        gpu_rec.end_to_end_time_ms = e2e_stats.mean;
        gpu_rec.median_time_ms = k_stats.median;
        gpu_rec.mean_time_ms = k_stats.mean;
        gpu_rec.min_time_ms = k_stats.min;
        gpu_rec.max_time_ms = k_stats.max;
        gpu_rec.std_dev_ms = k_stats.std_dev;
        gpu_rec.speedup_vs_cpu = (k_stats.mean > 0.0) ? (cpu_stats.mean / k_stats.mean) : 1.0;
        gpu_rec.speedup_vs_naive = 1.0;
        results.push_back(gpu_rec);

        return results;
    }

    // --- 2. MEAN BLUR ---
    std::vector<BenchmarkRecord> run_mean_blur(int filter_size = 3) {
        std::vector<BenchmarkRecord> results;
        Image h_src = generate_synthetic_image(width_, height_, 1, 202);
        Image h_dst(width_, height_, 1);
        size_t width_bytes = static_cast<size_t>(width_);

        // A. CPU Reference
        std::vector<double> cpu_times;
        cpu_times.reserve(iterations_);
        for (int i = 0; i < warmup_; ++i) {
            Image tmp = cpu::mean_blur(h_src, filter_size);
            (void)tmp;
        }
        for (int i = 0; i < iterations_; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Image tmp = cpu::mean_blur(h_src, filter_size);
            auto t1 = std::chrono::high_resolution_clock::now();
            (void)tmp;
            cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        Statistics cpu_stats = calculate_statistics(cpu_times);

        BenchmarkRecord cpu_rec;
        cpu_rec.operation = "mean_blur";
        cpu_rec.image_width = width_;
        cpu_rec.image_height = height_;
        cpu_rec.filter_size = filter_size;
        cpu_rec.implementation = "CPU Reference";
        cpu_rec.tile_width = 0;
        cpu_rec.tile_height = 0;
        cpu_rec.kernel_time_ms = cpu_stats.mean;
        cpu_rec.end_to_end_time_ms = cpu_stats.mean;
        cpu_rec.median_time_ms = cpu_stats.median;
        cpu_rec.mean_time_ms = cpu_stats.mean;
        cpu_rec.min_time_ms = cpu_stats.min;
        cpu_rec.max_time_ms = cpu_stats.max;
        cpu_rec.std_dev_ms = cpu_stats.std_dev;
        cpu_rec.speedup_vs_cpu = 1.0;
        cpu_rec.speedup_vs_naive = 1.0;
        results.push_back(cpu_rec);

        CudaStream stream;
        DeviceImage d_src(width_, height_, 1);
        DeviceImage d_dst(width_, height_, 1);
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        // Initial copy
        CUDA_CHECK(cudaMemcpy2D(d_src.data(), d_src.stride_bytes(),
                                h_src.raw_data(), h_src.stride_bytes,
                                width_bytes, height_, cudaMemcpyHostToDevice));

        // B. Naive CUDA
        for (int i = 0; i < warmup_; ++i) {
            DeviceImage tmp = cuda::mean_blur_naive(d_src, filter_size, stream);
            stream.synchronize();
            (void)tmp;
        }

        std::vector<double> naive_kernel_times;
        std::vector<double> naive_e2e_times;
        naive_kernel_times.reserve(iterations_);
        naive_e2e_times.reserve(iterations_);

        for (int i = 0; i < iterations_; ++i) {
            auto t_e2e_0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemcpy2DAsync(d_src.data(), d_src.stride_bytes(),
                                         h_src.raw_data(), h_src.stride_bytes,
                                         width_bytes, height_, cudaMemcpyHostToDevice, stream));

            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            DeviceImage tmp = cuda::mean_blur_naive(d_src, filter_size, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));

            CUDA_CHECK(cudaMemcpy2DAsync(h_dst.raw_data(), h_dst.stride_bytes,
                                         tmp.data(), tmp.stride_bytes(),
                                         width_bytes, height_, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t_e2e_1 = std::chrono::high_resolution_clock::now();

            float k_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_stop));
            naive_kernel_times.push_back(static_cast<double>(k_ms));
            naive_e2e_times.push_back(std::chrono::duration<double, std::milli>(t_e2e_1 - t_e2e_0).count());
        }

        Statistics naive_k_stats = calculate_statistics(naive_kernel_times);
        Statistics naive_e2e_stats = calculate_statistics(naive_e2e_times);

        BenchmarkRecord naive_rec;
        naive_rec.operation = "mean_blur";
        naive_rec.image_width = width_;
        naive_rec.image_height = height_;
        naive_rec.filter_size = filter_size;
        naive_rec.implementation = "Naive CUDA";
        naive_rec.tile_width = 16;
        naive_rec.tile_height = 16;
        naive_rec.kernel_time_ms = naive_k_stats.mean;
        naive_rec.end_to_end_time_ms = naive_e2e_stats.mean;
        naive_rec.median_time_ms = naive_k_stats.median;
        naive_rec.mean_time_ms = naive_k_stats.mean;
        naive_rec.min_time_ms = naive_k_stats.min;
        naive_rec.max_time_ms = naive_k_stats.max;
        naive_rec.std_dev_ms = naive_k_stats.std_dev;
        naive_rec.speedup_vs_cpu = (naive_k_stats.mean > 0.0) ? (cpu_stats.mean / naive_k_stats.mean) : 1.0;
        naive_rec.speedup_vs_naive = 1.0;
        results.push_back(naive_rec);

        // C. Tiled CUDA (3x3 only)
        if (filter_size == 3) {
            std::vector<std::pair<TileConfig, int>> configs = {
                {TileConfig::TILE_8x8, 8},
                {TileConfig::TILE_16x16, 16},
                {TileConfig::TILE_32x32, 32}
            };

            for (const auto& [cfg, tile_dim] : configs) {
                // Warmup
                for (int i = 0; i < warmup_; ++i) {
                    DeviceImage tmp = cuda::mean_blur_tiled_3x3(d_src, cfg, stream);
                    stream.synchronize();
                    (void)tmp;
                }

                std::vector<double> tiled_kernel_times;
                std::vector<double> tiled_e2e_times;
                tiled_kernel_times.reserve(iterations_);
                tiled_e2e_times.reserve(iterations_);

                for (int i = 0; i < iterations_; ++i) {
                    auto t_e2e_0 = std::chrono::high_resolution_clock::now();
                    CUDA_CHECK(cudaMemcpy2DAsync(d_src.data(), d_src.stride_bytes(),
                                                 h_src.raw_data(), h_src.stride_bytes,
                                                 width_bytes, height_, cudaMemcpyHostToDevice, stream));

                    CUDA_CHECK(cudaEventRecord(ev_start, stream));
                    DeviceImage tmp = cuda::mean_blur_tiled_3x3(d_src, cfg, stream);
                    CUDA_CHECK(cudaEventRecord(ev_stop, stream));

                    CUDA_CHECK(cudaMemcpy2DAsync(h_dst.raw_data(), h_dst.stride_bytes,
                                                 tmp.data(), tmp.stride_bytes(),
                                                 width_bytes, height_, cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    auto t_e2e_1 = std::chrono::high_resolution_clock::now();

                    float k_ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_stop));
                    tiled_kernel_times.push_back(static_cast<double>(k_ms));
                    tiled_e2e_times.push_back(std::chrono::duration<double, std::milli>(t_e2e_1 - t_e2e_0).count());
                }

                Statistics tiled_k_stats = calculate_statistics(tiled_kernel_times);
                Statistics tiled_e2e_stats = calculate_statistics(tiled_e2e_times);

                BenchmarkRecord tiled_rec;
                tiled_rec.operation = "mean_blur";
                tiled_rec.image_width = width_;
                tiled_rec.image_height = height_;
                tiled_rec.filter_size = filter_size;
                tiled_rec.implementation = "Tiled Shared Mem";
                tiled_rec.tile_width = tile_dim;
                tiled_rec.tile_height = tile_dim;
                tiled_rec.kernel_time_ms = tiled_k_stats.mean;
                tiled_rec.end_to_end_time_ms = tiled_e2e_stats.mean;
                tiled_rec.median_time_ms = tiled_k_stats.median;
                tiled_rec.mean_time_ms = tiled_k_stats.mean;
                tiled_rec.min_time_ms = tiled_k_stats.min;
                tiled_rec.max_time_ms = tiled_k_stats.max;
                tiled_rec.std_dev_ms = tiled_k_stats.std_dev;
                tiled_rec.speedup_vs_cpu = (tiled_k_stats.mean > 0.0) ? (cpu_stats.mean / tiled_k_stats.mean) : 1.0;
                tiled_rec.speedup_vs_naive = (tiled_k_stats.mean > 0.0) ? (naive_k_stats.mean / tiled_k_stats.mean) : 1.0;
                results.push_back(tiled_rec);
            }
        }

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        return results;
    }

    // --- 3. GAUSSIAN BLUR ---
    std::vector<BenchmarkRecord> run_gaussian_blur() {
        std::vector<BenchmarkRecord> results;
        Image h_src = generate_synthetic_image(width_, height_, 1, 303);
        Image h_dst(width_, height_, 1);
        size_t width_bytes = static_cast<size_t>(width_);

        // A. CPU Reference
        std::vector<double> cpu_times;
        cpu_times.reserve(iterations_);
        for (int i = 0; i < warmup_; ++i) {
            Image tmp = cpu::gaussian_blur_3x3(h_src);
            (void)tmp;
        }
        for (int i = 0; i < iterations_; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Image tmp = cpu::gaussian_blur_3x3(h_src);
            auto t1 = std::chrono::high_resolution_clock::now();
            (void)tmp;
            cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        Statistics cpu_stats = calculate_statistics(cpu_times);

        BenchmarkRecord cpu_rec;
        cpu_rec.operation = "gaussian_blur";
        cpu_rec.image_width = width_;
        cpu_rec.image_height = height_;
        cpu_rec.filter_size = 3;
        cpu_rec.implementation = "CPU Reference";
        cpu_rec.tile_width = 0;
        cpu_rec.tile_height = 0;
        cpu_rec.kernel_time_ms = cpu_stats.mean;
        cpu_rec.end_to_end_time_ms = cpu_stats.mean;
        cpu_rec.median_time_ms = cpu_stats.median;
        cpu_rec.mean_time_ms = cpu_stats.mean;
        cpu_rec.min_time_ms = cpu_stats.min;
        cpu_rec.max_time_ms = cpu_stats.max;
        cpu_rec.std_dev_ms = cpu_stats.std_dev;
        cpu_rec.speedup_vs_cpu = 1.0;
        cpu_rec.speedup_vs_naive = 1.0;
        results.push_back(cpu_rec);

        CudaStream stream;
        DeviceImage d_src(width_, height_, 1);
        DeviceImage d_dst(width_, height_, 1);
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        // Initial copy
        CUDA_CHECK(cudaMemcpy2D(d_src.data(), d_src.stride_bytes(),
                                h_src.raw_data(), h_src.stride_bytes,
                                width_bytes, height_, cudaMemcpyHostToDevice));

        // B. Naive CUDA
        for (int i = 0; i < warmup_; ++i) {
            DeviceImage tmp = cuda::gaussian_blur_naive_3x3(d_src, stream);
            stream.synchronize();
            (void)tmp;
        }

        std::vector<double> naive_kernel_times;
        std::vector<double> naive_e2e_times;
        naive_kernel_times.reserve(iterations_);
        naive_e2e_times.reserve(iterations_);

        for (int i = 0; i < iterations_; ++i) {
            auto t_e2e_0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemcpy2DAsync(d_src.data(), d_src.stride_bytes(),
                                         h_src.raw_data(), h_src.stride_bytes,
                                         width_bytes, height_, cudaMemcpyHostToDevice, stream));

            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            DeviceImage tmp = cuda::gaussian_blur_naive_3x3(d_src, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));

            CUDA_CHECK(cudaMemcpy2DAsync(h_dst.raw_data(), h_dst.stride_bytes,
                                         tmp.data(), tmp.stride_bytes(),
                                         width_bytes, height_, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t_e2e_1 = std::chrono::high_resolution_clock::now();

            float k_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_stop));
            naive_kernel_times.push_back(static_cast<double>(k_ms));
            naive_e2e_times.push_back(std::chrono::duration<double, std::milli>(t_e2e_1 - t_e2e_0).count());
        }

        Statistics naive_k_stats = calculate_statistics(naive_kernel_times);
        Statistics naive_e2e_stats = calculate_statistics(naive_e2e_times);

        BenchmarkRecord naive_rec;
        naive_rec.operation = "gaussian_blur";
        naive_rec.image_width = width_;
        naive_rec.image_height = height_;
        naive_rec.filter_size = 3;
        naive_rec.implementation = "Naive CUDA";
        naive_rec.tile_width = 16;
        naive_rec.tile_height = 16;
        naive_rec.kernel_time_ms = naive_k_stats.mean;
        naive_rec.end_to_end_time_ms = naive_e2e_stats.mean;
        naive_rec.median_time_ms = naive_k_stats.median;
        naive_rec.mean_time_ms = naive_k_stats.mean;
        naive_rec.min_time_ms = naive_k_stats.min;
        naive_rec.max_time_ms = naive_k_stats.max;
        naive_rec.std_dev_ms = naive_k_stats.std_dev;
        naive_rec.speedup_vs_cpu = (naive_k_stats.mean > 0.0) ? (cpu_stats.mean / naive_k_stats.mean) : 1.0;
        naive_rec.speedup_vs_naive = 1.0;
        results.push_back(naive_rec);

        // C. Tiled CUDA (8x8, 16x16, 32x32)
        std::vector<std::pair<TileConfig, int>> configs = {
            {TileConfig::TILE_8x8, 8},
            {TileConfig::TILE_16x16, 16},
            {TileConfig::TILE_32x32, 32}
        };

        for (const auto& [cfg, tile_dim] : configs) {
            for (int i = 0; i < warmup_; ++i) {
                DeviceImage tmp = cuda::gaussian_blur_tiled_3x3(d_src, cfg, stream);
                stream.synchronize();
                (void)tmp;
            }

            std::vector<double> tiled_kernel_times;
            std::vector<double> tiled_e2e_times;
            tiled_kernel_times.reserve(iterations_);
            tiled_e2e_times.reserve(iterations_);

            for (int i = 0; i < iterations_; ++i) {
                auto t_e2e_0 = std::chrono::high_resolution_clock::now();
                CUDA_CHECK(cudaMemcpy2DAsync(d_src.data(), d_src.stride_bytes(),
                                             h_src.raw_data(), h_src.stride_bytes,
                                             width_bytes, height_, cudaMemcpyHostToDevice, stream));

                CUDA_CHECK(cudaEventRecord(ev_start, stream));
                DeviceImage tmp = cuda::gaussian_blur_tiled_3x3(d_src, cfg, stream);
                CUDA_CHECK(cudaEventRecord(ev_stop, stream));

                CUDA_CHECK(cudaMemcpy2DAsync(h_dst.raw_data(), h_dst.stride_bytes,
                                             tmp.data(), tmp.stride_bytes(),
                                             width_bytes, height_, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                auto t_e2e_1 = std::chrono::high_resolution_clock::now();

                float k_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_stop));
                tiled_kernel_times.push_back(static_cast<double>(k_ms));
                tiled_e2e_times.push_back(std::chrono::duration<double, std::milli>(t_e2e_1 - t_e2e_0).count());
            }

            Statistics tiled_k_stats = calculate_statistics(tiled_kernel_times);
            Statistics tiled_e2e_stats = calculate_statistics(tiled_e2e_times);

            BenchmarkRecord tiled_rec;
            tiled_rec.operation = "gaussian_blur";
            tiled_rec.image_width = width_;
            tiled_rec.image_height = height_;
            tiled_rec.filter_size = 3;
            tiled_rec.implementation = "Tiled Shared Mem";
            tiled_rec.tile_width = tile_dim;
            tiled_rec.tile_height = tile_dim;
            tiled_rec.kernel_time_ms = tiled_k_stats.mean;
            tiled_rec.end_to_end_time_ms = tiled_e2e_stats.mean;
            tiled_rec.median_time_ms = tiled_k_stats.median;
            tiled_rec.mean_time_ms = tiled_k_stats.mean;
            tiled_rec.min_time_ms = tiled_k_stats.min;
            tiled_rec.max_time_ms = tiled_k_stats.max;
            tiled_rec.std_dev_ms = tiled_k_stats.std_dev;
            tiled_rec.speedup_vs_cpu = (tiled_k_stats.mean > 0.0) ? (cpu_stats.mean / tiled_k_stats.mean) : 1.0;
            tiled_rec.speedup_vs_naive = (tiled_k_stats.mean > 0.0) ? (naive_k_stats.mean / tiled_k_stats.mean) : 1.0;
            results.push_back(tiled_rec);
        }

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        return results;
    }

    // --- 4. SOBEL EDGE DETECTION ---
    std::vector<BenchmarkRecord> run_sobel() {
        std::vector<BenchmarkRecord> results;
        Image h_src = generate_synthetic_image(width_, height_, 1, 404);
        Image h_dst(width_, height_, 1);
        size_t width_bytes = static_cast<size_t>(width_);

        // A. CPU Reference
        std::vector<double> cpu_times;
        cpu_times.reserve(iterations_);
        for (int i = 0; i < warmup_; ++i) {
            Image tmp = cpu::sobel_edge_detect(h_src);
            (void)tmp;
        }
        for (int i = 0; i < iterations_; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            Image tmp = cpu::sobel_edge_detect(h_src);
            auto t1 = std::chrono::high_resolution_clock::now();
            (void)tmp;
            cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        Statistics cpu_stats = calculate_statistics(cpu_times);

        BenchmarkRecord cpu_rec;
        cpu_rec.operation = "sobel_edge_detect";
        cpu_rec.image_width = width_;
        cpu_rec.image_height = height_;
        cpu_rec.filter_size = 3;
        cpu_rec.implementation = "CPU Reference";
        cpu_rec.tile_width = 0;
        cpu_rec.tile_height = 0;
        cpu_rec.kernel_time_ms = cpu_stats.mean;
        cpu_rec.end_to_end_time_ms = cpu_stats.mean;
        cpu_rec.median_time_ms = cpu_stats.median;
        cpu_rec.mean_time_ms = cpu_stats.mean;
        cpu_rec.min_time_ms = cpu_stats.min;
        cpu_rec.max_time_ms = cpu_stats.max;
        cpu_rec.std_dev_ms = cpu_stats.std_dev;
        cpu_rec.speedup_vs_cpu = 1.0;
        cpu_rec.speedup_vs_naive = 1.0;
        results.push_back(cpu_rec);

        CudaStream stream;
        DeviceImage d_src(width_, height_, 1);
        DeviceImage d_dst(width_, height_, 1);
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        // Initial copy
        CUDA_CHECK(cudaMemcpy2D(d_src.data(), d_src.stride_bytes(),
                                h_src.raw_data(), h_src.stride_bytes,
                                width_bytes, height_, cudaMemcpyHostToDevice));

        // B. Naive CUDA
        for (int i = 0; i < warmup_; ++i) {
            DeviceImage tmp = cuda::sobel_edge_detect_naive(d_src, stream);
            stream.synchronize();
            (void)tmp;
        }

        std::vector<double> naive_kernel_times;
        std::vector<double> naive_e2e_times;
        naive_kernel_times.reserve(iterations_);
        naive_e2e_times.reserve(iterations_);

        for (int i = 0; i < iterations_; ++i) {
            auto t_e2e_0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemcpy2DAsync(d_src.data(), d_src.stride_bytes(),
                                         h_src.raw_data(), h_src.stride_bytes,
                                         width_bytes, height_, cudaMemcpyHostToDevice, stream));

            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            DeviceImage tmp = cuda::sobel_edge_detect_naive(d_src, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));

            CUDA_CHECK(cudaMemcpy2DAsync(h_dst.raw_data(), h_dst.stride_bytes,
                                         tmp.data(), tmp.stride_bytes(),
                                         width_bytes, height_, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t_e2e_1 = std::chrono::high_resolution_clock::now();

            float k_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_stop));
            naive_kernel_times.push_back(static_cast<double>(k_ms));
            naive_e2e_times.push_back(std::chrono::duration<double, std::milli>(t_e2e_1 - t_e2e_0).count());
        }

        Statistics naive_k_stats = calculate_statistics(naive_kernel_times);
        Statistics naive_e2e_stats = calculate_statistics(naive_e2e_times);

        BenchmarkRecord naive_rec;
        naive_rec.operation = "sobel_edge_detect";
        naive_rec.image_width = width_;
        naive_rec.image_height = height_;
        naive_rec.filter_size = 3;
        naive_rec.implementation = "Naive CUDA";
        naive_rec.tile_width = 16;
        naive_rec.tile_height = 16;
        naive_rec.kernel_time_ms = naive_k_stats.mean;
        naive_rec.end_to_end_time_ms = naive_e2e_stats.mean;
        naive_rec.median_time_ms = naive_k_stats.median;
        naive_rec.mean_time_ms = naive_k_stats.mean;
        naive_rec.min_time_ms = naive_k_stats.min;
        naive_rec.max_time_ms = naive_k_stats.max;
        naive_rec.std_dev_ms = naive_k_stats.std_dev;
        naive_rec.speedup_vs_cpu = (naive_k_stats.mean > 0.0) ? (cpu_stats.mean / naive_k_stats.mean) : 1.0;
        naive_rec.speedup_vs_naive = 1.0;
        results.push_back(naive_rec);

        // C. Tiled CUDA (8x8, 16x16, 32x32)
        std::vector<std::pair<TileConfig, int>> configs = {
            {TileConfig::TILE_8x8, 8},
            {TileConfig::TILE_16x16, 16},
            {TileConfig::TILE_32x32, 32}
        };

        for (const auto& [cfg, tile_dim] : configs) {
            for (int i = 0; i < warmup_; ++i) {
                DeviceImage tmp = cuda::sobel_edge_detect_tiled(d_src, cfg, stream);
                stream.synchronize();
                (void)tmp;
            }

            std::vector<double> tiled_kernel_times;
            std::vector<double> tiled_e2e_times;
            tiled_kernel_times.reserve(iterations_);
            tiled_e2e_times.reserve(iterations_);

            for (int i = 0; i < iterations_; ++i) {
                auto t_e2e_0 = std::chrono::high_resolution_clock::now();
                CUDA_CHECK(cudaMemcpy2DAsync(d_src.data(), d_src.stride_bytes(),
                                             h_src.raw_data(), h_src.stride_bytes,
                                             width_bytes, height_, cudaMemcpyHostToDevice, stream));

                CUDA_CHECK(cudaEventRecord(ev_start, stream));
                DeviceImage tmp = cuda::sobel_edge_detect_tiled(d_src, cfg, stream);
                CUDA_CHECK(cudaEventRecord(ev_stop, stream));

                CUDA_CHECK(cudaMemcpy2DAsync(h_dst.raw_data(), h_dst.stride_bytes,
                                             tmp.data(), tmp.stride_bytes(),
                                             width_bytes, height_, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                auto t_e2e_1 = std::chrono::high_resolution_clock::now();

                float k_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_stop));
                tiled_kernel_times.push_back(static_cast<double>(k_ms));
                tiled_e2e_times.push_back(std::chrono::duration<double, std::milli>(t_e2e_1 - t_e2e_0).count());
            }

            Statistics tiled_k_stats = calculate_statistics(tiled_kernel_times);
            Statistics tiled_e2e_stats = calculate_statistics(tiled_e2e_times);

            BenchmarkRecord tiled_rec;
            tiled_rec.operation = "sobel_edge_detect";
            tiled_rec.image_width = width_;
            tiled_rec.image_height = height_;
            tiled_rec.filter_size = 3;
            tiled_rec.implementation = "Tiled Shared Mem";
            tiled_rec.tile_width = tile_dim;
            tiled_rec.tile_height = tile_dim;
            tiled_rec.kernel_time_ms = tiled_k_stats.mean;
            tiled_rec.end_to_end_time_ms = tiled_e2e_stats.mean;
            tiled_rec.median_time_ms = tiled_k_stats.median;
            tiled_rec.mean_time_ms = tiled_k_stats.mean;
            tiled_rec.min_time_ms = tiled_k_stats.min;
            tiled_rec.max_time_ms = tiled_k_stats.max;
            tiled_rec.std_dev_ms = tiled_k_stats.std_dev;
            tiled_rec.speedup_vs_cpu = (tiled_k_stats.mean > 0.0) ? (cpu_stats.mean / tiled_k_stats.mean) : 1.0;
            tiled_rec.speedup_vs_naive = (tiled_k_stats.mean > 0.0) ? (naive_k_stats.mean / tiled_k_stats.mean) : 1.0;
            results.push_back(tiled_rec);
        }

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        return results;
    }

private:
    int width_;
    int height_;
    int iterations_;
    int warmup_;
};

} // namespace benchmark
} // namespace cuda_image_proc

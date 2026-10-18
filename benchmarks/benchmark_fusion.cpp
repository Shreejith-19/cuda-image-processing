#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <fstream>
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

Image create_synthetic_rgb(int width, int height, unsigned int seed = 42) {
    Image img(width, height, 3);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = img.row_ptr(y);
        for (int x = 0; x < width * 3; ++x) {
            row[x] = static_cast<uint8_t>(dist(rng));
        }
    }
    return img;
}

struct TimingStats {
    double min_ms{0.0};
    double max_ms{0.0};
    double median_ms{0.0};
    double mean_ms{0.0};
    double std_dev_ms{0.0};
};

TimingStats compute_timing_stats(std::vector<double>& samples) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());

    TimingStats stats;
    stats.min_ms = samples.front();
    stats.max_ms = samples.back();

    size_t n = samples.size();
    if (n % 2 == 0) {
        stats.median_ms = (samples[n / 2 - 1] + samples[n / 2]) * 0.5;
    } else {
        stats.median_ms = samples[n / 2];
    }

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean_ms = sum / static_cast<double>(n);

    double sq_diff_sum = 0.0;
    for (double val : samples) {
        sq_diff_sum += (val - stats.mean_ms) * (val - stats.mean_ms);
    }
    stats.std_dev_ms = std::sqrt(sq_diff_sum / static_cast<double>(n));

    return stats;
}

struct FusionBenchRow {
    std::string implementation;
    int width;
    int height;
    int kernel_launches;
    size_t intermediate_bytes_saved;
    TimingStats kernel_stats;
    TimingStats e2e_stats;
    double speedup_vs_baseline_kernel{1.0};
    double speedup_vs_baseline_e2e{1.0};
    bool bit_exact{true};
};

int main(int argc, char* argv[]) {
    std::cout << "========================================================================================\n";
    std::cout << " CUDA Kernel Fusion Benchmark: RGB-to-Grayscale + 3x3 Mean Blur Pipeline               \n";
    std::cout << "========================================================================================\n";

    std::vector<std::pair<int, int>> resolutions = {
        {256, 256},
        {512, 512},
        {1024, 1024},
        {2048, 2048}
    };

    const int warmup_runs = 10;
    const int timed_runs = 50;

    std::vector<FusionBenchRow> all_rows;

    for (const auto& res : resolutions) {
        int w = res.first;
        int h = res.second;
        std::cout << "\n>>> Benchmarking Resolution: " << w << " x " << h << " (RGB 3-channel, "
                  << (w * h * 3) / (1024 * 1024.0) << " MB input) <<<\n";

        Image h_rgb = create_synthetic_rgb(w, h);

        // Ground-truth CPU reference run
        auto cpu_start = std::chrono::high_resolution_clock::now();
        Image h_cpu_ref = cpu::rgb_to_mean_blur(h_rgb, 3);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        std::cout << "  CPU Reference execution time: " << std::fixed << std::setprecision(3) << cpu_ms << " ms\n";

        CudaStream stream;
        DeviceImage d_rgb = to_device(h_rgb, stream);
        stream.synchronize();

        // -------------------------------------------------------------------
        // 1. Baseline Pipeline: Separate Grayscale + Mean Blur
        // -------------------------------------------------------------------
        std::cout << "  Profiling Baseline Pipeline (Separate Kernels)...\n";
        // Warm-up
        for (int i = 0; i < warmup_runs; ++i) {
            DeviceImage d_out = cuda::rgb_to_mean_blur_baseline(d_rgb, 3, stream.get());
        }
        stream.synchronize();

        cudaEvent_t start_ev, stop_ev;
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));

        std::vector<double> base_kernel_times(timed_runs);
        std::vector<double> base_e2e_times(timed_runs);
        Image base_out_host;

        for (int i = 0; i < timed_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            DeviceImage d_in = to_device(h_rgb, stream);

            CUDA_CHECK(cudaEventRecord(start_ev, stream.get()));
            DeviceImage d_out = cuda::rgb_to_mean_blur_baseline(d_in, 3, stream.get());
            CUDA_CHECK(cudaEventRecord(stop_ev, stream.get()));

            base_out_host = to_host(d_out, stream);
            stream.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
            base_kernel_times[i] = ms;
            base_e2e_times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        TimingStats base_k_stats = compute_timing_stats(base_kernel_times);
        TimingStats base_e2e_stats = compute_timing_stats(base_e2e_times);
        bool base_exact = (cpu::compute_diff_stats(h_cpu_ref, base_out_host).max_diff <= 1.0);

        FusionBenchRow row_base;
        row_base.implementation = "Baseline Separate Pipeline (2 kernels)";
        row_base.width = w;
        row_base.height = h;
        row_base.kernel_launches = 2;
        row_base.intermediate_bytes_saved = 0;
        row_base.kernel_stats = base_k_stats;
        row_base.e2e_stats = base_e2e_stats;
        row_base.speedup_vs_baseline_kernel = 1.0;
        row_base.speedup_vs_baseline_e2e = 1.0;
        row_base.bit_exact = base_exact;
        all_rows.push_back(row_base);

        // -------------------------------------------------------------------
        // 2. Fused Tiled Shared Memory (16x16)
        // -------------------------------------------------------------------
        std::cout << "  Profiling Fused Tiled CUDA (16x16)...\n";
        for (int i = 0; i < warmup_runs; ++i) {
            DeviceImage d_out = cuda::rgb_to_mean_blur_fused(d_rgb, cuda::TileConfig::TILE_16x16, stream.get());
        }
        stream.synchronize();

        std::vector<double> fused16_kernel_times(timed_runs);
        std::vector<double> fused16_e2e_times(timed_runs);
        Image fused16_out_host;

        for (int i = 0; i < timed_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            DeviceImage d_in = to_device(h_rgb, stream);

            CUDA_CHECK(cudaEventRecord(start_ev, stream.get()));
            DeviceImage d_out = cuda::rgb_to_mean_blur_fused(d_in, cuda::TileConfig::TILE_16x16, stream.get());
            CUDA_CHECK(cudaEventRecord(stop_ev, stream.get()));

            fused16_out_host = to_host(d_out, stream);
            stream.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
            fused16_kernel_times[i] = ms;
            fused16_e2e_times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        TimingStats fused16_k_stats = compute_timing_stats(fused16_kernel_times);
        TimingStats fused16_e2e_stats = compute_timing_stats(fused16_e2e_times);
        bool fused16_exact = (cpu::compute_diff_stats(base_out_host, fused16_out_host).mismatched_pixels == 0);

        FusionBenchRow row_fused16;
        row_fused16.implementation = "Fused Tiled CUDA (16x16)";
        row_fused16.width = w;
        row_fused16.height = h;
        row_fused16.kernel_launches = 1;
        row_fused16.intermediate_bytes_saved = static_cast<size_t>(w) * static_cast<size_t>(h) * 2; // 1 byte write + 1 byte read
        row_fused16.kernel_stats = fused16_k_stats;
        row_fused16.e2e_stats = fused16_e2e_stats;
        row_fused16.speedup_vs_baseline_kernel = base_k_stats.mean_ms / fused16_k_stats.mean_ms;
        row_fused16.speedup_vs_baseline_e2e = base_e2e_stats.mean_ms / fused16_e2e_stats.mean_ms;
        row_fused16.bit_exact = fused16_exact;
        all_rows.push_back(row_fused16);

        // -------------------------------------------------------------------
        // 3. Fused Tiled Shared Memory (8x8)
        // -------------------------------------------------------------------
        std::cout << "  Profiling Fused Tiled CUDA (8x8)...\n";
        for (int i = 0; i < warmup_runs; ++i) {
            DeviceImage d_out = cuda::rgb_to_mean_blur_fused(d_rgb, cuda::TileConfig::TILE_8x8, stream.get());
        }
        stream.synchronize();

        std::vector<double> fused8_kernel_times(timed_runs);
        std::vector<double> fused8_e2e_times(timed_runs);
        Image fused8_out_host;

        for (int i = 0; i < timed_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            DeviceImage d_in = to_device(h_rgb, stream);

            CUDA_CHECK(cudaEventRecord(start_ev, stream.get()));
            DeviceImage d_out = cuda::rgb_to_mean_blur_fused(d_in, cuda::TileConfig::TILE_8x8, stream.get());
            CUDA_CHECK(cudaEventRecord(stop_ev, stream.get()));

            fused8_out_host = to_host(d_out, stream);
            stream.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
            fused8_kernel_times[i] = ms;
            fused8_e2e_times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        TimingStats fused8_k_stats = compute_timing_stats(fused8_kernel_times);
        TimingStats fused8_e2e_stats = compute_timing_stats(fused8_e2e_times);
        bool fused8_exact = (cpu::compute_diff_stats(base_out_host, fused8_out_host).mismatched_pixels == 0);

        FusionBenchRow row_fused8;
        row_fused8.implementation = "Fused Tiled CUDA (8x8)";
        row_fused8.width = w;
        row_fused8.height = h;
        row_fused8.kernel_launches = 1;
        row_fused8.intermediate_bytes_saved = static_cast<size_t>(w) * static_cast<size_t>(h) * 2;
        row_fused8.kernel_stats = fused8_k_stats;
        row_fused8.e2e_stats = fused8_e2e_stats;
        row_fused8.speedup_vs_baseline_kernel = base_k_stats.mean_ms / fused8_k_stats.mean_ms;
        row_fused8.speedup_vs_baseline_e2e = base_e2e_stats.mean_ms / fused8_e2e_stats.mean_ms;
        row_fused8.bit_exact = fused8_exact;
        all_rows.push_back(row_fused8);

        // -------------------------------------------------------------------
        // 4. Fused Tiled Shared Memory (32x32)
        // -------------------------------------------------------------------
        std::cout << "  Profiling Fused Tiled CUDA (32x32)...\n";
        for (int i = 0; i < warmup_runs; ++i) {
            DeviceImage d_out = cuda::rgb_to_mean_blur_fused(d_rgb, cuda::TileConfig::TILE_32x32, stream.get());
        }
        stream.synchronize();

        std::vector<double> fused32_kernel_times(timed_runs);
        std::vector<double> fused32_e2e_times(timed_runs);
        Image fused32_out_host;

        for (int i = 0; i < timed_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            DeviceImage d_in = to_device(h_rgb, stream);

            CUDA_CHECK(cudaEventRecord(start_ev, stream.get()));
            DeviceImage d_out = cuda::rgb_to_mean_blur_fused(d_in, cuda::TileConfig::TILE_32x32, stream.get());
            CUDA_CHECK(cudaEventRecord(stop_ev, stream.get()));

            fused32_out_host = to_host(d_out, stream);
            stream.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
            fused32_kernel_times[i] = ms;
            fused32_e2e_times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        TimingStats fused32_k_stats = compute_timing_stats(fused32_kernel_times);
        TimingStats fused32_e2e_stats = compute_timing_stats(fused32_e2e_times);
        bool fused32_exact = (cpu::compute_diff_stats(base_out_host, fused32_out_host).mismatched_pixels == 0);

        FusionBenchRow row_fused32;
        row_fused32.implementation = "Fused Tiled CUDA (32x32)";
        row_fused32.width = w;
        row_fused32.height = h;
        row_fused32.kernel_launches = 1;
        row_fused32.intermediate_bytes_saved = static_cast<size_t>(w) * static_cast<size_t>(h) * 2;
        row_fused32.kernel_stats = fused32_k_stats;
        row_fused32.e2e_stats = fused32_e2e_stats;
        row_fused32.speedup_vs_baseline_kernel = base_k_stats.mean_ms / fused32_k_stats.mean_ms;
        row_fused32.speedup_vs_baseline_e2e = base_e2e_stats.mean_ms / fused32_e2e_stats.mean_ms;
        row_fused32.bit_exact = fused32_exact;
        all_rows.push_back(row_fused32);

        // -------------------------------------------------------------------
        // 5. Fused Naive CUDA (No Shared Memory)
        // -------------------------------------------------------------------
        std::cout << "  Profiling Fused Naive CUDA (Global-only)...\n";
        for (int i = 0; i < warmup_runs; ++i) {
            DeviceImage d_out = cuda::rgb_to_mean_blur_fused_naive(d_rgb, 3, stream.get());
        }
        stream.synchronize();

        std::vector<double> naive_kernel_times(timed_runs);
        std::vector<double> naive_e2e_times(timed_runs);
        Image naive_out_host;

        for (int i = 0; i < timed_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            DeviceImage d_in = to_device(h_rgb, stream);

            CUDA_CHECK(cudaEventRecord(start_ev, stream.get()));
            DeviceImage d_out = cuda::rgb_to_mean_blur_fused_naive(d_in, 3, stream.get());
            CUDA_CHECK(cudaEventRecord(stop_ev, stream.get()));

            naive_out_host = to_host(d_out, stream);
            stream.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
            naive_kernel_times[i] = ms;
            naive_e2e_times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        TimingStats naive_k_stats = compute_timing_stats(naive_kernel_times);
        TimingStats naive_e2e_stats = compute_timing_stats(naive_e2e_times);
        bool naive_exact = (cpu::compute_diff_stats(base_out_host, naive_out_host).mismatched_pixels == 0);

        FusionBenchRow row_naive;
        row_naive.implementation = "Fused Naive CUDA (Global-only)";
        row_naive.width = w;
        row_naive.height = h;
        row_naive.kernel_launches = 1;
        row_naive.intermediate_bytes_saved = static_cast<size_t>(w) * static_cast<size_t>(h) * 2;
        row_naive.kernel_stats = naive_k_stats;
        row_naive.e2e_stats = naive_e2e_stats;
        row_naive.speedup_vs_baseline_kernel = base_k_stats.mean_ms / naive_k_stats.mean_ms;
        row_naive.speedup_vs_baseline_e2e = base_e2e_stats.mean_ms / naive_e2e_stats.mean_ms;
        row_naive.bit_exact = naive_exact;
        all_rows.push_back(row_naive);

        CUDA_CHECK(cudaEventDestroy(start_ev));
        CUDA_CHECK(cudaEventDestroy(stop_ev));
    }

    // Print Pretty Comparison Table
    std::cout << "\n========================================================================================================================\n";
    std::cout << " SUMMARY BENCHMARK RESULTS: Kernel Fusion vs. Baseline Pipeline                                                          \n";
    std::cout << "========================================================================================================================\n";
    std::cout << std::left
              << std::setw(12) << "Resolution"
              << std::setw(38) << "Implementation"
              << std::setw(10) << "Launches"
              << std::setw(16) << "Saved Mem (KB)"
              << std::setw(14) << "Kernel (ms)"
              << std::setw(14) << "E2E (ms)"
              << std::setw(16) << "Kernel Speedup"
              << std::setw(12) << "Exact Bit"
              << "\n";
    std::cout << "------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : all_rows) {
        std::string res_str = std::to_string(r.width) + "x" + std::to_string(r.height);
        std::cout << std::left
                  << std::setw(12) << res_str
                  << std::setw(38) << r.implementation
                  << std::setw(10) << r.kernel_launches
                  << std::setw(16) << (r.intermediate_bytes_saved / 1024)
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << r.kernel_stats.mean_ms
                  << std::setw(14) << r.e2e_stats.mean_ms
                  << std::setprecision(2)
                  << std::setw(16) << (std::to_string(r.speedup_vs_baseline_kernel) + "x")
                  << std::setw(12) << (r.bit_exact ? "YES" : "NO")
                  << "\n";
    }
    std::cout << "========================================================================================================================\n";

    // Export CSV
    fs::create_directories("results");
    std::string csv_path = "results/benchmark_fusion.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "resolution,implementation,width,height,kernel_launches,intermediate_bytes_saved,kernel_mean_ms,kernel_median_ms,kernel_min_ms,kernel_max_ms,kernel_std_dev_ms,e2e_mean_ms,e2e_median_ms,kernel_speedup_vs_baseline,e2e_speedup_vs_baseline,bit_exact\n";
        for (const auto& r : all_rows) {
            csv << r.width << "x" << r.height << ","
                << "\"" << r.implementation << "\","
                << r.width << ","
                << r.height << ","
                << r.kernel_launches << ","
                << r.intermediate_bytes_saved << ","
                << r.kernel_stats.mean_ms << ","
                << r.kernel_stats.median_ms << ","
                << r.kernel_stats.min_ms << ","
                << r.kernel_stats.max_ms << ","
                << r.kernel_stats.std_dev_ms << ","
                << r.e2e_stats.mean_ms << ","
                << r.e2e_stats.median_ms << ","
                << r.speedup_vs_baseline_kernel << ","
                << r.speedup_vs_baseline_e2e << ","
                << (r.bit_exact ? "true" : "false") << "\n";
        }
        std::cout << "\nResults exported successfully to: " << csv_path << "\n";
    } else {
        std::cerr << "Failed to export CSV to " << csv_path << "\n";
    }

    return 0;
}

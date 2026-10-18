#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include <cassert>
#include "cuda_image_proc/image.hpp"
#include "cuda_image_proc/image_io.hpp"
#include "cuda_image_proc/cpu_ops.hpp"
#include "cuda_image_proc/cuda_ops.cuh"

namespace fs = std::filesystem;
using namespace cuda_image_proc;

// Helper to create synthetic RGB test image
Image create_rgb_test_pattern(int width, int height, unsigned int seed = 42) {
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

std::string resolve_image_path(const std::string& rel_path) {
    if (fs::exists(rel_path)) return rel_path;
    if (fs::exists("../" + rel_path)) return "../" + rel_path;
    return rel_path;
}

struct TestCase {
    std::string name;
    int width;
    int height;
    bool from_file{false};
    std::string filepath{""};
};

int main() {
    std::cout << "========================================================================================\n";
    std::cout << " Kernel Fusion Correctness & Robustness Suite (RGB-to-Grayscale + 3x3 Mean Blur)       \n";
    std::cout << "========================================================================================\n";

    std::vector<TestCase> test_cases = {
        {"1. Single pixel (1x1)", 1, 1, false, ""},
        {"2. Small filter dimension (3x3)", 3, 3, false, ""},
        {"3. Asymmetric odd dimensions (7x5)", 7, 5, false, ""},
        {"4. Non-block multiple primes (17x19)", 17, 19, false, ""},
        {"5. Boundary spanning (31x33)", 31, 33, false, ""},
        {"6. Standard block multiple (256x256)", 256, 256, false, ""},
        {"7. Large irregular dimensions (831x517)", 831, 517, false, ""},
        {"8. Real image (images/input/sample.png)", 0, 0, true, "images/input/sample.png"}
    };

    int total_tests = 0;
    int passed_tests = 0;

    for (const auto& tc : test_cases) {
        Image input_rgb;
        if (tc.from_file) {
            std::string path = resolve_image_path(tc.filepath);
            if (!fs::exists(path)) {
                std::cout << "Skipping file test: " << path << " not found.\n";
                continue;
            }
            input_rgb = load_image(path);
        } else {
            input_rgb = create_rgb_test_pattern(tc.width, tc.height);
        }

        std::cout << "\n--- Running: " << tc.name << " (" << input_rgb.width << "x" << input_rgb.height << ", 3 channels) ---\n";

        // 1. CPU Reference Output
        Image cpu_out = cpu::rgb_to_mean_blur(input_rgb, 3);

        // 2. CUDA Baseline Pipeline (Separate rgb_to_grayscale + mean_blur_tiled)
        Image cuda_baseline_out = cuda::rgb_to_mean_blur_baseline(input_rgb, 3);

        // 3. CUDA Fused Naive
        Image cuda_fused_naive_out = cuda::rgb_to_mean_blur_fused_naive(input_rgb, 3);

        // 4. CUDA Fused Tiled (8x8, 16x16, 32x32)
        Image cuda_fused_t8_out = cuda::rgb_to_mean_blur_fused(input_rgb, cuda::TileConfig::TILE_8x8);
        Image cuda_fused_t16_out = cuda::rgb_to_mean_blur_fused(input_rgb, cuda::TileConfig::TILE_16x16);
        Image cuda_fused_t32_out = cuda::rgb_to_mean_blur_fused(input_rgb, cuda::TileConfig::TILE_32x32);

        // Sub-test A: CUDA Baseline vs. CPU Reference (Tolerance: <= 1 LSB due to GPU FMA contraction)
        total_tests++;
        cpu::ImageDiffStats stats_base_cpu = cpu::compute_diff_stats(cpu_out, cuda_baseline_out);
        bool cpu_pass = (stats_base_cpu.max_diff <= 1.0) && (stats_base_cpu.mean_absolute_error < 0.01);
        if (cpu_pass) {
            std::cout << "  [PASS] Baseline GPU vs. CPU Reference: Valid (Mismatches: " << stats_base_cpu.mismatched_pixels 
                      << ", Max Diff: " << stats_base_cpu.max_diff << ", MAE: " << stats_base_cpu.mean_absolute_error << ")\n";
            passed_tests++;
        } else {
            std::cerr << "  [FAIL] Baseline GPU vs. CPU Reference exceeded tolerance! Max Diff: " << stats_base_cpu.max_diff << "\n";
        }

        // Sub-test B: Fused Tiled (16x16) vs. Baseline GPU Pipeline
        total_tests++;
        cpu::ImageDiffStats stats_fused_base = cpu::compute_diff_stats(cuda_baseline_out, cuda_fused_t16_out);
        if (stats_fused_base.mismatched_pixels == 0) {
            std::cout << "  [PASS] Fused Tiled (16x16) vs. Baseline Pipeline: Exact Match (0 mismatches, max diff: 0)\n";
            passed_tests++;
        } else {
            std::cerr << "  [FAIL] Fused Tiled (16x16) vs. Baseline Pipeline mismatch! Mismatches: " << stats_fused_base.mismatched_pixels 
                      << ", max diff: " << stats_fused_base.max_diff << "\n";
        }

        // Sub-test C: Fused Naive vs. Baseline GPU Pipeline
        total_tests++;
        cpu::ImageDiffStats stats_fused_naive = cpu::compute_diff_stats(cuda_baseline_out, cuda_fused_naive_out);
        if (stats_fused_naive.mismatched_pixels == 0) {
            std::cout << "  [PASS] Fused Naive vs. Baseline Pipeline: Exact Match (0 mismatches, max diff: 0)\n";
            passed_tests++;
        } else {
            std::cerr << "  [FAIL] Fused Naive vs. Baseline Pipeline mismatch! Mismatches: " << stats_fused_naive.mismatched_pixels << "\n";
        }

        // Sub-test D: Tile Configuration Invariance (8x8 vs. 16x16 vs. 32x32)
        total_tests++;
        cpu::ImageDiffStats stats_t8_t16 = cpu::compute_diff_stats(cuda_fused_t8_out, cuda_fused_t16_out);
        cpu::ImageDiffStats stats_t32_t16 = cpu::compute_diff_stats(cuda_fused_t32_out, cuda_fused_t16_out);
        if (stats_t8_t16.mismatched_pixels == 0 && stats_t32_t16.mismatched_pixels == 0) {
            std::cout << "  [PASS] Tile Invariance (8x8 == 16x16 == 32x32): Exact Match across all tile layouts\n";
            passed_tests++;
        } else {
            std::cerr << "  [FAIL] Tile layout inconsistency! 8x8 vs 16x16: " << stats_t8_t16.mismatched_pixels
                      << ", 32x32 vs 16x16: " << stats_t32_t16.mismatched_pixels << "\n";
        }
    }

    // -----------------------------------------------------------------------
    // Robustness & Exception Handling Checks
    // -----------------------------------------------------------------------
    std::cout << "\n--- Running Robustness & Invalid Input Checks ---\n";

    // 1. Empty image check
    total_tests++;
    try {
        Image empty_img;
        cuda::rgb_to_mean_blur_fused(empty_img);
        std::cerr << "  [FAIL] Expected exception on empty image, but none was thrown\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  [PASS] Empty image correctly threw std::invalid_argument\n";
        passed_tests++;
    }

    // 2. Grayscale input to RGB function
    total_tests++;
    try {
        Image gray_img(64, 64, 1);
        cuda::rgb_to_mean_blur_fused(gray_img);
        std::cerr << "  [FAIL] Expected exception on 1-channel image to rgb_to_mean_blur_fused, but none was thrown\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  [PASS] Non-RGB (1-channel) image correctly threw std::invalid_argument\n";
        passed_tests++;
    }

    // 3. Invalid filter size in naive fused kernel
    total_tests++;
    try {
        Image rgb_img(64, 64, 3);
        cuda::rgb_to_mean_blur_fused_naive(rgb_img, 4); // Even filter size
        std::cerr << "  [FAIL] Expected exception on even filter size (4), but none was thrown\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  [PASS] Even filter size correctly threw std::invalid_argument\n";
        passed_tests++;
    }

    std::cout << "\n========================================================================================\n";
    std::cout << " Suite Summary: " << passed_tests << " / " << total_tests << " tests passed.\n";
    std::cout << "========================================================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}

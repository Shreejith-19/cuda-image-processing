#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include "cuda_image_proc/image.hpp"
#include "cuda_image_proc/image_io.hpp"
#include "cuda_image_proc/cpu_ops.hpp"
#include "cuda_image_proc/cuda_ops.cuh"

namespace fs = std::filesystem;
using namespace cuda_image_proc;

std::string resolve_image_path(const std::string& rel_path) {
    if (fs::exists(rel_path)) return rel_path;
    if (fs::exists("../" + rel_path)) return "../" + rel_path;
    return rel_path;
}

Image create_grayscale_pattern(int width, int height, unsigned int seed = 42) {
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

struct TestCase {
    std::string name;
    int width;
    int height;
    bool from_file{false};
    std::string filepath{""};
};

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " 3x3 Gaussian Blur: CPU Reference vs. Naive CUDA vs. Tiled Shared-Memory CUDA Suite               \n";
    std::cout << "==================================================================================================\n";

    std::vector<TestCase> cases = {
        {"1. Single pixel (1x1)", 1, 1, false, ""},
        {"2. Small non-block multiple (17x19)", 17, 19, false, ""},
        {"3. Block multiple boundary (31x33)", 31, 33, false, ""},
        {"4. Exact block multiple 16x16 (256x256)", 256, 256, false, ""},
        {"5. Prime dimensions non-multiple (127x97)", 127, 97, false, ""},
        {"6. Large arbitrary non-multiple (831x517)", 831, 517, false, ""},
        {"7. Real image (images/input/sample.png)", 0, 0, true, "images/input/sample.png"}
    };

    bool all_passed = true;

    std::cout << std::left << std::setw(46) << "Test Case / Kernel Variant"
              << std::setw(14) << "Dimensions"
              << std::setw(12) << "Match Rate"
              << std::setw(12) << "Max Diff"
              << std::setw(10) << "MAE"
              << "Status\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& tc : cases) {
        Image input_gray;
        if (tc.from_file) {
            try {
                std::string actual_path = resolve_image_path(tc.filepath);
                input_gray = load_image(actual_path, 1);
            } catch (const std::exception& e) {
                std::cerr << "Could not load test image " << tc.filepath << ": " << e.what() << "\n";
                all_passed = false;
                continue;
            }
        } else {
            input_gray = create_grayscale_pattern(tc.width, tc.height);
        }

        std::string dim_str = std::to_string(input_gray.width) + "x" + std::to_string(input_gray.height);

        // 1. CPU Reference Execution
        Image cpu_blurred = cpu::gaussian_blur_3x3(input_gray);

        // 2. Naive CUDA Execution
        Image naive_gpu = cuda::gaussian_blur_naive_3x3(input_gray);
        cpu::ImageDiffStats naive_stats = cpu::compute_diff_stats(cpu_blurred, naive_gpu);
        double naive_match_pct = 100.0 * static_cast<double>(naive_stats.identical_pixels) / naive_stats.total_pixels;
        bool naive_pass = (naive_stats.mismatched_pixels == 0) && (naive_stats.max_diff == 0.0);
        if (!naive_pass) all_passed = false;

        std::cout << std::left << std::setw(46) << tc.name
                  << std::setw(14) << dim_str
                  << std::fixed << std::setprecision(2) << std::setw(11) << naive_match_pct << "%"
                  << std::fixed << std::setprecision(1) << std::setw(12) << naive_stats.max_diff
                  << std::fixed << std::setprecision(4) << std::setw(10) << naive_stats.mean_absolute_error
                  << (naive_pass ? "[PASSED]" : "[FAILED]") << "\n";

        // 3. Tiled Configurations: 8x8, 16x16, 32x32
        std::vector<std::pair<std::string, cuda::TileConfig>> tiled_cfgs = {
            {"   └─ Tiled Shared-Mem 8x8", cuda::TileConfig::TILE_8x8},
            {"   └─ Tiled Shared-Mem 16x16", cuda::TileConfig::TILE_16x16},
            {"   └─ Tiled Shared-Mem 32x32", cuda::TileConfig::TILE_32x32}
        };

        for (const auto& [cfg_name, cfg] : tiled_cfgs) {
            Image tiled_gpu = cuda::gaussian_blur_tiled_3x3(input_gray, cfg);
            cpu::ImageDiffStats tiled_stats = cpu::compute_diff_stats(cpu_blurred, tiled_gpu);

            double t_match_pct = 100.0 * static_cast<double>(tiled_stats.identical_pixels) / tiled_stats.total_pixels;
            bool t_pass = (tiled_stats.mismatched_pixels == 0) && (tiled_stats.max_diff == 0.0);
            if (!t_pass) all_passed = false;

            std::cout << std::left << std::setw(46) << cfg_name
                      << std::setw(14) << dim_str
                      << std::fixed << std::setprecision(2) << std::setw(11) << t_match_pct << "%"
                      << std::fixed << std::setprecision(1) << std::setw(12) << tiled_stats.max_diff
                      << std::fixed << std::setprecision(4) << std::setw(10) << tiled_stats.mean_absolute_error
                      << (t_pass ? "[PASSED]" : "[FAILED]") << "\n";
        }

        // Save real image outputs for visual inspection
        if (tc.from_file) {
            fs::create_directories("images/output");
            save_image("images/output/sample_gray_gaussian_cpu.png", cpu_blurred);
            save_image("images/output/sample_gray_gaussian_naive_cuda.png", naive_gpu);
            save_image("images/output/sample_gray_gaussian_tiled_8x8.png", cuda::gaussian_blur_tiled_3x3(input_gray, cuda::TileConfig::TILE_8x8));
            save_image("images/output/sample_gray_gaussian_tiled_16x16.png", cuda::gaussian_blur_tiled_3x3(input_gray, cuda::TileConfig::TILE_16x16));
            save_image("images/output/sample_gray_gaussian_tiled_32x32.png", cuda::gaussian_blur_tiled_3x3(input_gray, cuda::TileConfig::TILE_32x32));

            std::cout << "    -> Saved outputs to:\n"
                      << "       CPU Reference   : images/output/sample_gray_gaussian_cpu.png\n"
                      << "       Naive CUDA      : images/output/sample_gray_gaussian_naive_cuda.png\n"
                      << "       Tiled CUDA 16x16: images/output/sample_gray_gaussian_tiled_16x16.png\n";
        }
    }

    std::cout << std::string(100, '-') << "\n";

    if (all_passed) {
        std::cout << "\n>>> ALL TEST CASES PASSED: 100.00% bit-exact match across all implementations & dimensions! <<<\n";
        return 0;
    } else {
        std::cerr << "\n>>> VERIFICATION FAILED: Differences detected! <<<\n";
        return 1;
    }
}

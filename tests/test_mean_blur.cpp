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

// Helper to generate synthetic grayscale test patterns
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

// Helper to resolve test file paths when executed from root or build/
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
    std::cout << " Mean Blur: CPU Reference vs. Naive CUDA Comparison Suite (Grayscale 3x3 & 5x5)          \n";
    std::cout << "========================================================================================\n";

    std::vector<TestCase> cases = {
        {"1. Single pixel (1x1)", 1, 1, false, ""},
        {"2. Small non-block multiple (17x19)", 17, 19, false, ""},
        {"3. Block multiple boundary (31x33)", 31, 33, false, ""},
        {"4. Exact block multiple 16x16 (256x256)", 256, 256, false, ""},
        {"5. Prime dimensions non-multiple (127x97)", 127, 97, false, ""},
        {"6. Large arbitrary non-multiple (831x517)", 831, 517, false, ""},
        {"7. Real image (images/input/sample.png)", 0, 0, true, "images/input/sample.png"}
    };

    std::vector<int> filter_sizes = {3, 5};
    bool all_passed = true;

    for (int filter_size : filter_sizes) {
        std::cout << "\n----------------------------------------------------------------------------------------\n";
        std::cout << " Evaluating Filter Size: " << filter_size << "x" << filter_size 
                  << " (Radius = " << (filter_size / 2) 
                  << ", Samples/Pixel = " << (filter_size * filter_size) << ")\n";
        std::cout << "----------------------------------------------------------------------------------------\n";

        std::cout << std::left << std::setw(42) << "Test Case"
                  << std::setw(14) << "Dimensions"
                  << std::setw(12) << "Match Rate"
                  << std::setw(12) << "Max Diff"
                  << std::setw(10) << "MAE"
                  << "Status\n";
        std::cout << std::string(96, '-') << "\n";

        for (const auto& tc : cases) {
            Image input_gray;
            if (tc.from_file) {
                try {
                    std::string actual_path = resolve_image_path(tc.filepath);
                    // Load real image directly as 1-channel grayscale
                    input_gray = load_image(actual_path, 1);
                } catch (const std::exception& e) {
                    std::cerr << "Could not load test image " << tc.filepath << ": " << e.what() << "\n";
                    all_passed = false;
                    continue;
                }
            } else {
                input_gray = create_grayscale_pattern(tc.width, tc.height);
            }


            // 1. CPU Reference Execution
            Image cpu_blurred = cpu::mean_blur(input_gray, filter_size);

            // 2. Naive CUDA Kernel Execution
            Image gpu_blurred = cuda::mean_blur_naive(input_gray, filter_size);

            // 3. Compute detailed difference statistics
            cpu::ImageDiffStats stats = cpu::compute_diff_stats(cpu_blurred, gpu_blurred);

            double match_pct = 100.0 * static_cast<double>(stats.identical_pixels) / stats.total_pixels;
            std::string dim_str = std::to_string(input_gray.width) + "x" + std::to_string(input_gray.height);

            // Strict bit-exact verification
            bool case_pass = (stats.mismatched_pixels == 0) && (stats.max_diff == 0.0);
            if (!case_pass) {
                all_passed = false;
            }

            std::cout << std::left << std::setw(42) << tc.name
                      << std::setw(14) << dim_str
                      << std::fixed << std::setprecision(2) << std::setw(11) << match_pct << "%"
                      << std::fixed << std::setprecision(1) << std::setw(12) << stats.max_diff
                      << std::fixed << std::setprecision(4) << std::setw(10) << stats.mean_absolute_error
                      << (case_pass ? "[PASSED]" : "[FAILED]") << "\n";

            // If 3x3, also verify all three shared-memory tiled configurations
            if (filter_size == 3) {
                std::vector<std::pair<std::string, cuda::TileConfig>> tiled_configs = {
                    {"   └─ Tiled 8x8", cuda::TileConfig::TILE_8x8},
                    {"   └─ Tiled 16x16", cuda::TileConfig::TILE_16x16},
                    {"   └─ Tiled 32x32", cuda::TileConfig::TILE_32x32}
                };

                for (const auto& [cfg_name, cfg] : tiled_configs) {
                    Image tiled_blurred = cuda::mean_blur_tiled_3x3(input_gray, cfg);
                    cpu::ImageDiffStats tiled_stats = cpu::compute_diff_stats(cpu_blurred, tiled_blurred);

                    double t_match_pct = 100.0 * static_cast<double>(tiled_stats.identical_pixels) / tiled_stats.total_pixels;
                    bool t_pass = (tiled_stats.mismatched_pixels == 0) && (tiled_stats.max_diff == 0.0);
                    if (!t_pass) {
                        all_passed = false;
                    }

                    std::cout << std::left << std::setw(42) << cfg_name
                              << std::setw(14) << dim_str
                              << std::fixed << std::setprecision(2) << std::setw(11) << t_match_pct << "%"
                              << std::fixed << std::setprecision(1) << std::setw(12) << tiled_stats.max_diff
                              << std::fixed << std::setprecision(4) << std::setw(10) << tiled_stats.mean_absolute_error
                              << (t_pass ? "[PASSED]" : "[FAILED]") << "\n";
                }
            }

            // If real image, save output artifacts for visual inspection
            if (tc.from_file) {
                fs::create_directories("images/output");
                std::string k_str = std::to_string(filter_size) + "x" + std::to_string(filter_size);
                std::string cpu_save_path = "images/output/sample_gray_blur_" + k_str + "_cpu.png";
                std::string gpu_save_path = "images/output/sample_gray_blur_" + k_str + "_cuda.png";

                save_image(cpu_save_path, cpu_blurred);
                save_image(gpu_save_path, gpu_blurred);

                if (filter_size == 3) {
                    Image t8 = cuda::mean_blur_tiled_3x3(input_gray, cuda::TileConfig::TILE_8x8);
                    Image t16 = cuda::mean_blur_tiled_3x3(input_gray, cuda::TileConfig::TILE_16x16);
                    Image t32 = cuda::mean_blur_tiled_3x3(input_gray, cuda::TileConfig::TILE_32x32);
                    save_image("images/output/sample_gray_blur_3x3_tiled_8x8.png", t8);
                    save_image("images/output/sample_gray_blur_3x3_tiled_16x16.png", t16);
                    save_image("images/output/sample_gray_blur_3x3_tiled_32x32.png", t32);
                }

                std::cout << "    -> Saved outputs to:\n"
                          << "       CPU Reference: " << cpu_save_path << "\n"
                          << "       CUDA Kernel  : " << gpu_save_path << "\n";
            }

        }
    }

    std::cout << "\n========================================================================================\n";
    if (all_passed) {
        std::cout << ">>> ALL TEST CASES PASSED: 100.00% bit-exact match across all dimensions & filters! <<<\n";
        return 0;
    } else {
        std::cerr << ">>> VERIFICATION FAILED: Differences detected between CPU and GPU results! <<<\n";
        return 1;
    }
}

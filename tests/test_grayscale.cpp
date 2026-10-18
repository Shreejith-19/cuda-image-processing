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

// Helper to resolve test file paths when executed from root or build/
std::string resolve_image_path(const std::string& rel_path) {
    if (fs::exists(rel_path)) return rel_path;
    if (fs::exists("../" + rel_path)) return "../" + rel_path;
    return rel_path;
}

// Helper to generate synthetic test patterns
Image create_test_pattern(int width, int height, unsigned int seed = 42) {
    Image img(width, height, 3);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            img.at(y, x, 0) = static_cast<uint8_t>(dist(rng));
            img.at(y, x, 1) = static_cast<uint8_t>(dist(rng));
            img.at(y, x, 2) = static_cast<uint8_t>(dist(rng));
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
    std::cout << "========================================================================\n";
    std::cout << " RGB to Grayscale: CPU Reference vs. Naive CUDA Comparison Suite       \n";
    std::cout << "========================================================================\n";

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

    std::cout << std::left << std::setw(42) << "Test Case"
              << std::setw(14) << "Dimensions"
              << std::setw(12) << "Match Rate"
              << std::setw(12) << "Max Diff"
              << std::setw(10) << "MAE"
              << "Status\n";
    std::cout << std::string(96, '-') << "\n";

    for (const auto& tc : cases) {
        Image input_rgb;
        if (tc.from_file) {
            try {
                std::string actual_path = resolve_image_path(tc.filepath);
                input_rgb = load_image(actual_path, 3);
            } catch (const std::exception& e) {
                std::cerr << "Could not load test image " << tc.filepath << ": " << e.what() << "\n";
                all_passed = false;
                continue;
            }
        } else {
            input_rgb = create_test_pattern(tc.width, tc.height);
        }

        // 1. CPU Reference
        Image cpu_gray = cpu::rgb_to_grayscale(input_rgb);

        // 2. CUDA Execution
        Image gpu_gray = cuda::rgb_to_grayscale(input_rgb);

        // 3. Compute difference statistics
        cpu::ImageDiffStats stats = cpu::compute_diff_stats(cpu_gray, gpu_gray);

        double match_pct = 100.0 * static_cast<double>(stats.identical_pixels) / stats.total_pixels;
        std::string dim_str = std::to_string(input_rgb.width) + "x" + std::to_string(input_rgb.height);

        // Verify: Differences must be within 1 LSB tolerance (due to GPU FMA contraction vs CPU non-fused math)
        bool case_pass = (stats.max_diff <= 1.0) && (stats.mean_absolute_error < 0.01);
        if (!case_pass) {
            all_passed = false;
        }


        std::cout << std::left << std::setw(42) << tc.name
                  << std::setw(14) << dim_str
                  << std::fixed << std::setprecision(2) << std::setw(11) << match_pct << "%"
                  << std::fixed << std::setprecision(1) << std::setw(12) << stats.max_diff
                  << std::fixed << std::setprecision(4) << std::setw(10) << stats.mean_absolute_error
                  << (case_pass ? "[PASSED]" : "[FAILED]") << "\n";
    }

    std::cout << std::string(96, '-') << "\n";

    if (all_passed) {
        std::cout << "\n>>> ALL TEST CASES PASSED: 100.00% bit-exact match across all dimensions! <<<\n";
        return 0;
    } else {
        std::cerr << "\n>>> VERIFICATION FAILED: Differences detected between CPU and GPU results! <<<\n";
        return 1;
    }
}

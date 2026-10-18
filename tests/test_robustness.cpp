#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <filesystem>
#include <cassert>
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
using namespace cuda_image_proc::cpu;
using namespace cuda_image_proc::cuda;

// Helper to resolve test file paths
std::string resolve_image_path(const std::string& rel_path) {
    if (fs::exists(rel_path)) return rel_path;
    if (fs::exists("../" + rel_path)) return "../" + rel_path;
    return rel_path;
}

// Synthetic pattern generator
Image create_pattern(int width, int height, int channels, unsigned int seed = 42) {
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

// ---------------------------------------------------------------------------
// TEST 1: Boundary & Explicit Border Handling (Impulse & Analytical Verification)
// ---------------------------------------------------------------------------
bool test_border_handling_explicit() {
    std::cout << "\n======================================================================\n";
    std::cout << " TEST 1: Explicit Border Handling & Clamp-to-Edge Analytical Checks   \n";
    std::cout << "======================================================================\n";
    bool pass = true;

    // For a 3x3 image with impulse at (0,0)=255, rest 0:
    // Under clamp-to-edge:
    // At (0,0): dx in [-1,0,1], dy in [-1,0,1].
    // dx <= 0 clamps to sx=0, dx = 1 gives sx=1.
    // dy <= 0 clamps to sy=0, dy = 1 gives sy=1.
    // The only non-zero pixel is at (sy=0, sx=0).
    // Combinations mapping to (0,0): dy in {-1, 0} and dx in {-1, 0} -> 2 x 2 = 4 samples.
    // Expected Sum = 4 * 255 = 1020.
    // Expected Average = 1020 / 9.0 = 113.33 -> roundf = 113.
    // At (1,1) (center), only 1 sample is at (0,0) with value 255, 8 samples are 0.
    // Expected Average = 255 / 9.0 = 28.33 -> roundf = 28.
    const uint8_t expected_corner_3x3 = static_cast<uint8_t>(std::round(4.0f * 255.0f / 9.0f)); // 113
    const uint8_t expected_center_3x3 = static_cast<uint8_t>(std::round(1.0f * 255.0f / 9.0f)); // 28

    Image impulse_img(3, 3, 1);
    std::fill(impulse_img.data.begin(), impulse_img.data.end(), 0);
    impulse_img.at(0, 0) = 255;

    // CPU Mean Blur 3x3
    Image cpu_impulse_blur = cpu::mean_blur(impulse_img, 3);
    uint8_t cpu_corner = cpu_impulse_blur.at(0, 0);
    uint8_t cpu_center = cpu_impulse_blur.at(1, 1);

    // CUDA Naive Mean Blur 3x3
    Image gpu_impulse_blur_naive = cuda::mean_blur_naive(impulse_img, 3);
    uint8_t gpu_naive_corner = gpu_impulse_blur_naive.at(0, 0);
    uint8_t gpu_naive_center = gpu_impulse_blur_naive.at(1, 1);

    // CUDA Tiled Mean Blur 3x3 (16x16)
    Image gpu_impulse_blur_tiled = cuda::mean_blur_tiled_3x3(impulse_img, TileConfig::TILE_16x16);
    uint8_t gpu_tiled_corner = gpu_impulse_blur_tiled.at(0, 0);
    uint8_t gpu_tiled_center = gpu_impulse_blur_tiled.at(1, 1);

    std::cout << "  [3x3 Corner Impulse Test]\n"
              << "    Analytical Expected  : Corner = " << static_cast<int>(expected_corner_3x3)
              << ", Center = " << static_cast<int>(expected_center_3x3) << "\n"
              << "    CPU Reference Result : Corner = " << static_cast<int>(cpu_corner)
              << ", Center = " << static_cast<int>(cpu_center) << "\n"
              << "    CUDA Naive Result    : Corner = " << static_cast<int>(gpu_naive_corner)
              << ", Center = " << static_cast<int>(gpu_naive_center) << "\n"
              << "    CUDA Tiled Result    : Corner = " << static_cast<int>(gpu_tiled_corner)
              << ", Center = " << static_cast<int>(gpu_tiled_center) << "\n";

    if (cpu_corner == expected_corner_3x3 && cpu_center == expected_center_3x3 &&
        gpu_naive_corner == expected_corner_3x3 && gpu_naive_center == expected_center_3x3 &&
        gpu_tiled_corner == expected_corner_3x3 && gpu_tiled_center == expected_center_3x3) {
        std::cout << "    -> [PASSED: Analytical clamp-to-edge bit-exact match on corners and center]\n";
    } else {
        std::cout << "    -> [FAILED: Clamp-to-edge values differ from expected!]\n";
        pass = false;
    }

    // 1x1 Image Corner Clamping (all 9 samples clamp to (0,0) -> expected 255)
    Image single_px(1, 1, 1);
    single_px.at(0, 0) = 255;
    Image cpu_1x1 = cpu::mean_blur(single_px, 3);
    Image gpu_1x1_naive = cuda::mean_blur_naive(single_px, 3);
    Image gpu_1x1_tiled = cuda::mean_blur_tiled_3x3(single_px, TileConfig::TILE_8x8);
    if (cpu_1x1.at(0, 0) == 255 && gpu_1x1_naive.at(0, 0) == 255 && gpu_1x1_tiled.at(0, 0) == 255) {
        std::cout << "  [1x1 Single Pixel Full Clamp Test]\n"
                  << "    -> [PASSED: All 9 neighborhood samples clamp to sole pixel (255)]\n";
    } else {
        std::cout << "  [1x1 Single Pixel Full Clamp Test]\n"
                  << "    -> [FAILED: Expected 255 for 1x1 clamped blur]\n";
        pass = false;
    }

    // Four corners impulse test on 5x5 image
    // Impulses at (0,0), (4,0), (0,4), (4,4)
    Image quad_corners(5, 5, 1);
    std::fill(quad_corners.data.begin(), quad_corners.data.end(), 0);
    quad_corners.at(0, 0) = 200;
    quad_corners.at(0, 4) = 200;
    quad_corners.at(4, 0) = 200;
    quad_corners.at(4, 4) = 200;

    Image cpu_quad = cpu::gaussian_blur_3x3(quad_corners);
    Image gpu_quad_naive = cuda::gaussian_blur_naive_3x3(quad_corners);
    Image gpu_quad_tiled = cuda::gaussian_blur_tiled_3x3(quad_corners, TileConfig::TILE_8x8);

    ImageDiffStats diff_quad_naive = cpu::compute_diff_stats(cpu_quad, gpu_quad_naive);
    ImageDiffStats diff_quad_tiled = cpu::compute_diff_stats(cpu_quad, gpu_quad_tiled);

    if (diff_quad_naive.identical_pixels == diff_quad_naive.total_pixels &&
        diff_quad_tiled.identical_pixels == diff_quad_tiled.total_pixels) {
        std::cout << "  [Four-Corners Impulse 5x5 Gaussian Test]\n"
                  << "    -> [PASSED: Exact match between CPU, Naive GPU, and Tiled GPU across all corners]\n";
    } else {
        std::cout << "  [Four-Corners Impulse 5x5 Gaussian Test]\n"
                  << "    -> [FAILED: Differences detected in corner responses]\n";
        pass = false;
    }

    return pass;
}

// ---------------------------------------------------------------------------
// TEST 2: Multi-Size & Irregular Dimensions Across All Operations
// ---------------------------------------------------------------------------
struct SizeTestCase {
    std::string label;
    int width;
    int height;
};

bool test_multi_size_and_irregular_dimensions() {
    std::cout << "\n======================================================================\n";
    std::cout << " TEST 2: Multi-Size, Tiny Images & Irregular Dimension Robustness     \n";
    std::cout << "======================================================================\n";
    bool pass = true;

    std::vector<SizeTestCase> test_sizes = {
        {"1x1 Single Pixel", 1, 1},
        {"2x2 Minimal Multi-Pixel", 2, 2},
        {"3x3 Filter Kernel Match", 3, 3},
        {"4x7 Asymmetric Tiny", 4, 7},
        {"17x19 Small Primes (Not Divisible by 8/16/32)", 17, 19},
        {"31x33 Boundary Crossings (32x32 Boundary)", 31, 33},
        {"64x64 Power of 2", 64, 64},
        {"127x97 Odd Prime Dimensions", 127, 97},
        {"256x256 Block Multiple", 256, 256},
        {"831x517 Large Irregular Non-Multiple", 831, 517}
    };

    std::cout << std::left << std::setw(40) << "Size Case"
              << std::setw(12) << "Dims"
              << std::setw(14) << "Grayscale"
              << std::setw(14) << "Mean Blur"
              << std::setw(14) << "Gaussian"
              << std::setw(14) << "Sobel"
              << "Status\n";
    std::cout << std::string(105, '-') << "\n";

    for (const auto& tc : test_sizes) {
        int w = tc.width;
        int h = tc.height;
        std::string dim_str = std::to_string(w) + "x" + std::to_string(h);

        // 1. RGB to Grayscale
        Image rgb_in = create_pattern(w, h, 3, 100);
        Image cpu_gray = cpu::rgb_to_grayscale(rgb_in);
        Image gpu_gray = cuda::rgb_to_grayscale(rgb_in);
        ImageDiffStats diff_gray = cpu::compute_diff_stats(cpu_gray, gpu_gray);
        // Documented tolerance: max_diff <= 1 due to FMA contraction vs non-fused float
        bool gray_ok = (diff_gray.max_diff <= 1.0) && (diff_gray.mean_absolute_error < 0.01);

        // 2. Mean Blur (Grayscale 1-channel, 3x3)
        Image gray_in = cpu_gray;
        Image cpu_mb = cpu::mean_blur(gray_in, 3);
        Image gpu_mb_naive = cuda::mean_blur_naive(gray_in, 3);
        Image gpu_mb_tiled8 = cuda::mean_blur_tiled_3x3(gray_in, TileConfig::TILE_8x8);
        Image gpu_mb_tiled16 = cuda::mean_blur_tiled_3x3(gray_in, TileConfig::TILE_16x16);
        Image gpu_mb_tiled32 = cuda::mean_blur_tiled_3x3(gray_in, TileConfig::TILE_32x32);

        ImageDiffStats diff_mb_naive = cpu::compute_diff_stats(cpu_mb, gpu_mb_naive);
        ImageDiffStats diff_mb_t8 = cpu::compute_diff_stats(cpu_mb, gpu_mb_tiled8);
        ImageDiffStats diff_mb_t16 = cpu::compute_diff_stats(cpu_mb, gpu_mb_tiled16);
        ImageDiffStats diff_mb_t32 = cpu::compute_diff_stats(cpu_mb, gpu_mb_tiled32);
        bool mb_ok = (diff_mb_naive.identical_pixels == diff_mb_naive.total_pixels) &&
                     (diff_mb_t8.identical_pixels == diff_mb_t8.total_pixels) &&
                     (diff_mb_t16.identical_pixels == diff_mb_t16.total_pixels) &&
                     (diff_mb_t32.identical_pixels == diff_mb_t32.total_pixels);

        // 3. Gaussian Blur (Grayscale 1-channel, 3x3)
        Image cpu_gb = cpu::gaussian_blur_3x3(gray_in);
        Image gpu_gb_naive = cuda::gaussian_blur_naive_3x3(gray_in);
        Image gpu_gb_t8 = cuda::gaussian_blur_tiled_3x3(gray_in, TileConfig::TILE_8x8);
        Image gpu_gb_t16 = cuda::gaussian_blur_tiled_3x3(gray_in, TileConfig::TILE_16x16);
        Image gpu_gb_t32 = cuda::gaussian_blur_tiled_3x3(gray_in, TileConfig::TILE_32x32);

        ImageDiffStats diff_gb_naive = cpu::compute_diff_stats(cpu_gb, gpu_gb_naive);
        ImageDiffStats diff_gb_t8 = cpu::compute_diff_stats(cpu_gb, gpu_gb_t8);
        ImageDiffStats diff_gb_t16 = cpu::compute_diff_stats(cpu_gb, gpu_gb_t16);
        ImageDiffStats diff_gb_t32 = cpu::compute_diff_stats(cpu_gb, gpu_gb_t32);
        bool gb_ok = (diff_gb_naive.identical_pixels == diff_gb_naive.total_pixels) &&
                     (diff_gb_t8.identical_pixels == diff_gb_t8.total_pixels) &&
                     (diff_gb_t16.identical_pixels == diff_gb_t16.total_pixels) &&
                     (diff_gb_t32.identical_pixels == diff_gb_t32.total_pixels);

        // 4. Sobel Edge Detection (Grayscale 1-channel)
        Image cpu_sobel = cpu::sobel_edge_detect(gray_in);
        Image gpu_sobel_naive = cuda::sobel_edge_detect_naive(gray_in);
        Image gpu_sobel_t8 = cuda::sobel_edge_detect_tiled(gray_in, TileConfig::TILE_8x8);
        Image gpu_sobel_t16 = cuda::sobel_edge_detect_tiled(gray_in, TileConfig::TILE_16x16);
        Image gpu_sobel_t32 = cuda::sobel_edge_detect_tiled(gray_in, TileConfig::TILE_32x32);

        ImageDiffStats diff_sobel_naive = cpu::compute_diff_stats(cpu_sobel, gpu_sobel_naive);
        ImageDiffStats diff_sobel_t8 = cpu::compute_diff_stats(cpu_sobel, gpu_sobel_t8);
        ImageDiffStats diff_sobel_t16 = cpu::compute_diff_stats(cpu_sobel, gpu_sobel_t16);
        ImageDiffStats diff_sobel_t32 = cpu::compute_diff_stats(cpu_sobel, gpu_sobel_t32);
        bool sobel_ok = (diff_sobel_naive.identical_pixels == diff_sobel_naive.total_pixels) &&
                        (diff_sobel_t8.identical_pixels == diff_sobel_t8.total_pixels) &&
                        (diff_sobel_t16.identical_pixels == diff_sobel_t16.total_pixels) &&
                        (diff_sobel_t32.identical_pixels == diff_sobel_t32.total_pixels);

        bool case_pass = gray_ok && mb_ok && gb_ok && sobel_ok;
        if (!case_pass) pass = false;

        std::cout << std::left << std::setw(40) << tc.label
                  << std::setw(12) << dim_str
                  << std::setw(14) << (gray_ok ? "100% (<=1 LSB)" : "FAIL")
                  << std::setw(14) << (mb_ok ? "100% Exact" : "FAIL")
                  << std::setw(14) << (gb_ok ? "100% Exact" : "FAIL")
                  << std::setw(14) << (sobel_ok ? "100% Exact" : "FAIL")
                  << (case_pass ? "[PASSED]" : "[FAILED]") << "\n";
    }

    return pass;
}

// ---------------------------------------------------------------------------
// TEST 3: Multi-Channel & Filter Configuration Validation
// ---------------------------------------------------------------------------
bool test_filter_configurations_and_channels() {
    std::cout << "\n======================================================================\n";
    std::cout << " TEST 3: Multi-Channel & Filter Configurations Validation             \n";
    std::cout << "======================================================================\n";
    bool pass = true;

    // 1. Mean blur with multiple filter sizes (3x3, 5x5, 7x7) on 3-channel RGB image
    Image rgb_img = create_pattern(64, 48, 3, 202);
    for (int k : {3, 5, 7}) {
        Image cpu_mb = cpu::mean_blur(rgb_img, k);
        Image gpu_mb = cuda::mean_blur_naive(rgb_img, k);
        ImageDiffStats stats = cpu::compute_diff_stats(cpu_mb, gpu_mb);
        bool ok = (stats.identical_pixels == stats.total_pixels);
        std::cout << "  [Mean Blur " << k << "x" << k << " on 3-Channel RGB (64x48)]: "
                  << (ok ? "100% Bit-Exact Match [PASSED]" : "MISMATCH [FAILED]") << "\n";
        if (!ok) pass = false;
    }

    // 2. Mean blur with 4-channel RGBA image (64x48)
    Image rgba_img = create_pattern(64, 48, 4, 303);
    {
        Image cpu_rgba_blur = cpu::mean_blur(rgba_img, 3);
        Image gpu_rgba_blur = cuda::mean_blur_naive(rgba_img, 3);
        ImageDiffStats stats = cpu::compute_diff_stats(cpu_rgba_blur, gpu_rgba_blur);
        bool ok = (stats.identical_pixels == stats.total_pixels);
        std::cout << "  [Mean Blur 3x3 on 4-Channel RGBA (64x48)]: "
                  << (ok ? "100% Bit-Exact Match [PASSED]" : "MISMATCH [FAILED]") << "\n";
        if (!ok) pass = false;
    }

    // 3. Gaussian blur on 3-channel RGB image (64x48)
    {
        Image cpu_gb = cpu::gaussian_blur_3x3(rgb_img);
        Image gpu_gb = cuda::gaussian_blur_naive_3x3(rgb_img);
        ImageDiffStats stats = cpu::compute_diff_stats(cpu_gb, gpu_gb);
        bool ok = (stats.identical_pixels == stats.total_pixels);
        std::cout << "  [Gaussian Blur 3x3 on 3-Channel RGB (64x48)]: "
                  << (ok ? "100% Bit-Exact Match [PASSED]" : "MISMATCH [FAILED]") << "\n";
        if (!ok) pass = false;
    }

    // 4. Gaussian blur on 4-channel RGBA image (64x48)
    {
        Image cpu_rgba_gb = cpu::gaussian_blur_3x3(rgba_img);
        Image gpu_rgba_gb = cuda::gaussian_blur_naive_3x3(rgba_img);
        ImageDiffStats stats = cpu::compute_diff_stats(cpu_rgba_gb, gpu_rgba_gb);
        bool ok = (stats.identical_pixels == stats.total_pixels);
        std::cout << "  [Gaussian Blur 3x3 on 4-Channel RGBA (64x48)]: "
                  << (ok ? "100% Bit-Exact Match [PASSED]" : "MISMATCH [FAILED]") << "\n";
        if (!ok) pass = false;
    }

    // 5. Padded stride test: row_stride > width * channels
    {
        int w = 53;
        int h = 37;
        int c = 1;
        size_t padded_stride = static_cast<size_t>(w * c) + 64; // add 64 bytes padding
        Image padded_src(w, h, c, padded_stride);
        std::mt19937 rng(404);
        std::uniform_int_distribution<int> dist(0, 255);
        for (int y = 0; y < h; ++y) {
            uint8_t* row = padded_src.row_ptr(y);
            for (int x = 0; x < w; ++x) {
                row[x] = static_cast<uint8_t>(dist(rng));
            }
        }

        Image cpu_pad_blur = cpu::mean_blur(padded_src, 3);
        Image gpu_pad_blur = cuda::mean_blur_naive(padded_src, 3);
        ImageDiffStats stats = cpu::compute_diff_stats(cpu_pad_blur, gpu_pad_blur);
        bool ok = (stats.identical_pixels == stats.total_pixels);
        std::cout << "  [Padded Row Stride Test (" << w << "x" << h << ", stride " << padded_stride << ")]: "
                  << (ok ? "100% Bit-Exact Match [PASSED]" : "MISMATCH [FAILED]") << "\n";
        if (!ok) pass = false;
    }

    return pass;
}

// ---------------------------------------------------------------------------
// TEST 4: Invalid Arguments, Negative/Empty Dimensions & Channel Mismatches
// ---------------------------------------------------------------------------
bool test_invalid_arguments_and_error_handling() {
    std::cout << "\n======================================================================\n";
    std::cout << " TEST 4: Invalid Arguments, Negative/Empty Dims & Channel Exceptions  \n";
    std::cout << "======================================================================\n";
    bool pass = true;

    auto expect_invalid_arg = [&](const std::string& name, auto func) {
        bool caught = false;
        try {
            func();
        } catch (const std::invalid_argument& e) {
            caught = true;
            std::cout << "  [Expected Exception: " << name << "] -> Caught: " << e.what() << " [PASSED]\n";
        } catch (const std::exception& e) {
            std::cout << "  [Wrong Exception Type: " << name << "] -> Caught: " << e.what() << " [FAILED]\n";
        }
        if (!caught) {
            std::cout << "  [Missed Exception: " << name << "] -> No exception thrown! [FAILED]\n";
            pass = false;
        }
    };

    auto expect_out_of_range = [&](const std::string& name, auto func) {
        bool caught = false;
        try {
            func();
        } catch (const std::out_of_range& e) {
            caught = true;
            std::cout << "  [Expected Out-Of-Range: " << name << "] -> Caught: " << e.what() << " [PASSED]\n";
        } catch (const std::exception& e) {
            std::cout << "  [Wrong Exception Type: " << name << "] -> Caught: " << e.what() << " [FAILED]\n";
        }
        if (!caught) {
            std::cout << "  [Missed Exception: " << name << "] -> No exception thrown! [FAILED]\n";
            pass = false;
        }
    };

    // 1. Negative dimensions in HostImage
    expect_invalid_arg("HostImage(-10, 10, 3)", []() {
        Image img(-10, 10, 3);
    });

    // 2. Negative dimensions in DeviceImage
    expect_invalid_arg("DeviceImage(10, -5, 1)", []() {
        DeviceImage img(10, -5, 1);
    });

    // 3. Zero dimensions in DeviceImage
    expect_invalid_arg("DeviceImage(0, 10, 1)", []() {
        DeviceImage img(0, 10, 1);
    });

    // 4. Empty HostImage passed to to_device()
    expect_invalid_arg("to_device(empty HostImage)", []() {
        Image empty_img;
        to_device(empty_img);
    });

    // 5. Empty DeviceImage passed to to_host()
    expect_invalid_arg("to_host(empty DeviceImage)", []() {
        DeviceImage empty_dev;
        to_host(empty_dev);
    });

    // 6. rgb_to_grayscale on 1-channel image (Host & GPU)
    expect_invalid_arg("cpu::rgb_to_grayscale(1-channel image)", []() {
        Image gray_img(10, 10, 1);
        cpu::rgb_to_grayscale(gray_img);
    });
    expect_invalid_arg("cuda::rgb_to_grayscale(1-channel DeviceImage)", []() {
        DeviceImage gray_dev(10, 10, 1);
        cuda::rgb_to_grayscale(gray_dev);
    });

    // 7. sobel_edge_detect on 3-channel RGB image (Host & GPU)
    expect_invalid_arg("cpu::sobel_edge_detect(3-channel image)", []() {
        Image rgb_img(10, 10, 3);
        cpu::sobel_edge_detect(rgb_img);
    });
    expect_invalid_arg("cuda::sobel_edge_detect_naive(3-channel DeviceImage)", []() {
        DeviceImage rgb_dev(10, 10, 3);
        cuda::sobel_edge_detect_naive(rgb_dev);
    });
    expect_invalid_arg("cuda::sobel_edge_detect_tiled(3-channel DeviceImage)", []() {
        DeviceImage rgb_dev(10, 10, 3);
        cuda::sobel_edge_detect_tiled(rgb_dev, TileConfig::TILE_16x16);
    });

    // 8. Mean blur with even filter size (Host & GPU)
    expect_invalid_arg("cpu::mean_blur(img, 4)", []() {
        Image img(10, 10, 1);
        cpu::mean_blur(img, 4);
    });
    expect_invalid_arg("cuda::mean_blur_naive(img, 0)", []() {
        Image img(10, 10, 1);
        cuda::mean_blur_naive(img, 0);
    });
    expect_invalid_arg("cuda::mean_blur_naive(img, -3)", []() {
        Image img(10, 10, 1);
        cuda::mean_blur_naive(img, -3);
    });

    // 9. Mean blur tiled on 3-channel image
    expect_invalid_arg("cuda::mean_blur_tiled_3x3(3-channel DeviceImage)", []() {
        DeviceImage rgb_dev(10, 10, 3);
        cuda::mean_blur_tiled_3x3(rgb_dev, TileConfig::TILE_16x16);
    });

    // 10. Gaussian blur tiled on 3-channel image
    expect_invalid_arg("cuda::gaussian_blur_tiled_3x3(3-channel DeviceImage)", []() {
        DeviceImage rgb_dev(10, 10, 3);
        cuda::gaussian_blur_tiled_3x3(rgb_dev, TileConfig::TILE_16x16);
    });

    // 11. Out-of-bounds indexing in Image::row_ptr() and Image::at()
    expect_out_of_range("Image::row_ptr(out of bounds)", []() {
        Image img(10, 10, 1);
        img.row_ptr(15);
    });
    expect_out_of_range("Image::at(out of bounds)", []() {
        Image img(10, 10, 1);
        img.at(5, 20);
    });

    return pass;
}

// ---------------------------------------------------------------------------
// TEST 5: GPU Resource Management & Memory Leak Audit
// ---------------------------------------------------------------------------
bool test_gpu_resource_leak_audit() {
    std::cout << "\n======================================================================\n";
    std::cout << " TEST 5: GPU Allocation, Transfer, Kernel Errors & Memory Leak Audit  \n";
    std::cout << "======================================================================\n";
    bool pass = true;

    // Synchronize to establish a clean memory baseline
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t free_before = 0;
    size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));

    std::cout << "  Initial GPU Free Memory: " << (free_before / (1024 * 1024)) << " MB / "
              << (total / (1024 * 1024)) << " MB (" << free_before << " bytes)\n";

    // Run 50 iterations of allocations, transfers, and kernel launches across all ops
    const int iterations = 50;
    const int w = 256;
    const int h = 256;

    for (int i = 0; i < iterations; ++i) {
        Image rgb_in = create_pattern(w, h, 3, 500 + i);

        // Scope to force DeviceImage destruction and deallocation via RAII
        {
            CudaStream stream;
            DeviceImage d_rgb = to_device(rgb_in, stream);

            // 1. Grayscale
            DeviceImage d_gray = cuda::rgb_to_grayscale(d_rgb, stream);

            // 2. Mean Blur Naive
            DeviceImage d_mb_naive = cuda::mean_blur_naive(d_gray, 3, stream);

            // 3. Mean Blur Tiled (8x8, 16x16, 32x32)
            DeviceImage d_mb_t8 = cuda::mean_blur_tiled_3x3(d_gray, TileConfig::TILE_8x8, stream);
            DeviceImage d_mb_t16 = cuda::mean_blur_tiled_3x3(d_gray, TileConfig::TILE_16x16, stream);
            DeviceImage d_mb_t32 = cuda::mean_blur_tiled_3x3(d_gray, TileConfig::TILE_32x32, stream);

            // 4. Gaussian Blur Naive & Tiled
            DeviceImage d_gb_naive = cuda::gaussian_blur_naive_3x3(d_gray, stream);
            DeviceImage d_gb_t16 = cuda::gaussian_blur_tiled_3x3(d_gray, TileConfig::TILE_16x16, stream);

            // 5. Sobel Naive & Tiled
            DeviceImage d_sobel_naive = cuda::sobel_edge_detect_naive(d_gray, stream);
            DeviceImage d_sobel_t16 = cuda::sobel_edge_detect_tiled(d_gray, TileConfig::TILE_16x16, stream);

            // Host read-back verification
            Image h_result = to_host(d_sobel_t16, stream);
            stream.synchronize();
            (void)h_result;
        }

        // Check for any asynchronous kernel launch or execution errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "  CUDA Error detected in iteration " << i << ": " << cudaGetErrorString(err) << "\n";
            pass = false;
            break;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));

    std::cout << "  GPU Free Memory after " << iterations << " iterations: "
              << (free_after / (1024 * 1024)) << " MB (" << free_after << " bytes)\n";

    ptrdiff_t leaked_bytes = static_cast<ptrdiff_t>(free_before) - static_cast<ptrdiff_t>(free_after);
    std::cout << "  Memory Differential: " << leaked_bytes << " bytes\n";

    if (leaked_bytes == 0) {
        std::cout << "  -> [PASSED: 0 bytes leaked across " << iterations << " complete execution cycles]\n";
    } else {
        std::cout << "  -> [FAILED: Memory leak detected! " << leaked_bytes << " bytes unaccounted for]\n";
        pass = false;
    }

    return pass;
}

// ---------------------------------------------------------------------------
// Main Robustness Test Driver
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=================================================================================\n";
    std::cout << " CUDA Image Processing: Automated Correctness & Robustness Test Suite             \n";
    std::cout << "=================================================================================\n";

    bool test1 = test_border_handling_explicit();
    bool test2 = test_multi_size_and_irregular_dimensions();
    bool test3 = test_filter_configurations_and_channels();
    bool test4 = test_invalid_arguments_and_error_handling();
    bool test5 = test_gpu_resource_leak_audit();

    bool overall = test1 && test2 && test3 && test4 && test5;

    std::cout << "\n=================================================================================\n";
    std::cout << " SUMMARY OF ROBUSTNESS TEST SUITE RESULTS:\n";
    std::cout << "   1. Border Handling & Clamp-to-Edge Analytical Checks   : " << (test1 ? "PASSED" : "FAILED") << "\n";
    std::cout << "   2. Multi-Size, Tiny Images & Irregular Dimensions      : " << (test2 ? "PASSED" : "FAILED") << "\n";
    std::cout << "   3. Multi-Channel & Filter Configurations Validation    : " << (test3 ? "PASSED" : "FAILED") << "\n";
    std::cout << "   4. Invalid Arguments, Negative Dims & Exception Guards : " << (test4 ? "PASSED" : "FAILED") << "\n";
    std::cout << "   5. GPU Resource Management & Memory Leak Audit         : " << (test5 ? "PASSED" : "FAILED") << "\n";
    std::cout << "=================================================================================\n";

    if (overall) {
        std::cout << ">>> OVERALL SUITE STATUS: ALL TESTS PASSED SUCCESSFULLY! <<<\n";
        return 0;
    } else {
        std::cerr << ">>> OVERALL SUITE STATUS: FAILURES DETECTED! <<<\n";
        return 1;
    }
}

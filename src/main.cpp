#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include "cuda_image_proc/image.hpp"
#include "cuda_image_proc/image_io.hpp"
#include "cuda_image_proc/cpu_ops.hpp"
#include "cuda_image_proc/cuda_utils.hpp"
#include "cuda_image_proc/device_image.cuh"
#include "cuda_image_proc/memory_transfer.cuh"
#include "cuda_image_proc/cuda_ops.cuh"

namespace fs = std::filesystem;
using namespace cuda_image_proc;

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --mode <cpu|cuda|both>                     Execution target (default: both)\n"
              << "  --op <grayscale|blur|gaussian|sobel|both>  Operation to execute (default: both)\n"
              << "  --filter-size <3|5|...>                    Odd filter dimension for mean blur (default: 3)\n"
              << "  --input <path>                             Path to input image (default: images/input/sample.png)\n"
              << "  --output <path>                            Path to output image (default: images/output/result.png)\n"
              << "  --help                                     Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=================================================================\n";
    std::cout << " CUDA Image Processing - Filters & Edge Detection Accelerator    \n";
    std::cout << "=================================================================\n";

    std::string mode = "both";
    std::string op = "both";
    int filter_size = 3;
    bool filter_size_specified = false;
    std::string input_path = "images/input/sample.png";
    std::string output_path = "images/output/result.png";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--op" && i + 1 < argc) {
            op = argv[++i];
        } else if (arg == "--filter-size" && i + 1 < argc) {
            filter_size = std::stoi(argv[++i]);
            filter_size_specified = true;
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            // Positional arguments for backward compatibility
            if (i == 1) input_path = arg;
            else if (i == 2) output_path = arg;
            else if (i == 3) mode = arg;
        }
    }

    if (mode != "cpu" && mode != "cuda" && mode != "both") {
        std::cerr << "Invalid mode '" << mode << "'. Expected 'cpu', 'cuda', or 'both'.\n";
        print_usage(argv[0]);
        return 1;
    }

    if (op != "grayscale" && op != "blur" && op != "gaussian" && op != "sobel" && op != "both") {
        std::cerr << "Invalid op '" << op << "'. Expected 'grayscale', 'blur', 'gaussian', 'sobel', or 'both'.\n";
        print_usage(argv[0]);
        return 1;
    }

    // Interactive prompt for filter size when not specified on command-line
    if (!filter_size_specified && (op == "blur" || op == "both")) {
        std::cout << "\n[Filter Selection for Mean Blur]\n"
                  << "Choose filter size (e.g. 3 for 3x3, 5 for 5x5, 7 for 7x7) [default: 3]: ";
        std::cout.flush();
        std::string line;
        if (std::getline(std::cin, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            size_t last = line.find_last_not_of(" \t\r\n");
            if (first != std::string::npos && last != std::string::npos) {
                try {
                    int val = std::stoi(line.substr(first, last - first + 1));
                    if (val > 0 && val % 2 != 0) {
                        filter_size = val;
                        std::cout << "--> Selected Mean Blur filter size: " << filter_size << "x" << filter_size << "\n\n";
                    } else {
                        std::cout << "--> Filter size must be a positive odd integer. Using default: 3x3\n\n";
                    }
                } catch (...) {
                    std::cout << "--> Invalid input. Using default: 3x3\n\n";
                }
            } else {
                std::cout << "--> Using default: 3x3\n\n";
            }
        }
    } else if (!filter_size_specified && op == "gaussian") {
        std::cout << "\n[Filter Selection for Gaussian Blur]\n"
                  << "Choose filter size (e.g. 3 for 3x3, 5, 7, 9, 15, 21) [default: 3]: ";
        std::cout.flush();
        std::string line;
        if (std::getline(std::cin, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            size_t last = line.find_last_not_of(" \t\r\n");
            if (first != std::string::npos && last != std::string::npos) {
                try {
                    int val = std::stoi(line.substr(first, last - first + 1));
                    if (val > 0 && val % 2 != 0 && val <= 31) {
                        filter_size = val;
                        std::cout << "--> Selected Gaussian Blur filter size: " << filter_size << "x" << filter_size << "\n\n";
                    } else {
                        std::cout << "--> Gaussian blur supports positive odd integers up to 31. Using default: 3x3\n\n";
                    }
                } catch (...) {
                    std::cout << "--> Invalid input. Using default: 3x3\n\n";
                }
            } else {
                std::cout << "--> Using default: 3x3\n\n";
            }
        }
    }

    if (filter_size <= 0 || filter_size % 2 == 0) {
        std::cerr << "Invalid filter-size " << filter_size << ". Must be a positive odd integer (e.g. 3, 5).\n";
        return 1;
    }

    std::cout << "Execution Mode : [" << mode << "]\n";
    std::cout << "Operation      : [" << op << "]\n";
    if (op == "blur" || op == "both" || op == "gaussian") {
        std::cout << "Filter Size    : " << filter_size << "x" << filter_size 
                  << " (Radius: " << (filter_size / 2) << ")\n";
    }
    std::cout << "Input Image    : " << input_path << "\n";
    std::cout << "Output Target  : " << output_path << "\n\n";

    try {
        // Query GPU info if CUDA or both selected
        if (mode == "cuda" || mode == "both") {
            cuda::DeviceInfo dev_info = cuda::get_device_info(0);
            std::cout << "[CUDA Device] " << dev_info.name 
                      << " (Compute Capability " << dev_info.major << "." << dev_info.minor << ")\n";
            std::cout << "              VRAM: " << (dev_info.total_memory_bytes / (1024 * 1024)) << " MB, "
                      << "SMs: " << dev_info.sm_count << "\n\n";
        }

        // 1. Load the Image
        std::cout << "[1] Loading input image...\n";
        Image input_img;
        if (op == "blur" || op == "gaussian") {
            // Load as 1-channel grayscale directly if only blurring
            input_img = load_image(input_path, 1);
        } else {
            // Load as RGB for grayscale conversion or pipeline
            input_img = load_image(input_path, 3);
        }
        std::cout << "    Dimensions  : " << input_img.width << " x " << input_img.height << "\n";
        std::cout << "    Channels    : " << input_img.channels << "\n";
        std::cout << "    Row Stride  : " << input_img.stride_bytes << " bytes\n";
        std::cout << "    Total Pixels: " << (input_img.width * input_img.height) << "\n\n";

        fs::path p(output_path);
        std::string stem = p.stem().string();
        std::string ext = p.extension().string();
        fs::path parent = p.parent_path();
        if (parent.empty()) parent = ".";
        fs::create_directories(parent);

        // Stage 1: Grayscale Conversion
        Image cpu_gray, gpu_gray;
        if (op == "grayscale" || op == "both") {
            std::cout << "[2] Running RGB-to-Grayscale Conversion...\n";
            if (mode == "cpu" || mode == "both") {
                auto t0 = std::chrono::high_resolution_clock::now();
                cpu_gray = cpu::rgb_to_grayscale(input_img);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "    CPU Grayscale Time : " << ms << " ms\n";
            }

            if (mode == "cuda" || mode == "both") {
                CudaStream stream;
                DeviceImage d_rgb = to_device(input_img, stream);

                cudaEvent_t start_event, stop_event;
                CUDA_CHECK(cudaEventCreate(&start_event));
                CUDA_CHECK(cudaEventCreate(&stop_event));

                CUDA_CHECK(cudaEventRecord(start_event, stream));
                DeviceImage d_gray = cuda::rgb_to_grayscale(d_rgb, stream);
                CUDA_CHECK(cudaEventRecord(stop_event, stream));
                CUDA_CHECK(cudaEventSynchronize(stop_event));

                float gpu_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_event, stop_event));
                CUDA_CHECK(cudaEventDestroy(start_event));
                CUDA_CHECK(cudaEventDestroy(stop_event));

                gpu_gray = to_host(d_gray, stream);
                std::cout << "    CUDA Grayscale Time: " << gpu_ms << " ms\n";
            }

            if (mode == "both") {
                cpu::ImageDiffStats stats = cpu::compute_diff_stats(cpu_gray, gpu_gray);
                std::cout << "    Grayscale Match Rate: " 
                          << (100.0 * stats.identical_pixels / stats.total_pixels) << "%"
                          << " (Max Diff: " << stats.max_diff << ")\n";
                std::string gray_path = (parent / (stem + "_gray" + ext)).string();
                save_image(gray_path, gpu_gray);
                std::cout << "    Saved Grayscale Image to: " << gray_path << "\n\n";
            }
        }

        // Stage 2: Mean Blur on Grayscale Image
        if (op == "blur" || op == "both") {
            Image blur_input = (op == "blur") ? input_img : ((mode == "cpu") ? cpu_gray : gpu_gray);
            Image cpu_blur, gpu_blur;

            std::cout << "[3] Running Mean Blur (" << filter_size << "x" << filter_size 
                      << " Box Filter, Clamp-to-Edge)...\n";

            if (mode == "cpu" || mode == "both") {
                auto t0 = std::chrono::high_resolution_clock::now();
                cpu_blur = cpu::mean_blur(blur_input, filter_size);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "    CPU Mean Blur Time : " << ms << " ms\n";
            }

            if (mode == "cuda" || mode == "both") {
                CudaStream stream;
                DeviceImage d_src = to_device(blur_input, stream);

                cudaEvent_t start_event, stop_event;
                CUDA_CHECK(cudaEventCreate(&start_event));
                CUDA_CHECK(cudaEventCreate(&stop_event));

                CUDA_CHECK(cudaEventRecord(start_event, stream));
                DeviceImage d_blur = cuda::mean_blur_naive(d_src, filter_size, stream);
                CUDA_CHECK(cudaEventRecord(stop_event, stream));
                CUDA_CHECK(cudaEventSynchronize(stop_event));

                float gpu_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_event, stop_event));
                CUDA_CHECK(cudaEventDestroy(start_event));
                CUDA_CHECK(cudaEventDestroy(stop_event));

                gpu_blur = to_host(d_blur, stream);
                std::cout << "    CUDA Mean Blur Time: " << gpu_ms << " ms\n";
            }

            if (mode == "both") {
                cpu::ImageDiffStats stats = cpu::compute_diff_stats(cpu_blur, gpu_blur);
                std::cout << "    Mean Blur Match Rate: " 
                          << (100.0 * stats.identical_pixels / stats.total_pixels) << "%\n"
                          << "    Identical Pixels    : " << stats.identical_pixels << " / " << stats.total_pixels << "\n"
                          << "    Mismatched Pixels   : " << stats.mismatched_pixels << "\n"
                          << "    Max Absolute Diff   : " << stats.max_diff << "\n"
                          << "    Mean Abs Error (MAE): " << stats.mean_absolute_error << "\n";

                std::string k_tag = "_" + std::to_string(filter_size) + "x" + std::to_string(filter_size);
                std::string cpu_save = (parent / (stem + "_blur" + k_tag + "_cpu" + ext)).string();
                std::string gpu_save = (parent / (stem + "_blur" + k_tag + "_cuda" + ext)).string();

                save_image(cpu_save, cpu_blur);
                save_image(gpu_save, gpu_blur);
                save_image(output_path, gpu_blur);

                std::cout << "\n    Saved Blurred Images:\n"
                          << "      - Primary Output : " << output_path << "\n"
                          << "      - CPU Reference  : " << cpu_save << "\n"
                          << "      - CUDA Generated : " << gpu_save << "\n";

                if (stats.mismatched_pixels == 0 && stats.max_diff == 0.0) {
                    std::cout << "\n>>> VALIDATION SUCCESS: CPU and CUDA blur results are 100% bit-exact! <<<\n";
                } else {
                    std::cerr << "\n>>> VALIDATION WARNING: Differences detected between CPU and CUDA blur! <<<\n";
                    return 1;
                }
            } else if (mode == "cpu") {
                save_image(output_path, cpu_blur);
                std::cout << "    Saved CPU blur output to: " << output_path << "\n";
            } else if (mode == "cuda") {
                save_image(output_path, gpu_blur);
                std::cout << "    Saved CUDA blur output to: " << output_path << "\n";
            }
        }

        // Stage 3: Gaussian Blur
        if (op == "gaussian") {
            Image g_input = input_img;
            Image cpu_g, gpu_naive_g, gpu_tiled_g;

            std::cout << "[3] Running " << filter_size << "x" << filter_size << " Gaussian Blur...\n";

            if (mode == "cpu" || mode == "both") {
                auto t0 = std::chrono::high_resolution_clock::now();
                cpu_g = cpu::gaussian_blur(g_input, filter_size);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "    CPU Gaussian Blur Time     : " << ms << " ms\n";
            }

            if (mode == "cuda" || mode == "both") {
                CudaStream stream;
                DeviceImage d_src = to_device(g_input, stream);

                // Naive kernel timing
                cudaEvent_t ev0, ev1;
                CUDA_CHECK(cudaEventCreate(&ev0));
                CUDA_CHECK(cudaEventCreate(&ev1));

                CUDA_CHECK(cudaEventRecord(ev0, stream));
                DeviceImage d_naive = cuda::gaussian_blur_naive(d_src, filter_size, stream);
                CUDA_CHECK(cudaEventRecord(ev1, stream));
                CUDA_CHECK(cudaEventSynchronize(ev1));

                float naive_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&naive_ms, ev0, ev1));
                gpu_naive_g = to_host(d_naive, stream);
                std::cout << "    CUDA Naive Gaussian Blur Time: " << naive_ms << " ms\n";

                // Tiled kernel timing (16x16) - 3x3 has specialized tiled kernel
                if (filter_size == 3) {
                    CUDA_CHECK(cudaEventRecord(ev0, stream));
                    DeviceImage d_tiled = cuda::gaussian_blur_tiled_3x3(d_src, cuda::TileConfig::TILE_16x16, stream);
                    CUDA_CHECK(cudaEventRecord(ev1, stream));
                    CUDA_CHECK(cudaEventSynchronize(ev1));

                    float tiled_ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, ev0, ev1));
                    gpu_tiled_g = to_host(d_tiled, stream);
                    std::cout << "    CUDA Tiled Gaussian Blur Time: " << tiled_ms << " ms\n";
                } else {
                    gpu_tiled_g = gpu_naive_g;
                }

                CUDA_CHECK(cudaEventDestroy(ev0));
                CUDA_CHECK(cudaEventDestroy(ev1));
            }

            if (mode == "both") {
                cpu::ImageDiffStats stats_naive = cpu::compute_diff_stats(cpu_g, gpu_naive_g);
                std::cout << "\n    Validation:\n"
                          << "      - Naive CUDA Match: " << (100.0 * stats_naive.identical_pixels / stats_naive.total_pixels) << "%\n";
                if (filter_size == 3) {
                    cpu::ImageDiffStats stats_tiled = cpu::compute_diff_stats(cpu_g, gpu_tiled_g);
                    std::cout << "      - Tiled CUDA Match: " << (100.0 * stats_tiled.identical_pixels / stats_tiled.total_pixels) << "%\n";
                }

                std::string k_tag = "_" + std::to_string(filter_size) + "x" + std::to_string(filter_size);
                std::string cpu_save = (parent / (stem + "_gaussian" + k_tag + "_cpu" + ext)).string();
                std::string naive_save = (parent / (stem + "_gaussian" + k_tag + "_naive_cuda" + ext)).string();

                save_image(cpu_save, cpu_g);
                save_image(naive_save, gpu_naive_g);
                if (filter_size == 3) {
                    std::string tiled_save = (parent / (stem + "_gaussian" + k_tag + "_tiled_cuda" + ext)).string();
                    save_image(tiled_save, gpu_tiled_g);
                    save_image(output_path, gpu_tiled_g);
                } else {
                    save_image(output_path, gpu_naive_g);
                }

                std::cout << "\n    Saved Gaussian Blurred Images:\n"
                          << "      - Output Image    : " << output_path << "\n"
                          << "      - CPU Reference   : " << cpu_save << "\n"
                          << "      - CUDA Generated  : " << naive_save << "\n";
            } else if (mode == "cpu") {
                save_image(output_path, cpu_g);
                std::cout << "    Saved CPU Gaussian Blur output to: " << output_path << "\n";
            } else if (mode == "cuda") {
                save_image(output_path, (filter_size == 3) ? gpu_tiled_g : gpu_naive_g);
                std::cout << "    Saved CUDA Gaussian Blur output to: " << output_path << "\n";
            }
        }

        // Stage 4: Sobel Edge Detection
        if (op == "sobel") {
            Image sobel_input = input_img;
            if (sobel_input.channels == 3) {
                std::cout << "    Input is 3-channel RGB. Converting to grayscale first for Sobel filter...\n";
                sobel_input = cpu::rgb_to_grayscale(sobel_input);
            }

            Image cpu_sobel, gpu_naive_sobel, gpu_tiled_sobel;

            std::cout << "[3] Running 3x3 Sobel Edge Detection (Gx, Gy, Magnitude = sqrt(Gx^2 + Gy^2))...\n";

            if (mode == "cpu" || mode == "both") {
                auto t0 = std::chrono::high_resolution_clock::now();
                cpu_sobel = cpu::sobel_edge_detect(sobel_input);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "    CPU Sobel Edge Detect Time     : " << ms << " ms\n";
            }

            if (mode == "cuda" || mode == "both") {
                CudaStream stream;
                DeviceImage d_src = to_device(sobel_input, stream);

                // Naive kernel timing
                cudaEvent_t ev0, ev1;
                CUDA_CHECK(cudaEventCreate(&ev0));
                CUDA_CHECK(cudaEventCreate(&ev1));

                CUDA_CHECK(cudaEventRecord(ev0, stream));
                DeviceImage d_naive = cuda::sobel_edge_detect_naive(d_src, stream);
                CUDA_CHECK(cudaEventRecord(ev1, stream));
                CUDA_CHECK(cudaEventSynchronize(ev1));

                float naive_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&naive_ms, ev0, ev1));
                gpu_naive_sobel = to_host(d_naive, stream);
                std::cout << "    CUDA Naive Sobel Edge Time     : " << naive_ms << " ms\n";

                // Tiled kernel timing (16x16)
                CUDA_CHECK(cudaEventRecord(ev0, stream));
                DeviceImage d_tiled = cuda::sobel_edge_detect_tiled(d_src, cuda::TileConfig::TILE_16x16, stream);
                CUDA_CHECK(cudaEventRecord(ev1, stream));
                CUDA_CHECK(cudaEventSynchronize(ev1));

                float tiled_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, ev0, ev1));
                gpu_tiled_sobel = to_host(d_tiled, stream);
                std::cout << "    CUDA Tiled Sobel Edge Time     : " << tiled_ms << " ms\n";

                CUDA_CHECK(cudaEventDestroy(ev0));
                CUDA_CHECK(cudaEventDestroy(ev1));
            }

            if (mode == "both") {
                cpu::ImageDiffStats stats_naive = cpu::compute_diff_stats(cpu_sobel, gpu_naive_sobel);
                cpu::ImageDiffStats stats_tiled = cpu::compute_diff_stats(cpu_sobel, gpu_tiled_sobel);

                std::cout << "\n    Validation:\n"
                          << "      - Naive CUDA Match: " << (100.0 * stats_naive.identical_pixels / stats_naive.total_pixels) << "%\n"
                          << "      - Tiled CUDA Match: " << (100.0 * stats_tiled.identical_pixels / stats_tiled.total_pixels) << "%\n";

                std::string cpu_save = (parent / (stem + "_sobel_cpu" + ext)).string();
                std::string naive_save = (parent / (stem + "_sobel_naive_cuda" + ext)).string();
                std::string tiled_save = (parent / (stem + "_sobel_tiled_cuda" + ext)).string();

                save_image(cpu_save, cpu_sobel);
                save_image(naive_save, gpu_naive_sobel);
                save_image(tiled_save, gpu_tiled_sobel);
                save_image(output_path, gpu_tiled_sobel);

                std::cout << "\n    Saved Sobel Edge-Detected Images:\n"
                          << "      - Output Image    : " << output_path << "\n"
                          << "      - CPU Reference   : " << cpu_save << "\n"
                          << "      - Naive CUDA      : " << naive_save << "\n"
                          << "      - Tiled CUDA      : " << tiled_save << "\n";
            } else if (mode == "cpu") {
                save_image(output_path, cpu_sobel);
                std::cout << "    Saved CPU Sobel output to: " << output_path << "\n";
            } else if (mode == "cuda") {
                save_image(output_path, gpu_tiled_sobel);
                std::cout << "    Saved CUDA Sobel output to: " << output_path << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
}



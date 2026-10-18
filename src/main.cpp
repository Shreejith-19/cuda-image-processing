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
              << "  --mode <cpu|cuda|both>       Execution target (default: both)\n"
              << "  --op <grayscale|blur|both>   Operation to execute (default: both)\n"
              << "  --filter-size <3|5|...>      Odd filter dimension for mean blur (default: 3)\n"
              << "  --input <path>               Path to input image (default: images/input/sample.png)\n"
              << "  --output <path>              Path to output image (default: images/output/result.png)\n"
              << "  --help                       Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=================================================================\n";
    std::cout << " CUDA Image Processing - RGB Grayscale & Mean Blur Accelerator   \n";
    std::cout << "=================================================================\n";

    std::string mode = "both";
    std::string op = "both";
    int filter_size = 3;
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

    if (op != "grayscale" && op != "blur" && op != "both") {
        std::cerr << "Invalid op '" << op << "'. Expected 'grayscale', 'blur', or 'both'.\n";
        print_usage(argv[0]);
        return 1;
    }

    if (filter_size <= 0 || filter_size % 2 == 0) {
        std::cerr << "Invalid filter-size " << filter_size << ". Must be a positive odd integer (e.g. 3, 5).\n";
        return 1;
    }

    std::cout << "Execution Mode : [" << mode << "]\n";
    std::cout << "Operation      : [" << op << "]\n";
    if (op == "blur" || op == "both") {
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
        if (op == "blur") {
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

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
}


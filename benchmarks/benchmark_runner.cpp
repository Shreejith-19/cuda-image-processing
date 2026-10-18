#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include "cuda_image_proc/benchmark.hpp"

using namespace cuda_image_proc;
using namespace cuda_image_proc::benchmark;

void print_banner(const cuda::DeviceInfo& dev) {
    std::cout << "========================================================================================================\n";
    std::cout << " CUDA Image Processing: Unified Reusable Benchmarking Framework                                          \n";
    std::cout << "========================================================================================================\n";
    std::cout << "GPU Device           : " << dev.name << " (Compute " << dev.major << "." << dev.minor << ")\n";
    std::cout << "Total Device Memory  : " << (dev.total_memory_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "Streaming Multiprocs : " << dev.sm_count << " SMs\n";
    std::cout << "Max Threads / Block  : " << dev.max_threads_per_block << "\n";
    std::cout << "Warp Size            : " << dev.warp_size << " threads\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";
    std::cout << "Measurement Policy & Timing Methodology:\n";
    std::cout << "  * GPU Kernel Time  : Measured strictly via hardware CUDA events (cudaEvent_t).\n";
    std::cout << "                       [EXCLUDES: Host/device memory allocations, H2D/D2H transfers, and disk I/O]\n";
    std::cout << "  * GPU End-to-End   : Measured via high-resolution host clock (std::chrono::high_resolution_clock).\n";
    std::cout << "                       [INCLUDES: Host-to-Device transfer + Kernel launch + Device-to-Host transfer]\n";
    std::cout << "                       [EXCLUDES: Disk file reading and disk image writing; buffers pre-allocated]\n";
    std::cout << "  * CPU Execution    : Measured via high-resolution host clock for pure in-memory compute.\n";
    std::cout << "  * Statistics       : Reports Median, Arithmetic Mean, Min, Max, and Sample Std Dev across runs.\n";
    std::cout << "========================================================================================================\n\n";
}

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --op <grayscale|blur|gaussian|sobel|all>   Target operation (default: all)\n"
              << "  --resolution <small|medium|large|ultra|irregular|all|WxH> Image resolution (default: all)\n"
              << "  --filter-size <3|5|7>                      Filter size for mean blur (default: 3)\n"
              << "  --iterations <N>                           Number of timed iterations per benchmark (default: 20)\n"
              << "  --warmup <W>                               Number of warmup runs per benchmark (default: 5)\n"
              << "  --output <path>                            Output CSV filepath (default: results/benchmark_results.csv)\n"
              << "  --test-mode                                Quick 1-iteration format validation run\n"
              << "  --help                                     Show this usage information\n";
}

struct Resolution {
    std::string name;
    int width;
    int height;
};

int main(int argc, char* argv[]) {
    std::string op_filter = "all";
    std::string res_filter = "all";
    int filter_size = 3;
    int iterations = 20;
    int warmup = 5;
    std::string csv_output = "results/benchmark_results.csv";
    bool test_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--op" && i + 1 < argc) {
            op_filter = argv[++i];
        } else if (arg == "--resolution" && i + 1 < argc) {
            res_filter = argv[++i];
        } else if (arg == "--filter-size" && i + 1 < argc) {
            filter_size = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            csv_output = argv[++i];
        } else if (arg == "--test-mode") {
            test_mode = true;
            iterations = 1;
            warmup = 1;
            csv_output = "results/test_benchmark_format.csv";
        } else if (arg == "--help") {
            print_help(argv[0]);
            return 0;
        }
    }

    cuda::DeviceInfo dev = cuda::get_device_info(0);
    print_banner(dev);

    std::vector<Resolution> resolutions;
    if (test_mode) {
        resolutions = {{"Test 128x128", 128, 128}};
        std::cout << "[TEST MODE ACTIVATED] Running lightweight single-pass run to verify CSV output format.\n\n";
    } else if (res_filter == "small") {
        resolutions = {{"Small (256x256)", 256, 256}};
    } else if (res_filter == "medium") {
        resolutions = {{"Medium (512x512)", 512, 512}};
    } else if (res_filter == "large") {
        resolutions = {{"Large (1024x1024)", 1024, 1024}};
    } else if (res_filter == "ultra") {
        resolutions = {{"Ultra (2048x2048)", 2048, 2048}};
    } else if (res_filter == "irregular") {
        resolutions = {{"Irregular (831x517)", 831, 517}};
    } else if (res_filter == "all") {
        resolutions = {
            {"Small (256x256)", 256, 256},
            {"Medium (512x512)", 512, 512},
            {"Large (1024x1024)", 1024, 1024},
            {"Ultra (2048x2048)", 2048, 2048},
            {"Irregular (831x517)", 831, 517}
        };
    } else {
        // Custom WxH parsing
        size_t x_pos = res_filter.find('x');
        if (x_pos != std::string::npos) {
            int w = std::stoi(res_filter.substr(0, x_pos));
            int h = std::stoi(res_filter.substr(x_pos + 1));
            resolutions = {{"Custom (" + res_filter + ")", w, h}};
        } else {
            std::cerr << "Invalid resolution specification: " << res_filter << "\n";
            return 1;
        }
    }

    std::vector<BenchmarkRecord> all_records;

    for (const auto& res : resolutions) {
        std::cout << ">>> Running Benchmarks on Resolution: " << res.name
                  << " (" << res.width << "x" << res.height << ", "
                  << (res.width * res.height) << " pixels) | Iterations: " << iterations
                  << " (Warmup: " << warmup << ") <<<\n";

        OperationBenchmark bench(res.width, res.height, iterations, warmup);

        // Header for table display
        std::cout << std::left
                  << std::setw(20) << "Operation"
                  << std::setw(20) << "Implementation"
                  << std::setw(12) << "Tile/Block"
                  << std::setw(16) << "Kernel Mean(ms)"
                  << std::setw(16) << "Kernel Med(ms)"
                  << std::setw(16) << "E2E Mean(ms)"
                  << std::setw(14) << "Speedup(CPU)"
                  << std::setw(14) << "Speedup(Naive)"
                  << "\n";
        std::cout << std::string(128, '-') << "\n";

        auto print_record = [](const BenchmarkRecord& r) {
            std::string tile_str = (r.tile_width > 0) ? (std::to_string(r.tile_width) + "x" + std::to_string(r.tile_height)) : "-";
            std::cout << std::left
                      << std::setw(20) << r.operation
                      << std::setw(20) << r.implementation
                      << std::setw(12) << tile_str
                      << std::fixed << std::setprecision(4)
                      << std::setw(16) << r.kernel_time_ms
                      << std::setw(16) << r.median_time_ms
                      << std::setw(16) << r.end_to_end_time_ms
                      << std::setprecision(2)
                      << std::setw(14) << (std::to_string(r.speedup_vs_cpu) + "x")
                      << std::setw(14) << (std::to_string(r.speedup_vs_naive) + "x")
                      << "\n";
        };

        // 1. Grayscale
        if (op_filter == "all" || op_filter == "grayscale") {
            auto recs = bench.run_grayscale();
            for (const auto& r : recs) {
                print_record(r);
                all_records.push_back(r);
            }
        }

        // 2. Mean Blur
        if (op_filter == "all" || op_filter == "blur") {
            auto recs = bench.run_mean_blur(filter_size);
            for (const auto& r : recs) {
                print_record(r);
                all_records.push_back(r);
            }
        }

        // 3. Gaussian Blur
        if (op_filter == "all" || op_filter == "gaussian") {
            auto recs = bench.run_gaussian_blur();
            for (const auto& r : recs) {
                print_record(r);
                all_records.push_back(r);
            }
        }

        // 4. Sobel Edge Detection
        if (op_filter == "all" || op_filter == "sobel") {
            auto recs = bench.run_sobel();
            for (const auto& r : recs) {
                print_record(r);
                all_records.push_back(r);
            }
        }

        std::cout << "\n";
    }

    std::cout << "Writing benchmark data to CSV: " << csv_output << "... ";
    export_results_to_csv(csv_output, all_records);
    std::cout << "Done! (" << all_records.size() << " records saved).\n\n";

    return 0;
}

#include <iostream>
#include <string>
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

int main(int argc, char* argv[]) {
    std::cout << "====================================================\n";
    std::cout << " CUDA Image Processing - Foundational Architecture  \n";
    std::cout << "====================================================\n";

    std::string input_path = "images/input/sample.png";
    std::string output_path = "images/output/sample_copy.png";

    if (argc >= 2) {
        input_path = argv[1];
    }
    if (argc >= 3) {
        output_path = argv[2];
    }

    std::cout << "Input file : " << input_path << "\n";
    std::cout << "Output file: " << output_path << "\n\n";

    try {
        // 1. Query CUDA Hardware Information
        cuda::DeviceInfo dev_info = cuda::get_device_info(0);
        std::cout << "[CUDA Device] " << dev_info.name 
                  << " (Compute " << dev_info.major << "." << dev_info.minor << ")\n";
        std::cout << "              VRAM: " << (dev_info.total_memory_bytes / (1024 * 1024)) << " MB, "
                  << "SMs: " << dev_info.sm_count << "\n\n";

        // 2. Load the RGB Image
        std::cout << "[Step 1] Loading image...\n";
        Image img = load_image(input_path, 3); // Force 3 channels (RGB)

        std::cout << "  - Width         : " << img.width << " px\n";
        std::cout << "  - Height        : " << img.height << " px\n";
        std::cout << "  - Channels      : " << img.channels << " (RGB)\n";
        std::cout << "  - Row Stride    : " << img.stride_bytes << " bytes\n";
        std::cout << "  - Total Bytes   : " << img.total_bytes() << " bytes\n";
        std::cout << "  - Tightly Packed: " << (img.is_tightly_packed() ? "Yes" : "No") << "\n\n";

        // 3. Save a Copy
        std::cout << "[Step 2] Saving copy to disk...\n";
        save_image(output_path, img);
        std::cout << "  Successfully written to: " << output_path << "\n\n";

        // 4. Verify Host Round-trip (Load back and compare)
        std::cout << "[Step 3] Verifying host load/save integrity...\n";
        Image reloaded = load_image(output_path, 3);
        if (!cpu::compare_images(img, reloaded, 0.0f)) {
            std::cerr << "  ERROR: Saved image content differs from original!\n";
            return 1;
        }
        std::cout << "  Verification PASSED: Saved image is identical to original.\n\n";

        // 5. Verify CUDA GPU Memory Transfer & Pitched Alignment
        std::cout << "[Step 4] Verifying CUDA RAII Device Memory & 2D Transfer...\n";
        DeviceImage d_img = to_device(img);
        std::cout << "  - Device pitch  : " << d_img.stride_bytes() << " bytes (aligned by CUDA driver)\n";
        std::cout << "  - Device memory : Allocated & managed by RAII DeviceImage\n";

        HostImage from_gpu = to_host(d_img);
        if (!cpu::compare_images(img, from_gpu, 0.0f)) {
            std::cerr << "  ERROR: Device-to-Host image transfer corruption detected!\n";
            return 1;
        }
        std::cout << "  Verification PASSED: GPU 2D memory transfer is 100% bit-exact.\n\n";

        std::cout << "====================================================\n";
        std::cout << " CHECKPOINT SUCCESS: Image loaded, saved, and CUDA  \n";
        std::cout << " memory subsystem validated without corruption!     \n";
        std::cout << "====================================================\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
}

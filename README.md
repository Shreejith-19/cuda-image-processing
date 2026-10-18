# cuda-image-processing

A high-performance image processing library implementing CPU baseline and optimized CUDA kernels with shared memory optimizations and reproducible benchmarks.

## Hardware Target & Environment
- **Processor**: Intel Core i5-12450H
- **Memory**: 16 GB RAM
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, Ampere SM 8.6)
- **Toolchain**: CUDA Toolkit 13.4, MSVC (C++17), CMake >= 3.20

---

## Directory Architecture

```
cuda-image-processing/
├── include/
│   ├── cuda_image_proc/
│   │   ├── cuda_utils.hpp         # CUDA_CHECK macros & RAII CudaStream wrapper
│   │   ├── image.hpp              # Host Image class (width, height, channels, explicit stride, data)
│   │   ├── device_image.cuh       # RAII DeviceImage (cudaMallocPitch, copy-deleted, moveable)
│   │   ├── memory_transfer.cuh    # 2D stride-aware host <-> device memory copy operations
│   │   ├── image_io.hpp           # Image load/save declarations
│   │   ├── cpu_ops.hpp            # CPU processing operations interface & verification utilities
│   │   └── cuda_ops.cuh           # CUDA processing operations interface & hardware queries
│   └── third_party/
│       └── stb/
│           ├── stb_image.h        # Lightweight header-only image decoder (PNG, JPG, BMP)
│           └── stb_image_write.h  # Lightweight header-only image encoder
├── src/
│   ├── cpu/
│   │   ├── image_io.cpp           # Image loading & saving implementations
│   │   └── cpu_ops.cpp            # CPU image verification and repacking logic
│   ├── cuda/
│   │   └── cuda_ops.cu            # GPU device info & launch configuration helpers
│   ├── smoke_test.cu              # Initial CUDA smoke test
│   └── main.cpp                   # Foundational demo: load, inspect, save & GPU round-trip test
├── tests/                         # Unit and regression test suites
├── benchmarks/                    # Benchmark drivers for CPU vs. GPU profiling
├── analysis/                      # Profiling analysis, charts, and metric notebooks
├── images/
│   ├── input/                     # Benchmark & input image assets
│   └── output/                    # Processed / generated images (git ignored)
├── results/                       # CSV benchmarks and profiling data (git ignored)
├── CMakeLists.txt                 # CMake build configuration with configurable sm_86 architecture
└── .gitignore                     # Git ignore rules for build, VS, Python, and image outputs
```

---

## Core Architecture Principles

1. **Explicit Row Stride Handling (`stride_bytes`)**:
   Instead of assuming tightly packed buffers ($width \times channels$), all image operations and transfers compute row offsets using `stride_bytes`. This allows seamless interoperability with padded rows, SIMD-aligned allocations, and GPU pitched memory.

2. **Strict Host/Device Memory Separation & RAII**:
   - `HostImage` (`Image`): Holds host memory in a `std::vector<uint8_t>` buffer.
   - `DeviceImage`: Manages GPU memory allocated via `cudaMallocPitch`. Copy construction and copy assignment are explicitly deleted (`= delete`) to avoid accidental double frees or shallow device pointer copies. Move semantics (`DeviceImage&&`) and explicit cloning (`.clone()`) are supported.
   - Destructors automatically release CUDA resources via `cudaFree`.

3. **2D Stride-Aware Transfers**:
   `to_device(...)` and `to_host(...)` use `cudaMemcpy2D` to copy between host buffers and pitched device memory rows cleanly without manual padding manipulation.

---

## Dependencies

- **CMake** (>= 3.20)
- **C++17 Compiler** (MSVC on Windows, GCC/Clang on Linux)
- **CUDA Toolkit** (>= 11.0, verified with CUDA 13.4)
- **stb_image & stb_image_write** (Included under `include/third_party/stb/`, public domain)

---

## Build and Run Instructions

### 1. Configure the Project
To configure targeting your NVIDIA RTX 3050 Laptop GPU (`sm_86`):
```powershell
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=86
```

### 2. Build the Project
```powershell
cmake --build build --config Release
```

### 3. Run Executables & Verification Suites

- **Run Grayscale & Mean Blur Accelerator Pipeline**:
```powershell
# Run full pipeline (RGB to Grayscale + Mean Blur) comparing CPU vs CUDA:
.\build\Release\cuda_image_app.exe --mode both --op both --filter-size 3

# Apply 5x5 Mean Blur directly on an image:
.\build\Release\cuda_image_app.exe --mode both --op blur --filter-size 5 --input images\input\sample.png --output images\output\blurred_5x5.png
```

- **Run 3x3 Gaussian Blur (CPU vs. Naive CUDA vs. Tiled CUDA)**:
```powershell
.\build\Release\cuda_image_app.exe --mode both --op gaussian --input images\input\sample.png --output images\output\gaussian_sample.png
```

- **Run 3x3 Sobel Edge Detection (CPU vs. Naive CUDA vs. Tiled CUDA)**:
```powershell
.\build\Release\cuda_image_app.exe --mode both --op sobel --input images\input\sample.png --output images\output\sample_sobel.png
```

- **Run Mean Blur Correctness Test Suite (Naive + Tiled 8x8, 16x16, 32x32 across 7 test patterns)**:
```powershell
.\build\Release\test_mean_blur.exe
```

- **Run 3x3 Gaussian Blur Correctness Test Suite (CPU vs. Naive vs. Tiled 8x8, 16x16, 32x32)**:
```powershell
.\build\Release\test_gaussian_blur.exe
```

- **Run 3x3 Sobel Edge Detection Test Suite (Gradient & Magnitude Validation across 7 patterns)**:
```powershell
.\build\Release\test_sobel.exe
```

- **Run Automated Correctness & Robustness Test Suite**:
```powershell
.\build\Release\test_robustness.exe
```

- **Run Unified Reusable Benchmark Suite across all operations & resolutions**:
```powershell
# Run full benchmark across all operations and resolutions, exporting to CSV:
.\build\Release\benchmark_runner.exe --iterations 20 --warmup 5 --output results/benchmark_results.csv

# Run quick verification test-mode generating test CSV:
.\build\Release\benchmark_runner.exe --test-mode

# Run specific operation on a specific resolution:
.\build\Release\benchmark_runner.exe --op sobel --resolution large --iterations 30
.\build\Release\benchmark_runner.exe --op blur --filter-size 5 --resolution 1920x1080
```

- **Run Individual Dedicated Operation Benchmarks**:
```powershell
.\build\Release\benchmark_mean_blur.exe
.\build\Release\benchmark_gaussian_blur.exe
.\build\Release\benchmark_sobel.exe
```

- **Generate Analysis Figures & Performance Report (Python)**:
```powershell
python analysis/generate_figures.py --csv results/benchmark_results.csv --out-dir results/figures --report results/performance_report.md
```

- **Run RGB-to-Grayscale Test Suite**:
```powershell
.\build\Release\test_grayscale.exe
```

- **Run Low-level CUDA Smoke Test**:
```powershell
.\build\Release\smoke_test.exe
```

- **Run All Automated Tests via CTest**:
```powershell
ctest --test-dir build -C Release --output-on-failure
```

---

## Running NVIDIA Compute Sanitizer

NVIDIA Compute Sanitizer is the modern GPU debugging suite replacing `cuda-memcheck`. Use it to diagnose memory bounds violations, synchronization race conditions, and uninitialized reads.

### 1. Memory Checking (`memcheck`)
Detects out-of-bounds global, shared, and pitched memory accesses, illegal addresses, device double-frees, and memory leaks:
```powershell
compute-sanitizer --tool memcheck .\build\Release\test_robustness.exe
compute-sanitizer --tool memcheck .\build\Release\test_sobel.exe
```

### 2. Synchronization Checking (`synccheck`)
Verifies barrier synchronization invariants (ensuring every active thread within a thread block safely reaches `__syncthreads()` and no divergent barriers occur):
```powershell
compute-sanitizer --tool synccheck .\build\Release\test_robustness.exe
```

### 3. Hazard & Race Detection (`racecheck`)
Detects shared-memory read-after-write (RAW), write-after-read (WAR), and write-after-write (WAW) hazards between concurrent threads:
```powershell
compute-sanitizer --tool racecheck .\build\Release\test_robustness.exe
```

### 4. Uninitialized Memory Checking (`initcheck`)
Identifies cases where device memory is read before being explicitly written:
```powershell
compute-sanitizer --tool initcheck .\build\Release\test_robustness.exe
```
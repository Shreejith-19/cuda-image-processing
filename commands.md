# CUDA Image Processing - CLI Command Reference

This document provides ready-to-run commands for processing your own images, running automated test suites, and executing performance benchmarks.

---

## 1. Prerequisites & Building the Project

Before running commands, ensure the project is built in `Release` mode:

```powershell
# Build the entire project from PowerShell in the project root
cmake --build build --config Release
```

All compiled binaries will be placed in `build\Release\`.

Supported image formats: **PNG, JPG / JPEG, BMP**.

---

## 2. Processing Your Own Images (`cuda_image_app.exe`)

The main CLI tool is located at `.\build\Release\cuda_image_app.exe`.

### General Syntax
```powershell
.\build\Release\cuda_image_app.exe --input "<path_to_input>" --output "<path_to_output>" --op <operation> --mode <target>
```

### Available Options
| Option | Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input <path>` | File path (`.png`, `.jpg`, `.bmp`) | `images/input/sample.png` | Path to source image |
| `--output <path>` | File path (`.png`, `.jpg`, `.bmp`) | `images/output/result.png` | Path to save processed image |
| `--op <type>` | `grayscale`, `blur`, `gaussian`, `sobel`, `both` | `both` | Image processing operation to execute |
| `--mode <target>` | `cuda`, `cpu`, `both` | `both` | Hardware target (comparing `both` benchmarks and verifies accuracy) |
| `--filter-size <N>`| Odd positive integer (`3`, `5`, `7`, ...) | `3` | Filter window dimension for mean blur |
| `--help` | | | Display usage information |

---

### Command Examples

#### 1. Full Pipeline Execution (Recommended First Run)
Runs RGB-to-Grayscale conversion, Mean Blur, and Sobel edge detection across both CPU and GPU, benchmarks their runtimes, checks pixel-level accuracy, and saves results:
```powershell
.\build\Release\cuda_image_app.exe --input "C:\path\to\my_photo.jpg" --output "images/output/my_photo_result.png"
```

#### 2. Convert an Image to Grayscale
```powershell
.\build\Release\cuda_image_app.exe --op grayscale --input "C:\path\to\my_photo.jpg" --output "images/output/my_photo_gray.png"
```

#### 3. Sobel Edge Detection (GPU-Accelerated)
Extracts horizontal and vertical edge gradients and computes the edge magnitude map on the GPU:
```powershell
.\build\Release\cuda_image_app.exe --mode cuda --op sobel --input "C:\path\to\my_photo.jpg" --output "images/output/my_photo_edges.png"
```

#### 4. 3x3 Gaussian Blur
Applies a weighted Gaussian filter with normalized coefficients $\frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$:
```powershell
.\build\Release\cuda_image_app.exe --op gaussian --input "C:\path\to\my_photo.jpg" --output "images/output/my_photo_gaussian.png"
```

#### 5. Configurable Mean Blur (Box Filter)
Applies a 5x5 box filter (radius 2) with clamp-to-edge boundary condition:
```powershell
.\build\Release\cuda_image_app.exe --op blur --filter-size 5 --input "C:\path\to\my_photo.jpg" --output "images/output/my_photo_blur5x5.png"
```

#### 6. High-Performance GPU-Only Mode
Skips CPU computation to process frames with minimal latency:
```powershell
.\build\Release\cuda_image_app.exe --mode cuda --op sobel --input "images/input/sample.png" --output "images/output/sample_edges.png"
```

---

## 3. Automated Correctness & Robustness Tests

To run the complete automated test suite verifying exact pixel agreement, numerical tolerances, and edge cases:

```powershell
# Run all 6 test suites via CTest
ctest --test-dir build -C Release --output-on-failure
```

Or execute individual test suites directly:
```powershell
.\build\Release\test_fusion.exe         # Kernel fusion test suite
.\build\Release\test_grayscale.exe      # RGB-to-Grayscale test suite
.\build\Release\test_mean_blur.exe      # Mean Blur (3x3 & 5x5) test suite
.\build\Release\test_gaussian_blur.exe  # Gaussian Blur test suite
.\build\Release\test_sobel.exe          # Sobel edge detection test suite
.\build\Release\test_robustness.exe     # Boundary & error handling test suite
```

---

## 4. Benchmarking Framework

To measure kernel latencies, end-to-end memory transfers, and speedups:

```powershell
# Benchmark Kernel Fusion (Baseline pipeline vs. Fused Tiled vs. Fused Naive)
.\build\Release\benchmark_fusion.exe

# Benchmark all operations across resolutions (256x256 to 2048x2048)
.\build\Release\benchmark_runner.exe

# Benchmark individual filters:
.\build\Release\benchmark_mean_blur.exe
.\build\Release\benchmark_gaussian_blur.exe
.\build\Release\benchmark_sobel.exe
```

Results are printed in formatted tables and automatically exported to CSV files in `results/`:
- `results/benchmark_fusion.csv`
- `results/benchmark_results.csv`

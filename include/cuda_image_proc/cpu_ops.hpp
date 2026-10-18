#pragma once

#include "cuda_image_proc/image.hpp"
#include <cstddef>

namespace cuda_image_proc {
namespace cpu {

struct ImageDiffStats {
    bool dimensions_match{false};
    size_t total_pixels{0};
    size_t identical_pixels{0};
    size_t mismatched_pixels{0};
    double max_diff{0.0};
    double mean_absolute_error{0.0};
    double root_mean_squared_error{0.0};
};

/**
 * @brief Checks if two images have matching dimensions, channel counts, and pixel values.
 *
 * Handles differing row strides by comparing pixel by pixel.
 *
 * @param a First image
 * @param b Second image
 * @param tolerance Maximum allowed difference per channel per pixel
 * @return true if images match within tolerance
 */
bool compare_images(const Image& a, const Image& b, float tolerance = 0.0f);

/**
 * @brief Computes detailed pixel-by-pixel difference statistics between two images.
 */
ImageDiffStats compute_diff_stats(const Image& a, const Image& b);

/**
 * @brief Creates a deep copy of an image with tight packing (row stride = width * channels).
 */
Image pack_tightly(const Image& src);

/**
 * @brief Converts an RGB image (3 channels) to Grayscale (1 channel) on CPU.
 *
 * Uses standard luminance formula: Y = 0.299 R + 0.587 G + 0.114 B
 * Rounding: std::round (half away from zero / nearest)
 * Clamping: [0, 255]
 *
 * @param rgb_image Input RGB image with uint8_t pixels and explicit row stride.
 * @return Image Grayscale 1-channel image.
 * @throws std::invalid_argument If image is empty or channel count != 3.
 */
Image rgb_to_grayscale(const Image& rgb_image);

/**
 * @brief Applies mean blur (box filter) to an image on the CPU.
 *
 * Border policy: Clamp-to-edge (coordinates outside [0, width-1] or [0, height-1]
 * are clamped to the nearest boundary pixel).
 *
 * Accumulation: 32-bit unsigned integer accumulation (sum of uint8 values)
 * followed by floating-point division by filter area (filter_size * filter_size).
 *
 * Rounding and clamping: std::round (half away from zero / nearest), clamped to [0, 255].
 *
 * @param src Input image with uint8_t pixels.
 * @param filter_size Odd filter dimension K (e.g. 3 for 3x3, 5 for 5x5).
 * @return Image Output blurred image with matching dimensions and channels.
 * @throws std::invalid_argument If image is empty or filter_size is not a positive odd integer.
 */
Image mean_blur(const Image& src, int filter_size = 3);

/**
 * @brief Applies 3x3 Gaussian blur to an image on the CPU.
 *
 * Kernel weights:
 * [ 1  2  1 ]
 * [ 2  4  2 ] / 16.0f
 * [ 1  2  1 ]
 *
 * Border policy: Clamp-to-edge (coordinates outside [0, width-1] or [0, height-1]
 * are clamped to the nearest boundary pixel).
 *
 * Accumulation: Floating-point accumulation using normalized weights (sum += pixel * weight).
 *
 * Rounding and clamping: std::round (half away from zero / nearest), clamped to [0, 255].
 *
 * @param src Input image with uint8_t pixels.
 * @return Image Output blurred image with matching dimensions and channels.
 * @throws std::invalid_argument If image is empty.
 */
Image gaussian_blur_3x3(const Image& src);

/**
 * @brief Applies 5x5 Gaussian blur to an image on the CPU.
 *
 * Kernel weights: Pascal / binomial normalized 2D Gaussian kernel (sum = 256.0f).
 *
 * @param src Input image with uint8_t pixels.
 * @return Image Output blurred image with matching dimensions and channels.
 * @throws std::invalid_argument If image is empty.
 */
Image gaussian_blur_5x5(const Image& src);

/**
 * @brief Computes normalized 2D Gaussian kernel weights for an odd filter size.
 *
 * For 3x3 and 5x5, returns the exact standard binomial weights.
 * For larger sizes (7, 9, ..., 31), computes normalized weights using standard Gaussian distribution.
 *
 * @param filter_size Odd filter dimension K.
 * @return std::vector<float> Normalized weights in row-major order (sum = 1.0f).
 */
std::vector<float> compute_gaussian_kernel(int filter_size);

/**
 * @brief Applies Gaussian blur with configurable filter size (any positive odd integer, e.g. 3, 5, 7, ..., 21, 31) on the CPU.
 */
Image gaussian_blur(const Image& src, int filter_size = 3);


struct SobelPixelGradient {
    float gx{0.0f};
    float gy{0.0f};
    float magnitude{0.0f};
};

/**
 * @brief Computes raw Sobel gradients (Gx, Gy) and magnitude at a specific coordinate (x, y).
 *
 * Uses clamp-to-edge border policy for boundary safety.
 * Gx operator: [-1 0 1; -2 0 2; -1 0 1]
 * Gy operator: [-1 -2 -1;  0 0 0;  1 2 1]
 * Magnitude: sqrt(Gx^2 + Gy^2)
 */
SobelPixelGradient compute_sobel_at(const Image& src, int x, int y, int channel = 0);

/**
 * @brief Computes raw floating-point Sobel gradients across the entire image.
 *
 * Useful for validating raw intermediate gradients before magnitude conversion.
 */
void compute_sobel_gradients(const Image& src,
                             std::vector<float>& gx,
                             std::vector<float>& gy,
                             std::vector<float>& magnitude,
                             int channel = 0);

/**
 * @brief Applies Sobel edge detection to an image on the CPU.
 *
 * Standard 3x3 Sobel kernels:
 * Gx = [-1 0 1; -2 0 2; -1 0 1]
 * Gy = [-1 -2 -1; 0 0 0; 1 2 1]
 *
 * Intermediates: Signed floating-point gradients (Gx, Gy). Negative values are preserved.
 * Magnitude calculation: magnitude = sqrt(Gx*Gx + Gy*Gy).
 * Scaling & clamping policy: std::round(magnitude), clamped to [0, 255].
 * Border policy: Clamp-to-edge.
 *
 * @param src Input 1-channel grayscale or multi-channel image.
 * @return Image Grayscale edge magnitude map (1-channel).
 * @throws std::invalid_argument If image is empty.
 */
Image sobel_edge_detect(const Image& src);

/**
 * @brief Performs RGB-to-grayscale conversion followed by mean blur (box filter) on CPU.
 *
 * Serves as the ground-truth reference for verifying fused CUDA kernels.
 *
 * @param rgb_image Input RGB image with 3 channels.
 * @param filter_size Odd filter dimension (default: 3).
 * @return Image Output 1-channel blurred grayscale image.
 * @throws std::invalid_argument If image is empty or channels != 3.
 */
Image rgb_to_mean_blur(const Image& rgb_image, int filter_size = 3);

} // namespace cpu
} // namespace cuda_image_proc




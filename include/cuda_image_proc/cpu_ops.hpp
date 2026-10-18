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

} // namespace cpu
} // namespace cuda_image_proc


#pragma once

#include "cuda_image_proc/image.hpp"

namespace cuda_image_proc {
namespace cpu {

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
 * @brief Creates a deep copy of an image with tight packing (row stride = width * channels).
 */
Image pack_tightly(const Image& src);

/**
 * @brief Future image processing operations interface for CPU implementations.
 * (Grayscale, Blur, Sobel to be implemented in subsequent phases)
 */

} // namespace cpu
} // namespace cuda_image_proc

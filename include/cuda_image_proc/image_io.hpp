#pragma once

#include <string>
#include "cuda_image_proc/image.hpp"

namespace cuda_image_proc {

/**
 * @brief Loads an image from file (PNG, JPG, BMP) into an Image object.
 *
 * @param filepath Path to the image file.
 * @param desired_channels Desired channel count (0: auto-detect, 1: Grayscale, 3: RGB, 4: RGBA).
 * @return Image The loaded image object.
 * @throws std::runtime_error If the file does not exist or decoding fails.
 */
Image load_image(const std::string& filepath, int desired_channels = 0);

/**
 * @brief Saves an Image object to disk. Deduced format by extension (.png, .jpg, .bmp).
 *
 * Correctly accounts for row stride in bytes.
 *
 * @param filepath Output path on disk.
 * @param image Image to save.
 * @param quality Quality setting for JPEG format (1-100, default 95).
 * @return true if successful.
 * @throws std::runtime_error If saving fails or format is unsupported.
 */
bool save_image(const std::string& filepath, const Image& image, int quality = 95);

} // namespace cuda_image_proc

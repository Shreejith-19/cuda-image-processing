#include "cuda_image_proc/image_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <iostream>

namespace fs = std::filesystem;

namespace cuda_image_proc {

Image load_image(const std::string& filepath, int desired_channels) {
    if (!fs::exists(filepath)) {
        throw std::runtime_error("File does not exist: " + filepath);
    }

    int width = 0;
    int height = 0;
    int channels_in_file = 0;

    uint8_t* raw = stbi_load(filepath.c_str(), &width, &height, &channels_in_file, desired_channels);
    if (!raw) {
        std::string reason = stbi_failure_reason() ? stbi_failure_reason() : "Unknown error";
        throw std::runtime_error("Failed to load image from " + filepath + ": " + reason);
    }

    int actual_channels = (desired_channels > 0) ? desired_channels : channels_in_file;
    size_t row_stride = static_cast<size_t>(width) * static_cast<size_t>(actual_channels);
    size_t total_bytes = row_stride * static_cast<size_t>(height);

    std::vector<uint8_t> buffer(raw, raw + total_bytes);
    stbi_image_free(raw);

    return Image(width, height, actual_channels, row_stride, std::move(buffer));
}

bool save_image(const std::string& filepath, const Image& image, int quality) {
    if (image.is_empty()) {
        throw std::runtime_error("Cannot save empty image: " + filepath);
    }

    fs::path out_path(filepath);
    if (out_path.has_parent_path()) {
        fs::create_directories(out_path.parent_path());
    }

    std::string ext = out_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });

    int w = image.width;
    int h = image.height;
    int comp = image.channels;
    int stride = static_cast<int>(image.stride_bytes);

    int result = 0;
    if (ext == ".png") {
        result = stbi_write_png(filepath.c_str(), w, h, comp, image.raw_data(), stride);
    } else if (ext == ".bmp") {
        if (image.is_tightly_packed()) {
            result = stbi_write_bmp(filepath.c_str(), w, h, comp, image.raw_data());
        } else {
            // Repack into tightly packed contiguous buffer for format writers expecting no row padding
            std::vector<uint8_t> packed(static_cast<size_t>(w * h * comp));
            size_t row_bytes = static_cast<size_t>(w * comp);
            for (int y = 0; y < h; ++y) {
                std::memcpy(packed.data() + y * row_bytes, image.row_ptr(y), row_bytes);
            }
            result = stbi_write_bmp(filepath.c_str(), w, h, comp, packed.data());
        }
    } else if (ext == ".jpg" || ext == ".jpeg") {
        if (image.is_tightly_packed()) {
            result = stbi_write_jpg(filepath.c_str(), w, h, comp, image.raw_data(), quality);
        } else {
            std::vector<uint8_t> packed(static_cast<size_t>(w * h * comp));
            size_t row_bytes = static_cast<size_t>(w * comp);
            for (int y = 0; y < h; ++y) {
                std::memcpy(packed.data() + y * row_bytes, image.row_ptr(y), row_bytes);
            }
            result = stbi_write_jpg(filepath.c_str(), w, h, comp, packed.data(), quality);
        }
    } else {
        throw std::runtime_error("Unsupported image file extension: " + ext);
    }

    if (!result) {
        throw std::runtime_error("Failed to write image to " + filepath);
    }

    return true;
}

} // namespace cuda_image_proc

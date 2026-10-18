#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <stdexcept>

namespace cuda_image_proc {

enum class ColorFormat {
    GRAYSCALE = 1,
    RGB = 3,
    RGBA = 4
};

/**
 * @brief Host-side image representation with explicit row stride support.
 * 
 * Supports grayscale (1 channel), RGB (3 channels), and RGBA (4 channels)
 * with uint8_t pixel values. The row stride in bytes is handled explicitly
 * and can accommodate padded rows (e.g. for alignment or hardware constraints).
 */
class Image {
public:
    int width{0};
    int height{0};
    int channels{0};
    size_t stride_bytes{0};     // Row stride in bytes (width * channels + padding)
    std::vector<uint8_t> data;  // Host pixel buffer

    Image() = default;

    Image(int w, int h, int c, size_t row_stride = 0)
        : width(w), height(h), channels(c) {
        if (w < 0 || h < 0 || c <= 0) {
            throw std::invalid_argument("Invalid image dimensions or channels");
        }
        size_t min_stride = static_cast<size_t>(w) * static_cast<size_t>(c);
        stride_bytes = (row_stride >= min_stride) ? row_stride : min_stride;
        data.resize(stride_bytes * static_cast<size_t>(h), 0);
    }

    Image(int w, int h, int c, size_t row_stride, std::vector<uint8_t> pixel_data)
        : width(w), height(h), channels(c), stride_bytes(row_stride), data(std::move(pixel_data)) {
        size_t min_stride = static_cast<size_t>(w) * static_cast<size_t>(c);
        if (stride_bytes < min_stride) {
            throw std::invalid_argument("stride_bytes cannot be smaller than width * channels");
        }
        if (data.size() < stride_bytes * static_cast<size_t>(h)) {
            throw std::invalid_argument("Data buffer is smaller than height * stride_bytes");
        }
    }

    // Accessors
    uint8_t* raw_data() noexcept { return data.data(); }
    const uint8_t* raw_data() const noexcept { return data.data(); }

    uint8_t* row_ptr(int y) {
        if (y < 0 || y >= height) throw std::out_of_range("Row index out of range");
        return data.data() + static_cast<size_t>(y) * stride_bytes;
    }

    const uint8_t* row_ptr(int y) const {
        if (y < 0 || y >= height) throw std::out_of_range("Row index out of range");
        return data.data() + static_cast<size_t>(y) * stride_bytes;
    }

    uint8_t& at(int y, int x, int c = 0) {
        if (x < 0 || x >= width || c < 0 || c >= channels) {
            throw std::out_of_range("Pixel coordinates or channel out of range");
        }
        return *(row_ptr(y) + static_cast<size_t>(x) * static_cast<size_t>(channels) + static_cast<size_t>(c));
    }

    const uint8_t& at(int y, int x, int c = 0) const {
        if (x < 0 || x >= width || c < 0 || c >= channels) {
            throw std::out_of_range("Pixel coordinates or channel out of range");
        }
        return *(row_ptr(y) + static_cast<size_t>(x) * static_cast<size_t>(channels) + static_cast<size_t>(c));
    }

    bool is_empty() const noexcept {
        return data.empty() || width <= 0 || height <= 0 || channels <= 0;
    }

    bool is_tightly_packed() const noexcept {
        return stride_bytes == (static_cast<size_t>(width) * static_cast<size_t>(channels));
    }

    size_t total_bytes() const noexcept {
        return stride_bytes * static_cast<size_t>(height);
    }

    size_t row_bytes() const noexcept {
        return static_cast<size_t>(width) * static_cast<size_t>(channels);
    }
};

using HostImage = Image;

} // namespace cuda_image_proc

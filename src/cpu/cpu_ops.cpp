#include "cuda_image_proc/cpu_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace cuda_image_proc {
namespace cpu {

bool compare_images(const Image& a, const Image& b, float tolerance) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels) {
        return false;
    }

    for (int y = 0; y < a.height; ++y) {
        const uint8_t* row_a = a.row_ptr(y);
        const uint8_t* row_b = b.row_ptr(y);
        size_t row_bytes = static_cast<size_t>(a.width) * static_cast<size_t>(a.channels);

        for (size_t i = 0; i < row_bytes; ++i) {
            float diff = std::fabs(static_cast<float>(row_a[i]) - static_cast<float>(row_b[i]));
            if (diff > tolerance) {
                return false;
            }
        }
    }

    return true;
}

ImageDiffStats compute_diff_stats(const Image& a, const Image& b) {
    ImageDiffStats stats;
    if (a.width != b.width || a.height != b.height || a.channels != b.channels) {
        stats.dimensions_match = false;
        return stats;
    }

    stats.dimensions_match = true;
    stats.total_pixels = static_cast<size_t>(a.width) * static_cast<size_t>(a.height);

    double sum_diff = 0.0;
    double sum_sq_diff = 0.0;
    double max_diff = 0.0;
    size_t identical = 0;
    size_t mismatched = 0;

    for (int y = 0; y < a.height; ++y) {
        const uint8_t* row_a = a.row_ptr(y);
        const uint8_t* row_b = b.row_ptr(y);

        for (int x = 0; x < a.width; ++x) {
            bool pixel_match = true;
            for (int c = 0; c < a.channels; ++c) {
                size_t idx = static_cast<size_t>(x) * static_cast<size_t>(a.channels) + static_cast<size_t>(c);
                double diff = std::fabs(static_cast<double>(row_a[idx]) - static_cast<double>(row_b[idx]));
                if (diff > 0.0) {
                    pixel_match = false;
                }
                if (diff > max_diff) {
                    max_diff = diff;
                }
                sum_diff += diff;
                sum_sq_diff += diff * diff;
            }
            if (pixel_match) {
                identical++;
            } else {
                mismatched++;
            }
        }
    }

    stats.identical_pixels = identical;
    stats.mismatched_pixels = mismatched;
    stats.max_diff = max_diff;
    size_t total_elements = stats.total_pixels * static_cast<size_t>(a.channels);
    stats.mean_absolute_error = (total_elements > 0) ? (sum_diff / total_elements) : 0.0;
    stats.root_mean_squared_error = (total_elements > 0) ? std::sqrt(sum_sq_diff / total_elements) : 0.0;

    return stats;
}

Image pack_tightly(const Image& src) {
    if (src.is_tightly_packed()) {
        return src;
    }
    Image dst(src.width, src.height, src.channels);
    size_t row_bytes = src.row_bytes();
    for (int y = 0; y < src.height; ++y) {
        std::memcpy(dst.row_ptr(y), src.row_ptr(y), row_bytes);
    }
    return dst;
}

Image rgb_to_grayscale(const Image& rgb_image) {
    if (rgb_image.is_empty()) {
        throw std::invalid_argument("Cannot convert empty image to grayscale");
    }
    if (rgb_image.channels != 3) {
        throw std::invalid_argument("rgb_to_grayscale expects an RGB image with 3 channels, got " +
                                    std::to_string(rgb_image.channels));
    }

    int width = rgb_image.width;
    int height = rgb_image.height;
    Image gray_image(width, height, 1);

    for (int y = 0; y < height; ++y) {
        const uint8_t* in_row = rgb_image.row_ptr(y);
        uint8_t* out_row = gray_image.row_ptr(y);

        for (int x = 0; x < width; ++x) {
            float r = static_cast<float>(in_row[x * 3 + 0]);
            float g = static_cast<float>(in_row[x * 3 + 1]);
            float b = static_cast<float>(in_row[x * 3 + 2]);

            // Standard Luminance Approximation: Y = 0.299 R + 0.587 G + 0.114 B
            float y_val = 0.299f * r + 0.587f * g + 0.114f * b;
            float rounded = std::round(y_val);
            out_row[x] = static_cast<uint8_t>(std::clamp(rounded, 0.0f, 255.0f));
        }
    }

    return gray_image;
}

Image mean_blur(const Image& src, int filter_size) {
    if (src.is_empty()) {
        throw std::invalid_argument("Cannot apply mean_blur to an empty image");
    }
    if (filter_size <= 0 || (filter_size % 2 == 0)) {
        throw std::invalid_argument("filter_size must be a positive odd integer (got " +
                                    std::to_string(filter_size) + ")");
    }

    int width = src.width;
    int height = src.height;
    int channels = src.channels;
    int radius = filter_size / 2;
    float num_samples = static_cast<float>(filter_size * filter_size);

    Image dst(width, height, channels);

    for (int y = 0; y < height; ++y) {
        uint8_t* out_row = dst.row_ptr(y);

        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                uint32_t sum = 0;

                for (int dy = -radius; dy <= radius; ++dy) {
                    // Clamp-to-edge border policy in Y dimension
                    int sy = std::clamp(y + dy, 0, height - 1);
                    const uint8_t* in_row = src.row_ptr(sy);

                    for (int dx = -radius; dx <= radius; ++dx) {
                        // Clamp-to-edge border policy in X dimension
                        int sx = std::clamp(x + dx, 0, width - 1);
                        sum += in_row[static_cast<size_t>(sx) * channels + c];
                    }
                }

                float avg = static_cast<float>(sum) / num_samples;
                float rounded = std::round(avg);
                out_row[static_cast<size_t>(x) * channels + c] = 
                    static_cast<uint8_t>(std::clamp(rounded, 0.0f, 255.0f));
            }
        }
    }

    return dst;
}

} // namespace cpu
} // namespace cuda_image_proc


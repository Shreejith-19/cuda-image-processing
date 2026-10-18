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

Image gaussian_blur_3x3(const Image& src) {
    if (src.is_empty()) {
        throw std::invalid_argument("Cannot apply gaussian_blur_3x3 to an empty image");
    }

    int width = src.width;
    int height = src.height;
    int channels = src.channels;

    Image dst(width, height, channels);

    // 3x3 Gaussian normalized kernel weights (sum = 16.0f)
    const float weights[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };

    for (int y = 0; y < height; ++y) {
        uint8_t* out_row = dst.row_ptr(y);

        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                // Floating-point accumulation
                float sum = 0.0f;

                for (int dy = -1; dy <= 1; ++dy) {
                    int sy = std::clamp(y + dy, 0, height - 1);
                    const uint8_t* in_row = src.row_ptr(sy);

                    for (int dx = -1; dx <= 1; ++dx) {
                        int sx = std::clamp(x + dx, 0, width - 1);
                        float pixel_val = static_cast<float>(in_row[static_cast<size_t>(sx) * channels + c]);
                        sum += pixel_val * weights[dy + 1][dx + 1];
                    }
                }

                float rounded = std::round(sum);
                out_row[static_cast<size_t>(x) * channels + c] = 
                    static_cast<uint8_t>(std::clamp(rounded, 0.0f, 255.0f));
            }
        }
    }

    return dst;
}

Image gaussian_blur_5x5(const Image& src) {
    if (src.is_empty()) {
        throw std::invalid_argument("Cannot apply Gaussian blur to an empty image");
    }

    int width = src.width;
    int height = src.height;
    int channels = src.channels;

    Image dst(width, height, channels);

    // 5x5 normalized binomial Gaussian kernel weights (sum = 256.0f)
    const float weights[5][5] = {
        { 1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f },
        { 4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f },
        { 6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f,  6.0f/256.0f },
        { 4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f },
        { 1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f }
    };

    for (int y = 0; y < height; ++y) {
        uint8_t* out_row = dst.row_ptr(y);

        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;

                for (int dy = -2; dy <= 2; ++dy) {
                    int sy = std::clamp(y + dy, 0, height - 1);
                    const uint8_t* in_row = src.row_ptr(sy);

                    for (int dx = -2; dx <= 2; ++dx) {
                        int sx = std::clamp(x + dx, 0, width - 1);
                        float pixel_val = static_cast<float>(in_row[static_cast<size_t>(sx) * channels + c]);
                        sum += pixel_val * weights[dy + 2][dx + 2];
                    }
                }

                float rounded = std::round(sum);
                out_row[static_cast<size_t>(x) * channels + c] = 
                    static_cast<uint8_t>(std::clamp(rounded, 0.0f, 255.0f));
            }
        }
    }

    return dst;
}

std::vector<float> compute_gaussian_kernel(int filter_size) {
    if (filter_size <= 0 || filter_size % 2 == 0) {
        throw std::invalid_argument("filter_size must be a positive odd integer (got " +
                                    std::to_string(filter_size) + ")");
    }
    if (filter_size == 3) {
        return {
            1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
            2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
            1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f
        };
    }
    if (filter_size == 5) {
        return {
            1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f,
            4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
            6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f,  6.0f/256.0f,
            4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
            1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f
        };
    }

    int radius = filter_size / 2;
    // Standard sigma heuristic: sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    float sigma = 0.3f * ((static_cast<float>(filter_size) - 1.0f) * 0.5f - 1.0f) + 0.8f;
    float two_sigma_sq = 2.0f * sigma * sigma;

    std::vector<float> weights(static_cast<size_t>(filter_size) * filter_size);
    double sum = 0.0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float dist_sq = static_cast<float>(dx * dx + dy * dy);
            float w = std::exp(-dist_sq / two_sigma_sq);
            weights[static_cast<size_t>(dy + radius) * filter_size + (dx + radius)] = w;
            sum += static_cast<double>(w);
        }
    }

    for (float& w : weights) {
        w = static_cast<float>(static_cast<double>(w) / sum);
    }

    return weights;
}

Image gaussian_blur(const Image& src, int filter_size) {
    if (src.is_empty()) {
        throw std::invalid_argument("Cannot apply Gaussian blur to an empty image");
    }
    if (filter_size <= 0 || filter_size % 2 == 0) {
        throw std::invalid_argument("filter_size must be a positive odd integer (got " +
                                    std::to_string(filter_size) + ")");
    }
    if (filter_size == 3) {
        return gaussian_blur_3x3(src);
    }
    if (filter_size == 5) {
        return gaussian_blur_5x5(src);
    }

    int width = src.width;
    int height = src.height;
    int channels = src.channels;
    int radius = filter_size / 2;

    std::vector<float> weights = compute_gaussian_kernel(filter_size);
    Image dst(width, height, channels);

    for (int y = 0; y < height; ++y) {
        uint8_t* out_row = dst.row_ptr(y);

        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                int weight_idx = 0;

                for (int dy = -radius; dy <= radius; ++dy) {
                    int sy = std::clamp(y + dy, 0, height - 1);
                    const uint8_t* in_row = src.row_ptr(sy);

                    for (int dx = -radius; dx <= radius; ++dx) {
                        int sx = std::clamp(x + dx, 0, width - 1);
                        float pixel_val = static_cast<float>(in_row[static_cast<size_t>(sx) * channels + c]);
                        sum += pixel_val * weights[weight_idx++];
                    }
                }

                float rounded = std::round(sum);
                out_row[static_cast<size_t>(x) * channels + c] = 
                    static_cast<uint8_t>(std::clamp(rounded, 0.0f, 255.0f));
            }
        }
    }

    return dst;
}


SobelPixelGradient compute_sobel_at(const Image& src, int x, int y, int channel) {
    int width = src.width;
    int height = src.height;
    int channels = src.channels;

    auto get_pixel = [&](int dy, int dx) -> float {
        int sy = std::clamp(y + dy, 0, height - 1);
        int sx = std::clamp(x + dx, 0, width - 1);
        const uint8_t* row = src.row_ptr(sy);
        return static_cast<float>(row[static_cast<size_t>(sx) * channels + channel]);
    };

    float p00 = get_pixel(-1, -1);
    float p01 = get_pixel(-1,  0);
    float p02 = get_pixel(-1,  1);

    float p10 = get_pixel( 0, -1);
    float p12 = get_pixel( 0,  1);

    float p20 = get_pixel( 1, -1);
    float p21 = get_pixel( 1,  0);
    float p22 = get_pixel( 1,  1);

    // Gx = [-1 0 1; -2 0 2; -1 0 1]
    float gx = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);

    // Gy = [-1 -2 -1; 0 0 0; 1 2 1]
    float gy = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

    float mag = std::sqrt(gx * gx + gy * gy);

    return SobelPixelGradient{gx, gy, mag};
}

void compute_sobel_gradients(const Image& src,
                             std::vector<float>& gx,
                             std::vector<float>& gy,
                             std::vector<float>& magnitude,
                             int channel) {
    if (src.is_empty()) {
        throw std::invalid_argument("Cannot compute Sobel gradients on an empty image");
    }

    size_t total_pixels = static_cast<size_t>(src.width) * static_cast<size_t>(src.height);
    gx.resize(total_pixels);
    gy.resize(total_pixels);
    magnitude.resize(total_pixels);

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            SobelPixelGradient grad = compute_sobel_at(src, x, y, channel);
            size_t idx = static_cast<size_t>(y) * static_cast<size_t>(src.width) + static_cast<size_t>(x);
            gx[idx] = grad.gx;
            gy[idx] = grad.gy;
            magnitude[idx] = grad.magnitude;
        }
    }
}

Image sobel_edge_detect(const Image& src) {
    if (src.is_empty()) {
        throw std::invalid_argument("Cannot apply Sobel edge detection to an empty image");
    }
    if (src.channels != 1) {
        throw std::invalid_argument("sobel_edge_detect expects 1-channel (grayscale) image, got " +
                                    std::to_string(src.channels));
    }

    int width = src.width;
    int height = src.height;
    Image dst(width, height, 1);

    for (int y = 0; y < height; ++y) {
        uint8_t* out_row = dst.row_ptr(y);

        for (int x = 0; x < width; ++x) {
            SobelPixelGradient grad = compute_sobel_at(src, x, y, 0);

            // Output rounding and clamping policy: std::round to nearest integer, clamp to [0, 255]
            float rounded = std::round(grad.magnitude);
            out_row[x] = static_cast<uint8_t>(std::clamp(rounded, 0.0f, 255.0f));
        }
    }

    return dst;
}

Image rgb_to_mean_blur(const Image& rgb_image, int filter_size) {
    if (rgb_image.is_empty()) {
        throw std::invalid_argument("Cannot perform rgb_to_mean_blur on empty image");
    }
    if (rgb_image.channels != 3) {
        throw std::invalid_argument("rgb_to_mean_blur expects 3 channels (RGB), got " +
                                    std::to_string(rgb_image.channels));
    }
    Image gray = rgb_to_grayscale(rgb_image);
    return mean_blur(gray, filter_size);
}

} // namespace cpu
} // namespace cuda_image_proc




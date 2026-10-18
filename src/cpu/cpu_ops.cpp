#include "cuda_image_proc/cpu_ops.hpp"
#include <cmath>

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

} // namespace cpu
} // namespace cuda_image_proc

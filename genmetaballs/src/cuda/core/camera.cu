#include <cstdint>
#include <cuda/std/ranges>
#include <cuda/std/utility>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "geometry.cuh"
#include "utils.cuh"

CUDA_CALLABLE Vec3D Intrinsics::get_ray_direction(uint32_t px, uint32_t py) const {
    auto x = (static_cast<float>(px) - cx) / fx;
    auto y = (static_cast<float>(py) - cy) / fy;
    return Vec3D{x, y, -1.0f};
}

CUDA_CALLABLE PixelCoord PixelCoordRange::Iterator::operator*() const {
    return cuda::std::make_pair(px, py);
}

CUDA_CALLABLE PixelCoordRange::Iterator& PixelCoordRange::Iterator::operator++() {
    ++px;               // move to the next column
    if (px >= px_end) { // move to the next row
        px = px_start;
        ++py;
    }
    return *this;
}

CUDA_CALLABLE bool operator!=(const PixelCoordRange::Iterator& it,
                              const PixelCoordRange::Sentinel& sentinel) {
    // stop if we reach the end of rows, or if the range is empty
    return it.py < sentinel.py_end && it.px_start < it.px_end && it.py_start < sentinel.py_end;
}

CUDA_CALLABLE PixelCoordRange::Iterator PixelCoordRange::begin() const {
    return Iterator{px_start, px_end, py_start, px_start, py_start};
}

CUDA_CALLABLE PixelCoordRange::Sentinel PixelCoordRange::end() const {
    return Sentinel{py_end};
}

CUDA_CALLABLE PixelCoord FlattenedPixelCoordRange::Iterator::operator*() const {
    return cuda::std::make_pair(pixel_idx % width, pixel_idx / width);
}

CUDA_CALLABLE FlattenedPixelCoordRange::Iterator& FlattenedPixelCoordRange::Iterator::operator++() {
    ++pixel_idx;
    return *this;
}

CUDA_CALLABLE bool operator!=(const FlattenedPixelCoordRange::Iterator& it,
                              const FlattenedPixelCoordRange::Sentinel& sentinel) {
    return it.pixel_idx < sentinel.pixel_idx_end && it.pixel_idx_start < it.pixel_idx_end;
}

CUDA_CALLABLE FlattenedPixelCoordRange::Iterator FlattenedPixelCoordRange::begin() const {
    return Iterator{pixel_idx_start, pixel_idx_end, width, height, pixel_idx_start};
}

CUDA_CALLABLE FlattenedPixelCoordRange::Sentinel FlattenedPixelCoordRange::end() const {
    return Sentinel{pixel_idx_end};
}

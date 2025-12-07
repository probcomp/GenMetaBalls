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

CUDA_CALLABLE cuda::std::pair<uint32_t, uint32_t> PixelCoordRange::Iterator::operator*() const {
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

CUDA_CALLABLE bool PixelCoordRange::Sentinel::operator==(const Iterator& it) const {
    // stop if we reach the end of rows, or if the range is empty
    return it.py >= py_end || it.px_start >= it.px_end || it.py_start >= py_end;
}

CUDA_CALLABLE PixelCoordRange::Iterator PixelCoordRange::begin() const {
    return Iterator{px_start, px_end, py_start, px_start, py_start};
}

CUDA_CALLABLE PixelCoordRange::Sentinel PixelCoordRange::end() const {
    return Sentinel{py_end};
}

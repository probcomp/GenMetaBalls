#include <cstdint>
#include <cuda/std/ranges>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "geometry.cuh"
#include "utils.cuh"

CUDA_CALLABLE Vec3D Intrinsics::get_ray_direction(uint32_t px, uint32_t py) const {
    auto x = (static_cast<float>(px) - cx) / fx;
    auto y = (static_cast<float>(py) - cy) / fy;
    return Vec3D{x, y, -1.0f};
}

template <DeviceType device>
CUDA_CALLABLE Array2D<Vec3D, device>& Intrinsics::get_ray_directions(Array2D<Vec3D, device> buffer,
                                                                     uint32_t px_start,
                                                                     uint32_t px_end,
                                                                     uint32_t py_start,
                                                                     uint32_t py_end) const {
    for (auto i = max(0, py_start); i < min(height, py_end); ++i) {
        for (auto j = max(0, px_start); j < min(width, px_end); ++j) {
            buffer[i][j] = get_ray_direction(j, i);
        }
    }
    return buffer;
}

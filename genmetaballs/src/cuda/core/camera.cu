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

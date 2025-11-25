#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "geometry.cuh"
#include "utils.cuh"

struct Intrinsics {
    uint32_t height;
    uint32_t width;
    float fx;
    float fy;
    float cx;
    float cy;

    // Returns the direction of the ray going through pixel (px, py) in camera frame.
    // For efficiency, this function does not check if the pixel is within bounds.
    CUDA_CALLABLE Vec3D get_ray_direction(uint32_t px, uint32_t py) const;

    // Returns a 2D array of ray directions in camera frame in the specified pixel range
    // and store them in the provided buffer. By default, the full image is used
    // For efficiency, this function does not check if Array2D buffer has correct size
    template <DeviceType device>
    CUDA_CALLABLE Array2D<Vec3D, device>& get_ray_directions(Array2D<Vec3D, device> buffer,
                                                             uint32_t px_start = 0,
                                                             uint32_t px_end = UINT32_MAX,
                                                             uint32_t py_start = 0,
                                                             uint32_t py_end = UINT32_MAX) const;
};

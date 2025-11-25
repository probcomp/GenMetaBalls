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

    // returns the direction of the ray going through pixel (px, py) in camera frame
    // for efficiency, this function does not check if the pixel is within bounds
    CUDA_CALLABLE Vec3D get_ray_direction(uint32_t px, uint32_t py) const;
};

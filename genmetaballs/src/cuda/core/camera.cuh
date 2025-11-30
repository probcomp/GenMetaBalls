#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "geometry.cuh"
#include "utils.cuh"

struct Intrinsics {
    uint32_t height; // in x direction
    uint32_t width;  // in y direction
    float fx;
    float fy;
    float cx;
    float cy;

    // Returns the direction of the ray going through pixel (px, py) in camera frame.
    // For efficiency, this function does not check if the pixel is within bounds.
    CUDA_CALLABLE Vec3D get_ray_direction(uint32_t px, uint32_t py) const;

    // Returns a 2D array of ray directions in camera frame in the specified pixel range
    // and store them in the provided buffer. By default, the full image is used
    template <MemoryLocation location>
    CUDA_CALLABLE Array2D<Vec3D, location>& get_ray_directions(Array2D<Vec3D, location> buffer,
                                                               uint32_t px_start = 0,
                                                               uint32_t px_end = UINT32_MAX,
                                                               uint32_t py_start = 0,
                                                               uint32_t py_end = UINT32_MAX) const {
        for (auto i = max(0, px_start); i < min(height, px_end); ++i) {
            for (auto j = max(0, py_start); j < min(width, py_end); ++j) {
                buffer[i][j] = get_ray_direction(j, i);
            }
        }
        return buffer;
    }
};

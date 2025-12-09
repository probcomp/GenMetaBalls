#pragma once

#include <cstdint>
#include <cuda/std/utility>
#include <cuda_runtime.h>

#include "geometry.cuh"
#include "utils.cuh"

struct Intrinsics {
    uint32_t width;  // in x direction
    uint32_t height; // in y direction
    float fx;
    float fy;
    float cx;
    float cy;

    // Returns the direction of the ray going through pixel (px, py) in camera frame.
    // For efficiency, this function does not check if the pixel is within bounds.
    CUDA_CALLABLE Vec3D get_ray_direction(uint32_t px, uint32_t py) const;
};

using PixelCoord = cuda::std::pair<uint32_t, uint32_t>;

struct PixelCoordRange {
    uint32_t px_start;
    uint32_t px_end;
    uint32_t py_start;
    uint32_t py_end;

    // the Iterator class holds the current pixel coordinates
    struct Iterator {
        // pixel range
        uint32_t px_start;
        uint32_t px_end;
        uint32_t py_start;

        // current pixel coordinates
        uint32_t px;
        uint32_t py;

        // Returns the (px, py) coordinates of the current pixel
        CUDA_CALLABLE PixelCoord operator*() const;

        // pre-increment operator that advances to the next pixel
        CUDA_CALLABLE Iterator& operator++();
    };

    // the Sentinel class only needs to hold the stop value (i.e. final row)
    struct Sentinel {
        uint32_t py_end;
    };

    // stopping criterion: true if current row (py) reaches py_end
    friend CUDA_CALLABLE bool operator!=(const Iterator& it, const Sentinel& sentinel);

    // range methods
    CUDA_CALLABLE Iterator begin() const;
    CUDA_CALLABLE Sentinel end() const;
};

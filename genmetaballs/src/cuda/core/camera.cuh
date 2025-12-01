#pragma once

#include <cstdint>
#include <cuda/std/ranges>
#include <cuda/std/utility>
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
    CUDA_CALLABLE Array2D<Vec3D, location>& get_ray_directions(Array2D<Vec3D, location>& buffer,
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

struct PixelCoordRange : public cuda::std::ranges::view_interface<PixelCoordRange> {
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
        CUDA_CALLABLE cuda::std::pair<uint32_t, uint32_t> operator*() const;

        // pre-increment operator that advances to the next pixel
        CUDA_CALLABLE Iterator& operator++();
    };

    // the Sentinel class only needs to hold the stop value (i.e. final row)
    struct Sentinel {
        uint32_t py_end;

        // stopping criterion: true if current row (py) reaches py_end
        CUDA_CALLABLE bool operator==(const Iterator& it) const;
    };

    // range methods
    CUDA_CALLABLE constexpr Iterator begin() const;
    CUDA_CALLABLE constexpr Sentinel end() const;
};

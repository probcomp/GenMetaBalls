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

    CUDA_CALLABLE constexpr uint32_t num_pixels() const noexcept {
        return width * height;
    }
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

// loop over "flattened" pixels (i.e. 1D indexing)
struct FlattenedPixelCoordRange {
    uint32_t pixel_idx_start;
    uint32_t pixel_idx_end;
    uint32_t width;
    uint32_t height;
    uint32_t stride;

    // the Iterator class holds the current pixel coordinates
    struct Iterator {
        // pixel range
        uint32_t pixel_idx_start;
        uint32_t pixel_idx_end;
        uint32_t stride;

        // image dimensions
        uint32_t width;
        uint32_t height;

        // current pixel index
        uint32_t pixel_idx;

        // Returns the pixel index of the current pixel
        CUDA_CALLABLE PixelCoord operator*() const;

        // pre-increment operator that advances to the next pixel
        CUDA_CALLABLE Iterator& operator++();
    };

    // the Sentinel class only needs to hold the stop value (i.e. final row)
    struct Sentinel {
        uint32_t pixel_idx_end;
    };

    // stopping criterion: true if current pixel index reaches pixel_idx_end
    friend CUDA_CALLABLE bool operator!=(const Iterator& it, const Sentinel& sentinel);

    // range methods
    CUDA_CALLABLE Iterator begin() const;
    CUDA_CALLABLE Sentinel end() const;
};

// Declaration of get_pixel_coords (defined in forward.cu)
CUDA_CALLABLE PixelCoordRange get_pixel_coords(const dim3 thread_idx, const dim3 block_idx,
                                               const dim3 block_dim, const dim3 grid_dim,
                                               const Intrinsics& intr);

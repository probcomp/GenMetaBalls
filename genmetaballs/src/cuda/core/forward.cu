#include <cstdint>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "forward.cuh"
#include "utils.cuh"

CUDA_CALLABLE PixelCoordRange get_pixel_coords(const dim3 thread_idx, const dim3 block_idx,
                                               const dim3 block_dim, const dim3 grid_dim,
                                               const Intrinsics& intr) {
    // compute the number of pixels each thread should process
    const auto num_pixels_x = int_ceil_div(intr.width, grid_dim.x * block_dim.x);
    const auto num_pixels_y = int_ceil_div(intr.height, grid_dim.y * block_dim.y);
    const auto start_x = (block_idx.x * block_dim.x + thread_idx.x) * num_pixels_x;
    const auto start_y = (block_idx.y * block_dim.y + thread_idx.y) * num_pixels_y;
    return PixelCoordRange{.px_start = start_x,
                           .px_end = min(start_x + num_pixels_x, intr.width),
                           .py_start = start_y,
                           .py_end = min(start_y + num_pixels_y, intr.height)};
}

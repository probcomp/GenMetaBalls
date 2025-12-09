#include <cstdint>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "forward.cuh"
#include "temp_buffer.cuh"
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

// Kernel 1b: Reduce across FMB chunks
__global__ void render_kernel_fmb_reduce(TempBufferView<MemoryLocation::DEVICE> temp_buffers,
                                         uint32_t num_fmb_chunks, const Intrinsics& intr) {
    // Calculate pixel coordinates
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;

    // Early exit if out of bounds
    if (px >= intr.width || py >= intr.height)
        return;

    const int img_row = intr.height - py - 1;
    const uint32_t pixel_offset = img_row * intr.width + px;

    // Load values from all FMB chunks for this pixel
    float depth_numer = 0.0F;
    float depth_denom = 0.0F;
    float conf_tmp = 0.0F;

    for (uint32_t chunk = 0; chunk < num_fmb_chunks; ++chunk) {
        // Get pointers for each buffer type for this chunk
        float* depth_numer_ptr = temp_buffers.get_buffer_ptr(chunk, 0);
        float* depth_denom_ptr = temp_buffers.get_buffer_ptr(chunk, 1);
        float* conf_tmp_ptr = temp_buffers.get_buffer_ptr(chunk, 2);

        depth_numer += depth_numer_ptr[pixel_offset];
        depth_denom += depth_denom_ptr[pixel_offset];
        conf_tmp += conf_tmp_ptr[pixel_offset];
    }

    // Write reduced values to first 3 buffers (overwrite chunk 0)
    float* reduced_depth_numer = temp_buffers.get_buffer_ptr(0, 0);
    float* reduced_depth_denom = temp_buffers.get_buffer_ptr(0, 1);
    float* reduced_conf_tmp = temp_buffers.get_buffer_ptr(0, 2);

    reduced_depth_numer[pixel_offset] = depth_numer;
    reduced_depth_denom[pixel_offset] = depth_denom;
    reduced_conf_tmp[pixel_offset] = conf_tmp;
}

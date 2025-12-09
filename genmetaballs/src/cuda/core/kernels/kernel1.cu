#include <cstdint>
#include <cuda_runtime.h>

#include "core/camera.cuh"
#include "core/temp_buffer.cuh"
#include "core/utils.cuh"
#include "core/kernels/kernel1.cuh"

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


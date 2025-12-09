#include <cstdint>
#include <cuda_runtime.h>

#include "core/camera.cuh"
#include "core/kernels/kernel1.cuh"
#include "core/temp_buffer.cuh"
#include "core/utils.cuh"

// Simple sum operation for parallel reduction
struct SumOp {
    using Data = float;

    static __device__ __forceinline__ Data identity() {
        return 0.0f;
    }

    static __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }
};

// Kernel 1b: Reduce across FMB chunks using parallel sum
// CRITICAL: Must use same pixel tiling as kernel 1a to read from correct pixels
__global__ void render_kernel_fmb_reduce(TempBufferView<MemoryLocation::DEVICE> temp_buffers,
                                         uint32_t num_fmb_chunks, const Intrinsics& intr) {
    // Use same pixel coordinate calculation as kernel 1a (pixel tiling)
    dim3 thread_idx_2d(threadIdx.x, threadIdx.y, 0);
    dim3 block_idx_2d(blockIdx.x, blockIdx.y, 0);
    dim3 block_dim_2d(blockDim.x, blockDim.y, 1);
    dim3 grid_dim_2d(gridDim.x, gridDim.y, 1);

    auto pixel_coords =
        get_pixel_coords_inline(thread_idx_2d, block_idx_2d, block_dim_2d, grid_dim_2d, intr);

    const int chunk_idx = threadIdx.z; // Chunk index from z-dimension

    if (chunk_idx >= (int)num_fmb_chunks)
        return;

    // Calculate pixel tile dimensions (same as kernel 1a)
    const int num_pixels_x = int_ceil_div(intr.width, gridDim.x * blockDim.x);
    const int num_pixels_y = int_ceil_div(intr.height, gridDim.y * blockDim.y);
    const int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * num_pixels_x;
    const int start_y = (blockIdx.y * blockDim.y + threadIdx.y) * num_pixels_y;

    // Calculate block-level tile dimensions for shared memory
    const int block_start_x = blockIdx.x * blockDim.x * num_pixels_x;
    const int block_start_y = blockIdx.y * blockDim.y * num_pixels_y;
    const int max_pixels_in_block_tile = blockDim.x * num_pixels_x * blockDim.y * num_pixels_y;

    // Allocate shared memory ONCE (not in loop): [max_pixels_in_block_tile * num_chunks] for each
    // buffer
    extern __shared__ __align__(16) char shmem_raw[];
    float* shmem_numer = reinterpret_cast<float*>(shmem_raw);
    float* shmem_denom = shmem_numer + max_pixels_in_block_tile * blockDim.z;
    float* shmem_conf = shmem_denom + max_pixels_in_block_tile * blockDim.z;

    // Process each pixel in the tile (same as kernel 1a)
    // We need to iterate over ALL pixels in the block's tile to keep threads in sync
    for (int local_py = 0; local_py < num_pixels_y; local_py++) {
        for (int local_px = 0; local_px < num_pixels_x; local_px++) {
            const int px = start_x + local_px;
            const int py = start_y + local_py;

            // Check if this pixel is valid for this thread (in pixel_coords)
            bool pixel_valid = (px < intr.width && py < intr.height);
            if (pixel_valid) {
                for (const auto [px_check, py_check] : pixel_coords) {
                    if (px_check == px && py_check == py) {
                        pixel_valid = true;
                        break;
                    }
                }
            }

            // Calculate shared memory index for this pixel (block-level)
            const int block_local_px = px - block_start_x;
            const int block_local_py = py - block_start_y;
            const int pixel_idx_in_block_tile =
                block_local_py * (blockDim.x * num_pixels_x) + block_local_px;
            const int shmem_idx = pixel_idx_in_block_tile * blockDim.z + chunk_idx;

            if (pixel_valid) {
                const int img_row = intr.height - py - 1;
                const uint32_t pixel_offset = img_row * intr.width + px;

                // Load one chunk's value for this pixel
                float depth_numer = temp_buffers.get_buffer_ptr(chunk_idx, 0)[pixel_offset];
                float depth_denom = temp_buffers.get_buffer_ptr(chunk_idx, 1)[pixel_offset];
                float conf_tmp = temp_buffers.get_buffer_ptr(chunk_idx, 2)[pixel_offset];

                // Store values in shared memory
                shmem_numer[shmem_idx] = depth_numer;
                shmem_denom[shmem_idx] = depth_denom;
                shmem_conf[shmem_idx] = conf_tmp;
            } else {
                // Invalid pixel for this thread - write zeros (won't affect reduction if other
                // threads have valid data)
                shmem_numer[shmem_idx] = 0.0f;
                shmem_denom[shmem_idx] = 0.0f;
                shmem_conf[shmem_idx] = 0.0f;
            }
            __syncthreads(); // All threads have stored their chunk's value for this pixel

            // Parallel sum across chunks for this pixel using reduction pattern
            float val_numer = shmem_numer[shmem_idx];
            float val_denom = shmem_denom[shmem_idx];
            float val_conf = shmem_conf[shmem_idx];

            // Reduction phase: build sum tree
            for (uint32_t delta = 1; delta < blockDim.z; delta <<= 1) {
                if (chunk_idx >= delta) {
                    float partial_numer =
                        shmem_numer[pixel_idx_in_block_tile * blockDim.z + chunk_idx - delta];
                    float partial_denom =
                        shmem_denom[pixel_idx_in_block_tile * blockDim.z + chunk_idx - delta];
                    float partial_conf =
                        shmem_conf[pixel_idx_in_block_tile * blockDim.z + chunk_idx - delta];

                    val_numer = SumOp::combine(partial_numer, val_numer);
                    val_denom = SumOp::combine(partial_denom, val_denom);
                    val_conf = SumOp::combine(partial_conf, val_conf);
                }
                __syncthreads();

                if (chunk_idx >= delta) {
                    shmem_numer[shmem_idx] = val_numer;
                    shmem_denom[shmem_idx] = val_denom;
                    shmem_conf[shmem_idx] = val_conf;
                }
                __syncthreads();
            }

            // After reduction, thread with highest chunk_idx has the total sum
            // Write it to chunk 0's buffer (overwriting it) only if pixel is valid
            if (chunk_idx == blockDim.z - 1 && pixel_valid) {
                const int img_row = intr.height - py - 1;
                const uint32_t pixel_offset = img_row * intr.width + px;
                temp_buffers.get_buffer_ptr(0, 0)[pixel_offset] = val_numer;
                temp_buffers.get_buffer_ptr(0, 1)[pixel_offset] = val_denom;
                temp_buffers.get_buffer_ptr(0, 2)[pixel_offset] = val_conf;
            }
            __syncthreads(); // Sync before processing next pixel
        }
    }
}

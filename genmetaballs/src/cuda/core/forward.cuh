#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "fmb.cuh"
#include "geometry.cuh"
#include "image.cuh"
#include "temp_buffer.cuh"
#include "utils.cuh"

// Include kernel implementations
#include "kernels/kernel0.cuh"
#include "kernels/kernel1.cuh"

// ============================================================================
// MAIN RENDER FUNCTION (switches between kernels)
// ============================================================================
// Kernel IDs:
// 0 = Original slow working kernel (for verification)
// 1 = 3-kernel FMB chunk parallelization
// 2+ = Reserved for future optimizations

// Timing structure for individual kernel timings (in microseconds)
struct KernelTimings {
    float kernel_1a_us = 0.0f;
    float kernel_1b_us = 0.0f;
    float kernel_1c_us = 0.0f;
    float total_us = 0.0f;
};

template <typename Getter, typename Intersector, typename Blender, typename Confidence>
void render_fmbs(const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender,
                 const Confidence& confidence, const Intrinsics& intr, const Pose& extr,
                 ImageView<MemoryLocation::DEVICE> img, const dim3 grid_size, const dim3 block_size,
                 int kernel_id, TempBufferView<MemoryLocation::DEVICE>* temp_buffers = nullptr,
                 uint32_t num_fmb_chunks = 8, KernelTimings* timings = nullptr) {
    switch (kernel_id) {
        case 0: {
            // Original slow working kernel (for verification)
            render_kernel_original<Getter, Intersector, Blender, Confidence>
                <<<grid_size, block_size>>>(fmbs, blender, confidence, intr, extr, img);
            break;
        }
        case 1: {
            // 3-kernel FMB chunk parallelization
            if (temp_buffers == nullptr) {
                // Fallback to original if temp_buffers not provided
                render_kernel_original<Getter, Intersector, Blender, Confidence>
                    <<<grid_size, block_size>>>(fmbs, blender, confidence, intr, extr, img);
                break;
            }

            const uint32_t num_fmbs = fmbs.size();
            const uint32_t fmb_chunk_size = (num_fmbs + num_fmb_chunks - 1) / num_fmb_chunks;

            // Kernel 1a: Process FMB chunks
            // Use 3D block size: z-dimension = num_fmb_chunks
            // CUDA limit: max 1024 threads per block
            // Ensure block_size.x * block_size.y * num_fmb_chunks <= 1024
            constexpr uint32_t max_threads_per_block = 1024;
            uint32_t required_threads = block_size.x * block_size.y * num_fmb_chunks;

            dim3 block_size_3d;
            dim3 grid_size_3d = grid_size;

            if (required_threads <= max_threads_per_block) {
                // All chunks fit in one block
                block_size_3d = dim3(block_size.x, block_size.y, num_fmb_chunks);
            } else {
                // Need to reduce x/y to fit all chunks
                // Calculate max x*y that fits: x*y*z <= 1024, so x*y <= 1024/z
                uint32_t max_xy = max_threads_per_block / num_fmb_chunks;
                // Find x and y that are close to original but fit
                uint32_t xy = block_size.x * block_size.y;
                if (xy > max_xy) {
                    // Reduce proportionally
                    float scale = sqrtf((float)max_xy / (float)xy);
                    block_size_3d.x = (uint32_t)(block_size.x * scale);
                    block_size_3d.y = (uint32_t)(block_size.y * scale);
                    // Ensure at least 1
                    if (block_size_3d.x == 0)
                        block_size_3d.x = 1;
                    if (block_size_3d.y == 0)
                        block_size_3d.y = 1;
                    // Adjust grid to cover same area
                    grid_size_3d.x =
                        (grid_size.x * block_size.x + block_size_3d.x - 1) / block_size_3d.x;
                    grid_size_3d.y =
                        (grid_size.y * block_size.y + block_size_3d.y - 1) / block_size_3d.y;
                } else {
                    block_size_3d.x = block_size.x;
                    block_size_3d.y = block_size.y;
                }
                block_size_3d.z = num_fmb_chunks;
            }

            // Create CUDA events for timing
            cudaEvent_t start_1a, stop_1a, start_1b, stop_1b, start_1c, stop_1c, start_total,
                stop_total;
            if (timings != nullptr) {
                cudaEventCreate(&start_1a);
                cudaEventCreate(&stop_1a);
                cudaEventCreate(&start_1b);
                cudaEventCreate(&stop_1b);
                cudaEventCreate(&start_1c);
                cudaEventCreate(&stop_1c);
                cudaEventCreate(&start_total);
                cudaEventCreate(&stop_total);
                cudaEventRecord(start_total);
            }

            // Kernel 1a: Process FMB chunks
            if (timings != nullptr)
                cudaEventRecord(start_1a);
            // Calculate shared memory for FMB cache
            // CachedFMB: ~48 bytes (with alignment), cache ALL FMBs
            const size_t total_fmbs = fmbs.size();
            // i hardcoded this to FMB datastruc
            const size_t fmb_cache_shmem =
                total_fmbs *
                48; // CachedFMB size with alignment (48 bytes per FMB with all the diff items
                    // inside -- might have done the math wrong on this, but if i am not wrong, this
                    // will hold up to 1000 metaballs before extending the shmem to 100kb)

            render_kernel_fmb_chunk_processing<Getter, Intersector, Blender>
                <<<grid_size_3d, block_size_3d, fmb_cache_shmem>>>(
                    fmbs, blender, intr, extr, *temp_buffers, num_fmb_chunks, fmb_chunk_size);
            if (timings != nullptr) {
                cudaEventRecord(stop_1a);
                cudaEventSynchronize(stop_1a);
                float ms;
                cudaEventElapsedTime(&ms, start_1a, stop_1a);
                timings->kernel_1a_us = ms * 1000.0f; // Convert ms to microseconds
            } else {
                cudaDeviceSynchronize();
            }

            // Kernel 1b: Reduce across chunks using parallel sum
            // Allocate shared memory: 3 buffers * (max_pixels_in_block_tile * num_chunks)
            // Calculate max pixels in block tile (same as get_pixel_coords)
            const int num_pixels_x = int_ceil_div(intr.width, grid_size_3d.x * block_size_3d.x);
            const int num_pixels_y = int_ceil_div(intr.height, grid_size_3d.y * block_size_3d.y);
            const int max_pixels_in_block_tile =
                block_size_3d.x * num_pixels_x * block_size_3d.y * num_pixels_y;
            size_t shmem_size = 3 * max_pixels_in_block_tile * block_size_3d.z * sizeof(float);
            if (timings != nullptr)
                cudaEventRecord(start_1b);
            render_kernel_fmb_reduce<<<grid_size_3d, block_size_3d, shmem_size>>>(
                *temp_buffers, num_fmb_chunks, intr);
            if (timings != nullptr) {
                cudaEventRecord(stop_1b);
                cudaEventSynchronize(stop_1b);
                float ms;
                cudaEventElapsedTime(&ms, start_1b, stop_1b);
                timings->kernel_1b_us = ms * 1000.0f; // Convert ms to microseconds
            } else {
                cudaDeviceSynchronize();
            }

            // Kernel 1c: Finalize
            // Must use same grid/block dimensions as kernel 1a and 1b (3D) for pixel tiling to
            // match But over here we dont need to use the z-dimension (since this is post-reduce),
            // so we use 2D block and grid.
            dim3 block_size_2d(block_size_3d.x, block_size_3d.y, 1);
            dim3 grid_size_2d(grid_size_3d.x, grid_size_3d.y, 1);
            if (timings != nullptr)
                cudaEventRecord(start_1c);
            render_kernel_fmb_finalize<Confidence>
                <<<grid_size_2d, block_size_2d>>>(*temp_buffers, confidence, intr, img);
            if (timings != nullptr) {
                cudaEventRecord(stop_1c);
                cudaEventRecord(stop_total);
                cudaEventSynchronize(stop_total);
                float ms;
                cudaEventElapsedTime(&ms, start_1c, stop_1c);
                timings->kernel_1c_us = ms * 1000.0f; // Convert ms to microseconds
                cudaEventElapsedTime(&ms, start_total, stop_total);
                timings->total_us = ms * 1000.0f; // Convert ms to microseconds
                // Cleanup events
                cudaEventDestroy(start_1a);
                cudaEventDestroy(stop_1a);
                cudaEventDestroy(start_1b);
                cudaEventDestroy(stop_1b);
                cudaEventDestroy(start_1c);
                cudaEventDestroy(stop_1c);
                cudaEventDestroy(start_total);
                cudaEventDestroy(stop_total);
            }
            break;
        }
        default:
            // Fallback to original kernel for unknown kernel IDs
            render_kernel_original<Getter, Intersector, Blender, Confidence>
                <<<grid_size, block_size>>>(fmbs, blender, confidence, intr, extr, img);
            break;
    }
}

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
template <typename Getter, typename Intersector, typename Blender, typename Confidence>
void render_fmbs(const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender,
                 const Confidence& confidence, const Intrinsics& intr, const Pose& extr,
                 ImageView<MemoryLocation::DEVICE> img, const dim3 grid_size, const dim3 block_size,
                 int kernel_id, TempBufferView<MemoryLocation::DEVICE>* temp_buffers = nullptr,
                 uint32_t num_fmb_chunks = 8) {
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
            uint32_t max_threads_per_block = 1024;
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

            render_kernel_fmb_chunk_processing<Getter, Intersector, Blender>
                <<<grid_size_3d, block_size_3d>>>(fmbs, blender, intr, extr, *temp_buffers,
                                                  num_fmb_chunks, fmb_chunk_size);
            // Synchronize to ensure kernel 1a completes before kernel 1b
            cudaDeviceSynchronize();

            // Kernel 1b: Reduce across chunks
            render_kernel_fmb_reduce<<<grid_size, block_size>>>(*temp_buffers, num_fmb_chunks,
                                                                intr);
            // Synchronize to ensure kernel 1b completes before kernel 1c
            cudaDeviceSynchronize();

            // Kernel 1c: Finalize
            render_kernel_fmb_finalize<Confidence>
                <<<grid_size, block_size>>>(*temp_buffers, confidence, intr, img);
            break;
        }
        default:
            // Fallback to original kernel for unknown kernel IDs
            render_kernel_original<Getter, Intersector, Blender, Confidence>
                <<<grid_size, block_size>>>(fmbs, blender, confidence, intr, extr, img);
            break;
    }
}

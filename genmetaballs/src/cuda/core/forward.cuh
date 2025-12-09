#pragma once

#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "fmb.cuh"
#include "geometry.cuh"
#include "image.cuh"
#include "temp_buffer.cuh"
#include "utils.cuh"

// Optimized configuration
constexpr int THREADS_PER_BLOCK_1D = 256; // 8 warps per block
constexpr int WARP_SIZE = 32;             // CUDA warp size
constexpr int MAX_FMBS_IN_SHARED = 128;   // Max FMBs to store in shared memory

CUDA_CALLABLE PixelCoordRange get_pixel_coords(const dim3 thread_idx, const dim3 block_idx,
                                               const dim3 block_dim, const dim3 grid_dim,
                                               const Intrinsics& intr);

// ============================================================================
// ORIGINAL KERNEL (for verification)
// ============================================================================
template <typename Getter, typename Intersector, typename Blender, typename Confidence>
__global__ void render_kernel_original(const FMBScene<MemoryLocation::DEVICE>& fmbs,
                                       const Blender& blender, const Confidence& confidence,
                                       const Intrinsics& intr, const Pose& extr,
                                       ImageView<MemoryLocation::DEVICE> img) {
    auto pixel_coords = get_pixel_coords(threadIdx, blockIdx, blockDim, gridDim, intr);
    auto fmb_getter = Getter(fmbs, extr);

    for (const auto [px, py] : pixel_coords) {
        float depth_denom = 0.0f, depth_numer = 0.0f, conf_tmp = 0.0f;
        auto ray = intr.get_ray_direction(px, py);
        for (const auto& [fmb, lambda] : fmb_getter.get_metaballs(ray)) {
            // d: intersection point along the ray
            // q: square of Mahalanobis distance at intersection point
            const auto& [d, q] = Intersector::intersect(fmb, ray, extr);
            auto tmp = -0.5f * q + lambda;
            // the next check is needed to match the reference implementation
            // even though it is not in the paper.
            auto w_tilde = d > 0 ? blender.blend(tmp, d) : 1e-20f;
            conf_tmp += exp(tmp); // numerically unstable. use logsumexp
            depth_numer += d * w_tilde;
            depth_denom += w_tilde;
        }
        // the indexing is done this way because the underlying array2ds use
        // ij indexing, whereas the pixels uses xy indexing
        img.confidence[intr.height - py - 1][px] = confidence.get_confidence(conf_tmp);
        img.depth[intr.height - py - 1][px] = depth_numer / depth_denom;
    }
}

// ============================================================================
// KERNEL 1: 3-KERNEL FMB CHUNK PARALLELIZATION
// ============================================================================

// Kernel 1a: Process FMB chunks in parallel
// Parallelizes over pixels (2D grid) AND FMB chunks (3rd dimension in block)
template <typename Getter, typename Intersector, typename Blender>
__global__ void render_kernel_fmb_chunk_processing(
    const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender, const Intrinsics& intr,
    const Pose& extr, TempBufferView<MemoryLocation::DEVICE> temp_buffers, uint32_t num_fmb_chunks,
    uint32_t fmb_chunk_size) {
    // Calculate pixel coordinates from 2D grid
    // Note: blockDim.x and blockDim.y are still the x and y dimensions even with 3D block
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    const int fmb_chunk_idx = threadIdx.z; // FMB chunk index from 3rd block dimension

    // Early exit if out of bounds
    if (px >= (int)intr.width || py >= (int)intr.height || fmb_chunk_idx >= (int)num_fmb_chunks)
        return;

    const Vec3D ray = intr.get_ray_direction(px, py);
    auto fmb_getter = Getter(fmbs, extr);
    const auto& fmb_scene = fmb_getter.get_metaballs(ray);
    const int num_fmbs = fmb_scene.size();

    // Process FMBs for this chunk
    float depth_numer = 0.0f;
    float depth_denom = 0.0f;
    float conf_tmp = 0.0f;

    const int start_fmb_idx = fmb_chunk_idx * fmb_chunk_size;
    const int end_fmb_idx =
        (start_fmb_idx + fmb_chunk_size < num_fmbs) ? (start_fmb_idx + fmb_chunk_size) : num_fmbs;

    for (int fmb_idx = start_fmb_idx; fmb_idx < end_fmb_idx; ++fmb_idx) {
        const auto& [fmb, lambda] = fmb_scene[fmb_idx];

        // Match original computation exactly
        const auto& [d, q] = Intersector::intersect(fmb, ray, extr);
        auto tmp = -0.5f * q + lambda;
        // the next check is needed to match the reference implementation
        // even though it is not in the paper.
        auto w_tilde = d > 0 ? blender.blend(tmp, d) : 1e-20f;
        conf_tmp += exp(tmp); // Match original: use exp (not expf)
        depth_numer += d * w_tilde;
        depth_denom += w_tilde;
    }

    // Write to temporary buffers
    // type: 0=depth_numer, 1=depth_denom, 2=conf_tmp
    const int img_row = intr.height - py - 1;

    // Calculate buffer offsets directly
    const uint32_t pixel_offset = img_row * intr.width + px;

    // Get pointers for each buffer type
    float* depth_numer_ptr = temp_buffers.get_buffer_ptr(fmb_chunk_idx, 0);
    float* depth_denom_ptr = temp_buffers.get_buffer_ptr(fmb_chunk_idx, 1);
    float* conf_tmp_ptr = temp_buffers.get_buffer_ptr(fmb_chunk_idx, 2);

    // Write directly using pointer arithmetic
    // Note: Even if no FMBs are processed, we write the accumulated values (which may be 0)
    depth_numer_ptr[pixel_offset] = depth_numer;
    depth_denom_ptr[pixel_offset] = depth_denom;
    conf_tmp_ptr[pixel_offset] = conf_tmp;
}

// Kernel 1b: Reduce across FMB chunks using CUB BlockReduce
// Declared here, defined in forward.cu to avoid multiple definition errors
__global__ void render_kernel_fmb_reduce(TempBufferView<MemoryLocation::DEVICE> temp_buffers,
                                         uint32_t num_fmb_chunks, const Intrinsics& intr);

// Kernel 1c: Finalize confidence and depth
template <typename Confidence>
__global__ void render_kernel_fmb_finalize(TempBufferView<MemoryLocation::DEVICE> temp_buffers,
                                           const Confidence& confidence, const Intrinsics& intr,
                                           ImageView<MemoryLocation::DEVICE> img) {
    // Calculate pixel coordinates
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;

    // Early exit if out of bounds
    if (px >= intr.width || py >= intr.height)
        return;

    const int img_row = intr.height - py - 1;
    const uint32_t pixel_offset = img_row * intr.width + px;

    // Read reduced values from first 3 buffers (chunk 0)
    float* reduced_depth_numer = temp_buffers.get_buffer_ptr(0, 0);
    float* reduced_depth_denom = temp_buffers.get_buffer_ptr(0, 1);
    float* reduced_conf_tmp = temp_buffers.get_buffer_ptr(0, 2);

    float depth_numer = reduced_depth_numer[pixel_offset];
    float depth_denom = reduced_depth_denom[pixel_offset];
    float conf_tmp = reduced_conf_tmp[pixel_offset];

    // Apply confidence and compute final depth
    // Match original: no safety check, just divide (original doesn't check)
    // But ensure we don't divide by zero to avoid NaN
    img.confidence[img_row][px] = confidence.get_confidence(conf_tmp);
    img.depth[img_row][px] = (depth_denom > 0.0f) ? (depth_numer / depth_denom) : 0.0f;
}

// ============================================================================
// MAIN RENDER FUNCTION (switches between original and optimized)
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

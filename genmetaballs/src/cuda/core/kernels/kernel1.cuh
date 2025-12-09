#pragma once

#include "core/camera.cuh"
#include "core/fmb.cuh"
#include "core/geometry.cuh"
#include "core/image.cuh"
#include "core/temp_buffer.cuh"
#include "core/utils.cuh"

// ============================================================================
// KERNEL 1: 3-KERNEL FMB CHUNK PARALLELIZATION
// ============================================================================

// Kernel 1a: Process FMB chunks in parallel
// Parallelizes over pixels (2D grid) AND FMB chunks (3rd dimension in block)
// Uses coalesced memory access pattern: threads in same warp access consecutive FMB indices
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

    // Process FMBs for this chunk with COALESCED access pattern
    // Instead of: thread 0 accesses FMBs 0-9, thread 1 accesses FMBs 10-19, etc.
    // We do: thread 0 accesses FMBs 0,4,8,12..., thread 1 accesses FMBs 1,5,9,13..., etc.
    // This ensures threads in the same warp access consecutive FMB indices (coalesced)
    float depth_numer = 0.0f;
    float depth_denom = 0.0f;
    float conf_tmp = 0.0f;

    // Coalesced access: each thread processes FMBs with stride = num_fmb_chunks
    // Thread 0 (chunk 0): FMBs 0, 4, 8, 12, ...
    // Thread 1 (chunk 1): FMBs 1, 5, 9, 13, ...
    // Thread 2 (chunk 2): FMBs 2, 6, 10, 14, ...
    // Thread 3 (chunk 3): FMBs 3, 7, 11, 15, ...
    // This way, when all threads access their first FMB, they access 0,1,2,3 (coalesced!)
    const int start_fmb_idx = fmb_chunk_idx; // Start at chunk index, not chunk * chunk_size
    const int stride = num_fmb_chunks;       // Stride by number of chunks

    for (int fmb_idx = start_fmb_idx; fmb_idx < num_fmbs; fmb_idx += stride) {
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

// Kernel 1b: Reduce across FMB chunks
// Declared here, defined in kernel1.cu to avoid multiple definition errors
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

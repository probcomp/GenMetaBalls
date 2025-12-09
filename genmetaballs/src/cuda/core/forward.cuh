#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "fmb.cuh"
#include "geometry.cuh"
#include "image.cuh"
#include "utils.cuh"

// Note: Kernel choice is now a runtime parameter, not compile-time

// Original configuration
constexpr auto NUM_BLOCKS = dim3(4, 4);
constexpr auto THREADS_PER_BLOCK = dim3(16, 16);

// Optimized configuration
constexpr int THREADS_PER_BLOCK_1D = 256;  // 8 warps per block
constexpr int WARP_SIZE = 32;               // CUDA warp size
constexpr int MAX_FMBS_IN_SHARED = 128;     // Max FMBs to store in shared memory

CUDA_CALLABLE PixelCoordRange get_pixel_coords(const dim3 thread_idx, const dim3 block_idx,
                                               const dim3 block_dim, const dim3 grid_dim,
                                               const Intrinsics& intr);

// ============================================================================
// ORIGINAL KERNEL (for verification)
// ============================================================================
template <typename Getter, typename Intersector, typename Blender, typename Confidence>
__global__ void render_kernel_original(const FMBScene<MemoryLocation::DEVICE>& fmbs, 
                                       const Blender& blender,
                                       const Confidence& confidence, 
                                       const Intrinsics& intr,
                                       const Pose& extr, 
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
// OPTIMIZED KERNEL (with FMB parallelization)
// ============================================================================

// Note: compute_fmb_contribution removed - using inline computation to match original exactly

// Optimized kernel: TRUE parallelization over FMBs
// Strategy:
// 1. Each warp processes ONE pixel
// 2. Threads in the warp process DIFFERENT FMBs in parallel
// 3. Warp-level reduction combines all FMB contributions
// 4. This parallelizes both intersection (per-FMB) and accumulation (across FMBs)
template <typename Getter, typename Intersector, typename Blender, typename Confidence>
__global__ void render_kernel_fmb_parallel(const FMBScene<MemoryLocation::DEVICE>& fmbs, 
                                            const Blender& blender,
                                            const Confidence& confidence, 
                                            const Intrinsics& intr,
                                            const Pose& extr, 
                                            ImageView<MemoryLocation::DEVICE> img) {
    // Calculate which pixel this warp processes
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int total_pixels = intr.width * intr.height;
    
    // Early exit if warp is beyond image bounds
    if (warp_id >= total_pixels) return;
    
    // All threads in warp work on the same pixel
    const int px = warp_id % intr.width;
    const int py = warp_id / intr.width;
    const Vec3D ray = intr.get_ray_direction(px, py);
    
    // Use Getter like original (matches original exactly)
    auto fmb_getter = Getter(fmbs, extr);
    const auto& fmb_scene = fmb_getter.get_metaballs(ray);
    const int num_fmbs_to_process = fmb_scene.size();
    
    // PARALLELIZE OVER FMBs: Each thread in the warp processes a subset of FMBs
    // Thread lane_id processes FMBs at indices: lane_id, lane_id + WARP_SIZE, lane_id + 2*WARP_SIZE, ...
    // This matches the original logic exactly, just parallelized across warp
    float depth_numer = 0.0f;
    float depth_denom = 0.0f;
    float conf_tmp = 0.0f;
    
    for (int fmb_idx = lane_id; fmb_idx < num_fmbs_to_process; fmb_idx += WARP_SIZE) {
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
    
    // WARP-LEVEL REDUCTION: Combine contributions from all threads in the warp
    // Use warp shuffle for efficient reduction (faster than shared memory)
    // All threads in warp participate (even if they didn't process FMBs, they have 0.0f)
    unsigned int mask = 0xFFFFFFFF; // All 32 threads in warp
    
    // Butterfly reduction pattern
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        depth_numer += __shfl_down_sync(mask, depth_numer, offset);
        depth_denom += __shfl_down_sync(mask, depth_denom, offset);
        conf_tmp += __shfl_down_sync(mask, conf_tmp, offset);
    }
    
    // Lane 0 writes the final result for this pixel
    // Match original: no safety check, just divide (original doesn't check)
    if (lane_id == 0) {
        const int img_row = intr.height - py - 1;
        img.confidence[img_row][px] = confidence.get_confidence(conf_tmp);
        img.depth[img_row][px] = depth_numer / depth_denom;
    }
}

// ============================================================================
// MAIN RENDER FUNCTION (switches between original and optimized)
// ============================================================================
template <typename Getter, typename Intersector, typename Blender, typename Confidence>
void render_fmbs(const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender,
                 const Confidence& confidence, const Intrinsics& intr, const Pose& extr,
                 ImageView<MemoryLocation::DEVICE> img, bool use_optimized = true) {
    if (use_optimized) {
        // Use optimized FMB-parallelized kernel
        const int total_pixels = intr.width * intr.height;
        const int num_warps_needed = (total_pixels + WARP_SIZE - 1) / WARP_SIZE;
        const int num_threads_needed = num_warps_needed * WARP_SIZE;
        const int num_blocks = (num_threads_needed + THREADS_PER_BLOCK_1D - 1) / THREADS_PER_BLOCK_1D;
        
        render_kernel_fmb_parallel<Getter, Intersector, Blender, Confidence>
            <<<num_blocks, THREADS_PER_BLOCK_1D>>>(fmbs, blender, confidence, intr, extr, img);
    } else {
        // Use original kernel (for verification)
        render_kernel_original<Getter, Intersector, Blender, Confidence>
            <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(fmbs, blender, confidence, intr, extr, img);
    }
}

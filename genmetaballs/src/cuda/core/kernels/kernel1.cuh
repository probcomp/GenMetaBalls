#pragma once

#include "core/camera.cuh"
#include "core/fmb.cuh"
#include "core/geometry.cuh"
#include "core/image.cuh"
#include "core/temp_buffer.cuh"
#include "core/utils.cuh"

// Inline implementation of get_pixel_coords to avoid multiple definition errors
// (get_pixel_coords is declared in camera.cuh and defined in forward.cu)
CUDA_CALLABLE __forceinline__ PixelCoordRange get_pixel_coords_inline(const dim3 thread_idx,
                                                                      const dim3 block_idx,
                                                                      const dim3 block_dim,
                                                                      const dim3 grid_dim,
                                                                      const Intrinsics& intr) {
    // compute the number of pixels each thread should process
    const auto num_pixels_x = int_ceil_div(intr.width, grid_dim.x * block_dim.x);
    const auto num_pixels_y = int_ceil_div(intr.height, grid_dim.y * block_dim.y);
    const auto start_x = (block_idx.x * block_dim.x + thread_idx.x) * num_pixels_x;
    const auto start_y = (block_idx.y * block_dim.y + thread_idx.y) * num_pixels_y;
    const auto end_x =
        (start_x + num_pixels_x < intr.width) ? (start_x + num_pixels_x) : intr.width;
    const auto end_y =
        (start_y + num_pixels_y < intr.height) ? (start_y + num_pixels_y) : intr.height;
    return PixelCoordRange{
        .px_start = start_x, .px_end = end_x, .py_start = start_y, .py_end = end_y};
}

// ============================================================================
// KERNEL 1: 3-KERNEL FMB CHUNK PARALLELIZATION
// ============================================================================

// Kernel 1a: Process FMB chunks in parallel (OPTIMIZED with pixel tiling)
// Parallelizes over pixels (2D grid with tiling) AND FMB chunks (3rd dimension in block)
// Uses pixel coordinate tiling like kernel 0 to improve occupancy and reduce thread overhead
// Optimizations:
// 1. Pixel tiling: each thread processes multiple pixels (like kernel 0)
// 2. Pre-compute FMB getter once per thread
// 3. Cache buffer pointers per chunk
// 4. Coalesced memory access pattern for FMBs
template <typename Getter, typename Intersector, typename Blender>
__global__ void render_kernel_fmb_chunk_processing(
    const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender, const Intrinsics& intr,
    const Pose& extr, TempBufferView<MemoryLocation::DEVICE> temp_buffers, uint32_t num_fmb_chunks,
    uint32_t fmb_chunk_size) {
    // Get pixel coordinate range for this thread (like kernel 0)
    // Note: blockDim.x and blockDim.y are still the x and y dimensions even with 3D block
    // We need to create a 2D block_dim and grid_dim for get_pixel_coords
    dim3 thread_idx_2d(threadIdx.x, threadIdx.y, 0);
    dim3 block_idx_2d(blockIdx.x, blockIdx.y, 0);
    dim3 block_dim_2d(blockDim.x, blockDim.y, 1);
    // Use the actual grid dimensions (which account for 3D block)
    dim3 grid_dim_2d(gridDim.x, gridDim.y, 1);

    auto pixel_coords =
        get_pixel_coords_inline(thread_idx_2d, block_idx_2d, block_dim_2d, grid_dim_2d, intr);

    const int fmb_chunk_idx = threadIdx.z; // FMB chunk index from 3rd block dimension

    // Early exit if out of bounds for chunk index
    if (fmb_chunk_idx >= (int)num_fmb_chunks)
        return;

    // Pre-compute FMB getter once per thread (reused for all pixels)
    auto fmb_getter = Getter(fmbs, extr);

    // Get FMB scene once per thread (not per pixel!) since get_metaballs() returns all FMBs
    // Use a dummy ray since AllGetter ignores it anyway
    const Vec3D dummy_ray(0.0f, 0.0f, 1.0f);
    const auto& fmb_scene = fmb_getter.get_metaballs(dummy_ray);
    const int num_fmbs = fmb_scene.size();

    // Pre-compute buffer pointers per chunk (cached, reused for all pixels)
    float* depth_numer_ptr = temp_buffers.get_buffer_ptr(fmb_chunk_idx, 0);
    float* depth_denom_ptr = temp_buffers.get_buffer_ptr(fmb_chunk_idx, 1);
    float* conf_tmp_ptr = temp_buffers.get_buffer_ptr(fmb_chunk_idx, 2);

    // Coalesced access: each thread processes FMBs with stride = num_fmb_chunks
    const int start_fmb_idx = fmb_chunk_idx;
    const int stride = num_fmb_chunks;

    // Cooperative shared memory loading of ALL FMBs
    // Each FMB is ~48 bytes (Pose + extent + log_weight)
    // Store FMB data in shared memory for faster access
    extern __shared__ __align__(16) char shmem_raw[];

    // Cache structure: store FMB pose, extent, and log_weight
    struct CachedFMB {
        float4 rot_quat;  // 16 bytes - rotation quaternion
        float3 tran;      // 12 bytes - translation (aligned to 16)
        float3 extent;    // 12 bytes - extent (aligned to 16)
        float log_weight; // 4 bytes - log weight
        // Total: ~48 bytes with alignment
    };

    CachedFMB* fmb_cache = reinterpret_cast<CachedFMB*>(shmem_raw);

    // Cooperative loading: each thread loads a subset of FMBs
    const int total_threads = blockDim.x * blockDim.y * blockDim.z;
    const int linear_thread_id =
        threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    // Each thread loads its assigned FMBs into shared memory
    for (int i = linear_thread_id; i < num_fmbs; i += total_threads) {
        const auto fmb_tuple = fmb_scene[i];
        const FMB& fmb = cuda::std::get<0>(fmb_tuple);
        const float lambda = cuda::std::get<1>(fmb_tuple);

        const Pose pose = fmb.get_pose();
        const Rotation rot = pose.get_rot();
        const float4 quat = rot.get_quat();

        fmb_cache[i].rot_quat = quat;
        fmb_cache[i].tran = pose.get_tran();
        fmb_cache[i].extent = fmb.get_extent();
        fmb_cache[i].log_weight = lambda;
    }
    __syncthreads(); // Ensure all FMBs are loaded before processing

    // Process each pixel in the tile
    for (const auto [px, py] : pixel_coords) {
        // Compute ray direction for this pixel (needed for intersection)

        // NOTE: The intersection math is written out manually
        // as opposed to using the Intersector class.

        const Vec3D ray = intr.get_ray_direction(px, py);

        // Process FMBs for this chunk with COALESCED access pattern
        float depth_numer = 0.0f;
        float depth_denom = 0.0f;
        float conf_tmp = 0.0f;

        // Optimized loop: use expf() for single precision (faster than exp())
        // Use FMBs from shared memory cache
        for (int fmb_idx = start_fmb_idx; fmb_idx < num_fmbs; fmb_idx += stride) {
            // Load FMB data from shared memory cache
            const CachedFMB& cached = fmb_cache[fmb_idx];
            const float lambda = cached.log_weight;

            // Use cached FMB data for intersection computation
            // Reconstruct rotation and pose from cached data
            const Rotation rot = Rotation::from_quat(cached.rot_quat.x, cached.rot_quat.y,
                                                     cached.rot_quat.z, cached.rot_quat.w);
            const Pose fmb_pose = Pose::from_components(rot, cached.tran);

            // Compute intersection using cached data (matches LinearIntersector::intersect exactly)
            const Vec3D v = extr.get_rot().apply(ray);

            // Compute cov_inv_apply using cached data (matches FMB::cov_inv_apply)
            // rot.apply(vecdiv(rot.inv().apply(vec), extent_))
            const Vec3D rot_inv_v = rot.inv().apply(v);
            const Vec3D vecdiv_result = {rot_inv_v.x / cached.extent.x,
                                         rot_inv_v.y / cached.extent.y,
                                         rot_inv_v.z / cached.extent.z};
            const Vec3D cov_inv_v = rot.apply(vecdiv_result);

            const Vec3D cam_tran = extr.get_tran();
            const Vec3D mean = cached.tran;
            const float d = dot(mean - cam_tran, cov_inv_v) / dot(v, cov_inv_v);

            // Compute quadratic form using cached data (matches FMB::quadratic_form)
            const Vec3D point = cam_tran + d * v;
            const Vec3D shifted_vec = point - mean;
            // cov_inv_apply(shifted_vec)
            const Vec3D rot_inv_shifted = rot.inv().apply(shifted_vec);
            const Vec3D vecdiv_shifted = {rot_inv_shifted.x / cached.extent.x,
                                          rot_inv_shifted.y / cached.extent.y,
                                          rot_inv_shifted.z / cached.extent.z};
            const Vec3D cov_inv_shifted = rot.apply(vecdiv_shifted);
            const float q = dot(shifted_vec, cov_inv_shifted);
            const float tmp = -0.5f * q + lambda;
            // the next check is needed to match the reference implementation
            // even though it is not in the paper.
            const float w_tilde = d > 0.0f ? blender.blend(tmp, d) : 1e-20f;
            const float exp_tmp = expf(tmp); // Use expf for single precision (faster than exp)

            // Accumulate values
            conf_tmp += exp_tmp;
            depth_numer += d * w_tilde;
            depth_denom += w_tilde;
        }

        // Write to temporary buffers
        const int img_row = intr.height - py - 1;
        const uint32_t pixel_offset = img_row * intr.width + px;

        depth_numer_ptr[pixel_offset] = depth_numer;
        depth_denom_ptr[pixel_offset] = depth_denom;
        conf_tmp_ptr[pixel_offset] = conf_tmp;
    }
}

// Kernel 1b: Reduce across FMB chunks
// Declared here, defined in kernel1.cu to avoid multiple definition errors
__global__ void render_kernel_fmb_reduce(TempBufferView<MemoryLocation::DEVICE> temp_buffers,
                                         uint32_t num_fmb_chunks, const Intrinsics& intr);

// Kernel 1c: Finalize confidence and depth
// CRITICAL: Must use same pixel tiling as kernel 1a and 1b to read from correct pixels
template <typename Confidence>
__global__ void render_kernel_fmb_finalize(TempBufferView<MemoryLocation::DEVICE> temp_buffers,
                                           const Confidence& confidence, const Intrinsics& intr,
                                           ImageView<MemoryLocation::DEVICE> img) {
    // Use same pixel coordinate calculation as kernel 1a and 1b (pixel tiling)
    dim3 thread_idx_2d(threadIdx.x, threadIdx.y, 0);
    dim3 block_idx_2d(blockIdx.x, blockIdx.y, 0);
    dim3 block_dim_2d(blockDim.x, blockDim.y, 1);
    dim3 grid_dim_2d(gridDim.x, gridDim.y, 1);

    auto pixel_coords =
        get_pixel_coords_inline(thread_idx_2d, block_idx_2d, block_dim_2d, grid_dim_2d, intr);

    // Read reduced values from first 3 buffers (chunk 0)
    float* reduced_depth_numer = temp_buffers.get_buffer_ptr(0, 0);
    float* reduced_depth_denom = temp_buffers.get_buffer_ptr(0, 1);
    float* reduced_conf_tmp = temp_buffers.get_buffer_ptr(0, 2);

    // Process each pixel in the tile (same order as kernel 1a and 1b)
    for (const auto [px, py] : pixel_coords) {
        const int img_row = intr.height - py - 1;
        const uint32_t pixel_offset = img_row * intr.width + px;

        float depth_numer = reduced_depth_numer[pixel_offset];
        float depth_denom = reduced_depth_denom[pixel_offset];
        float conf_tmp = reduced_conf_tmp[pixel_offset];

        // Apply confidence and compute final depth
        // Match original: no safety check, just divide (original doesn't check)
        // But ensure we don't divide by zero to avoid NaN
        img.confidence[img_row][px] = confidence.get_confidence(conf_tmp);
        img.depth[img_row][px] = (depth_denom > 0.0f) ? (depth_numer / depth_denom) : 0.0f;
    }
}

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "fmb.cuh"
#include "geometry.cuh"
#include "image.cuh"
#include "utils.cuh"

// TODO: tune this number
constexpr auto NUM_BLOCKS = dim3(4, 4);
constexpr auto THREADS_PER_BLOCK = dim3(16, 16);

CUDA_CALLABLE PixelCoordRange get_pixel_coords(const dim3 thread_idx, const dim3 block_idx,
                                               const dim3 block_dim, const dim3 grid_dim,
                                               const Intrinsics& intr);

template <typename Getter, typename Intersector, typename Blender, typename Confidence>
__global__ void render_kernel(const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender,
                              const Confidence& confidence, const Intrinsics& intr,
                              const Pose& extr, ImageView<MemoryLocation::DEVICE> img) {
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
            auto w_tilde = blender.blend(tmp, d);
            conf_tmp += exp(-tmp); // numerically unstable. use logsumexp
            depth_numer += d * w_tilde;
            depth_denom += w_tilde;
        }
        img.confidence[py][px] = confidence.get_confidence(conf_tmp);
        img.depth[py][px] = depth_numer / depth_denom;
    }
}

template <typename Getter, typename Intersector, typename Blender, typename Confidence>
void render_fmbs(const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender,
                 const Confidence& confidence, const Intrinsics& intr, const Pose& extr,
                 ImageView<MemoryLocation::DEVICE> img) {
    render_kernel<Getter, Intersector, Blender, Confidence>
        <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(fmbs, blender, confidence, intr, extr, img);
}

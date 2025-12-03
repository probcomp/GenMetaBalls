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
        float w0 = 0.0f, tf = 0.0f, sumexpd = 0.0f;
        auto ray = intr.get_ray_direction(px, py);
        for (const auto& [fmb, log_lambda] : fmb_getter.get_metaballs(ray)) {
            // t: intersection point along the ray
            // d_: square of Mahalanobis distance at intersection point
            const auto& [t, d_] = Intersector::intersect(fmb, ray, extr);
            // d2: unnormalized log distance of each Gaussian
            // d0 & d_i follows equation (2) in FMB-plus paper
            const auto d = -0.5f * d_ + log_lambda;
            auto w = blender.blend(t, d);
            sumexpd += exp(d); // numerically unstable. use logsumexp
            tf += t;
            w0 += w;
        }
        img.confidence[px][py] = confidence.get_confidence(sumexpd);
        img.depth[px][py] = tf / w0;
    }
}

template <typename Getter, typename Intersector, typename Blender, typename Confidence>
void render_fmbs(const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender,
                 const Confidence& confidence, const Intrinsics& intr, const Pose& extr,
                 ImageView<MemoryLocation::DEVICE> img) {
    render_kernel<Getter, Intersector, Blender, Confidence>
        <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(fmbs, blender, confidence, intr, extr, img);
}

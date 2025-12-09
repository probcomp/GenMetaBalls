#pragma once

#include "core/camera.cuh"
#include "core/fmb.cuh"
#include "core/geometry.cuh"
#include "core/image.cuh"
#include "core/utils.cuh"

// ============================================================================
// KERNEL 0: ORIGINAL KERNEL (for verification)
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

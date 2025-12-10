#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "forward.cuh"

class FMBSceneGradient {
private:
    Vec3D* tau_buf_;
    Mat33* rho_buf_;
    float* lambda_buf_;
    Vec3D* mu_buf_;
    Mat33* pi_buf_;

    Vec3D* tau_grad_;
    Mat33* rho_grad_;
    float* lambda_grad_;
    Vec3D* mu_grad_;
    Mat33* pi_grad_;

    const FMBScene<MemoryLocation::DEVICE>& fmbs_;
    const Intrinsics& intr_;
    const Pose& extr_;

public:
    __host__ FMBSceneGradient(const FMBScene<MemoryLocation::DEVICE>&, const Intrinsics&,
                              const Pose&);
    __host__ ~FMBSceneGradient();

    __device__ void buffer_grads(uint32_t py, uint32_t px, uint32_t fmb_idx, float d, float q);
    __device__ void accumulate_grads(uint32_t py, uint32_t px, uint32_t fmb_idx, float confidence,
                                     float expected_confidence);
};

template <typename Getter, typename Intersector, typename Blender, typename Confidence>
__global__ void fwdbwd_kernel(const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender,
                              const Confidence& confidence, const Intrinsics& intr,
                              const Pose& extr,
                              const ImageView<MemoryLocation::DEVICE> expected_img,
                              FMBSceneGradient grad, ImageView<MemoryLocation::DEVICE> output_img) {

    auto pixel_coords = get_pixel_coords(threadIdx, blockIdx, blockDim, gridDim, intr);
    auto fmb_getter = Getter(fmbs, extr);

    for (const auto [px, py] : pixel_coords) {
        float depth_denom = 0.0f, depth_numer = 0.0f, conf_tmp = 0.0f;
        auto ray = intr.get_ray_direction(px, py);
        uint32_t fmb_idx = 0;
        for (const auto& [fmb, lambda] : fmb_getter.get_metaballs(ray)) {
            // d: intersection point along the ray
            // q: square of Mahalanobis distance at intersection point
            const auto& [d, q] = Intersector::intersect(fmb, ray, extr);
            auto l = -0.5f * q + lambda;
            // the next check is needed to match the reference implementation
            // even though it is not in the paper.
            auto w_tilde = d > 0 ? blender.blend(l, d) : 1e-20f;
            // numerically unstable. use logsumexp
            conf_tmp += exp(l);
            depth_numer += d * w_tilde;
            depth_denom += w_tilde;

            grad.buffer_grads(py, px, fmb_idx++, d, q);
        }
        // the indexing is done this way because the underlying array2ds use
        // ij indexing, whereas the pixels uses xy indexing
        auto conf = confidence.get_confidence(conf_tmp);
        output_img.confidence[intr.height - py][px] = conf;
        output_img.depth[intr.height - py][px] = depth_numer / depth_denom;

        // backward
        auto expected_confidence = expected_img.confidence[intr.height - py][px];

        fmb_idx = 0;
        for (const auto& [fmb, lambda] : fmb_getter.get_metaballs(ray)) {
            grad.accumulate_grads(py, px, fmb_idx++, conf, expected_confidence);
        }
    }

    // update global grads using grad_helper
}

template <typename Getter, typename Intersector, typename Blender, typename Confidence>
void fwdbwd(const FMBScene<MemoryLocation::DEVICE>& fmbs, const Blender& blender,
            const Confidence& confidence, const Intrinsics& intr, const Pose& extr,
            const ImageView<MemoryLocation::DEVICE> expected_img, FMBSceneGradient grad,
            ImageView<MemoryLocation::DEVICE> output_img) {
    auto kernel = fwdbwd_kernel<Getter, Intersector, Blender, Confidence>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(fmbs, blender, confidence, intr, extr, expected_img,
                                              grad, output_img);
}

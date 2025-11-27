#pragma once

#include <cuda/std/tuple>

#include "fmb.cuh"
#include "geometry.cuh"

// implement equation (6) in the paper
class LinearIntersector {
public:
    /*
     * Ray should be in camera frame
     */
    CUDA_CALLABLE static cuda::std::tuple<float, float> intersect(const FMB& fmb, const Ray& ray,
                                                                  const Pose& cam_pose) {
        const auto v = cam_pose.get_rot().apply(ray.direction);
        const auto cov_inv_v = fmb.cov_inv_apply(v);
        const auto cam_tran = cam_pose.get_tran();
        const auto t = dot(fmb.get_mean() - cam_tran, cov_inv_v) / dot(v, cov_inv_v);
        return {t, fmb.quadratic_form(cam_tran + t * v)};
    }
};

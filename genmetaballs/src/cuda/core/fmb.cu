#include "fmb.cuh"
#include "geometry.cuh"
#include "utils.cuh"

CUDA_CALLABLE __forceinline__ Vec3D vecdiv(const Vec3D u, const Vec3D v) {
    return {u.x / v.x, u.y / v.y, u.z / v.z};
}

CUDA_CALLABLE Vec3D FMB::cov_inv_apply(const Vec3D vec) const {
    const auto rot = pose_.get_rot();
    // Compute R @ diag(1/extent) @ R^T @ vec
    // Step 1: R^T @ vec (apply inverse rotation)
    const auto vec_rotated = rot.inv().apply(vec);
    // Step 2: diag(1/extent) @ (R^T @ vec)
    const auto vec_scaled = vecdiv(vec_rotated, extent_);
    // Step 3: R @ (diag(1/extent) @ R^T @ vec)
    return rot.apply(vec_scaled);
}

CUDA_CALLABLE float FMB::quadratic_form(const Vec3D vec) const {
    const auto shifted_vec = vec - get_mean();
    return dot(shifted_vec, cov_inv_apply(shifted_vec));
}

#include "fmb.cuh"
#include "geometry.cuh"
#include "utils.cuh"

CUDA_CALLABLE __forceinline__ Vec3D vecdiv(const Vec3D u, const Vec3D v) {
    return {u.x / v.x, u.y / v.y, u.z / v.z};
}

// CUDA_CALLABLE Vec3D FMB::cov_inv_apply(const Vec3D vec) const {
//     const auto rot = pose_.get_rot();
//     return rot.inv().apply(vecdiv(rot.apply(vec), extent_));
// }

CUDA_CALLABLE Vec3D FMB::cov_inv_apply(const Vec3D vec) const {
    const auto rot = pose_.get_rot();
    // Wanted to add more infor here
    // Basically the order of the operation has bee swapper to look something like this:
    // R @ diag(1/extent) @ R^T @ vec however, i dont think this fixes everything
    return rot.apply(vecdiv(rot.inv().apply(vec), extent_));
}

CUDA_CALLABLE float FMB::quadratic_form(const Vec3D vec) const {
    const auto shifted_vec = vec - get_mean();
    return dot(shifted_vec, cov_inv_apply(shifted_vec));
}

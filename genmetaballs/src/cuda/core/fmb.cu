#include "fmb.cuh"
#include "geometry.cuh"
#include "utils.cuh"

CUDA_CALLABLE __forceinline__ Vec3D vecdiv(const Vec3D u, const Vec3D v) {
    return {u.x / v.x, u.y / v.y, u.z / v.z};
}

CUDA_CALLABLE Vec3D FMB::cov_inv_apply(const Vec3D vec) const {
    const auto rot = pose_.get_rot();
    return rot.inv().apply(vecdiv(rot.apply(vec), extent_));
}

CUDA_CALLABLE float FMB::quadratic_form(const Vec3D vec) const {
    const auto shifted_vec = vec - get_mean();
    return dot(shifted_vec, cov_inv_apply(shifted_vec));
}

template <>
__host__ FMBScene<MemoryLocation::HOST>::FMBScene(size_t size)
    : fmbs_{new FMB[size]}, log_weights_{new float[size]}, size_{size} {}

template <>
__host__ FMBScene<MemoryLocation::DEVICE>::FMBScene(size_t size) : size_{size} {
    CUDA_CHECK(cudaMalloc(&fmbs_, size * sizeof(FMB)));
    CUDA_CHECK(cudaMalloc(&log_weights_, size * sizeof(float)));
}

template <>
__host__ FMBScene<MemoryLocation::HOST>::~FMBScene() {
    delete[] fmbs_;
    delete[] log_weights_;
}

template <>
__host__ FMBScene<MemoryLocation::DEVICE>::~FMBScene() {
    CUDA_CHECK(cudaFree(fmbs_));
    CUDA_CHECK(cudaFree(log_weights_));
}

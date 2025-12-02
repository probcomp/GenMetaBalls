#include "fmb.cuh"
#include "geometry.cuh"
#include "utils.cuh"

CUDA_CALLABLE float FMB::quadratic_form(const Vec3D vec) const {
    const auto shftd_vec = vec - pose_.get_tran();
    const auto rot_shftd_vec = pose_.get_rot().apply(shftd_vec);
    const auto scaled_rot_shftd_vec = Vec3D(
        rot_shftd_vec.x / extent_.x, rot_shftd_vec.y / extent_.y, rot_shftd_vec.z / extent_.z);
    return dot(rot_shftd_vec, scaled_rot_shftd_vec);
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

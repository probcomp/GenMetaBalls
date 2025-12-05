#pragma once

#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <vector>

#include "geometry.cuh"
#include "utils.cuh"

class FMB {
private:
    // In Gaussian terms:
    // - mean: pose.tran
    // - cov: pose.rot.mat().inv() * diag(extent) * pose.rot.mat()
    // note: note how the inverse is on the left. This means that pose
    // corresponds to the precision matrix, i.e.
    // prec = pose.rot.mat() * diag(1/extent) * pose.rot.mat().inv()
    Pose pose_;
    float3 extent_;

public:
    FMB() : pose_{}, extent_{1.0f, 1.0f, 1.0f} {};

    FMB(const Pose& pose, float x_extent, float y_extent, float z_extent) noexcept(false)
        : pose_{pose} {
        if (x_extent <= 0 || y_extent <= 0 || z_extent <= 0)
            throw std::domain_error("a metaball cannot have negative extent");
        extent_ = {x_extent, y_extent, z_extent};
    }

    CUDA_CALLABLE Pose get_pose() const {
        return pose_;
    }
    CUDA_CALLABLE float3 get_extent() const {
        return extent_;
    }
    CUDA_CALLABLE float3 get_mean() const {
        return pose_.get_tran();
    }

    CUDA_CALLABLE Vec3D cov_inv_apply(const Vec3D) const;

    CUDA_CALLABLE float quadratic_form(const Vec3D) const;
};

template <MemoryLocation location>
class FMBScene {
private:
    // Host memory -> thrust::host_vector
    // Device memory -> thrust::device_vector
    template <typename T>
    using vector_t = std::conditional_t<location == MemoryLocation::HOST, thrust::host_vector<T>,
                                        thrust::device_vector<T>>;

    vector_t<FMB> fmbs_;
    vector_t<float> log_weights_;
    size_t size_;

public:
    __host__ FMBScene(size_t size) : size_{size}, fmbs_(size), log_weights_(size) {};

    // Copy constructor from std::vector
    // This enables easy construction from Python side
    __host__ FMBScene<location>(const std::vector<FMB>& fmbs, const std::vector<float>& log_weights)
        : size_{fmbs.size()}, fmbs_(fmbs.begin(), fmbs.end()),
          log_weights_(log_weights.begin(), log_weights.end()) {
        if (fmbs.size() != log_weights.size()) {
            throw std::invalid_argument(
                "FMBScene constructor: fmbs and log_weights must have the same size");
        }
    }

    CUDA_CALLABLE auto operator[](const uint32_t i) {
        return cuda::std::make_tuple(fmbs_[i], log_weights_[i]);
    }

    CUDA_CALLABLE auto operator[](const uint32_t i) const {
        return cuda::std::make_tuple(fmbs_[i], log_weights_[i]);
    }

    CUDA_CALLABLE auto begin() {
        return thrust::make_zip_iterator(fmbs_.begin(), log_weights_.begin());
    }
    CUDA_CALLABLE auto end() {
        return thrust::make_zip_iterator(fmbs_.end(), log_weights_.end());
    }
    CUDA_CALLABLE auto begin() const {
        return thrust::make_zip_iterator(fmbs_.begin(), log_weights_.begin());
    }
    CUDA_CALLABLE auto end() const {
        return thrust::make_zip_iterator(fmbs_.end(), log_weights_.end());
    }
    CUDA_CALLABLE const FMB& get_fmb(uint32_t idx) const {
        return fmbs_[idx];
    }
    CUDA_CALLABLE size_t size() const {
        return size_;
    }
};

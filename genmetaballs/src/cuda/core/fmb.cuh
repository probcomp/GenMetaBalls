#pragma once

#include <cuda/std/span>
#include <cuda/std/tuple>
#include <stdexcept>

#include "geometry.cuh"
#include "utils.cuh"

class FMB {
private:
    // In Gaussian terms:
    // - mean: pose.tran
    // - cov: pose.rot.mat().inv() * diag(extent) * pose.rot.mat()
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

    CUDA_CALLABLE float quadratic_form(const Vec3D) const;
};

template <MemoryLocation location>
class FMBScene {
private:
    FMB* fmbs_;
    float* log_weights_;
    size_t size_;

public:
    __host__ FMBScene(size_t size);

    __host__ ~FMBScene();

    CUDA_CALLABLE cuda::std::tuple<FMB&, float&> operator[](const uint32_t i) {
        return cuda::std::tie(fmbs_[i], log_weights_[i]);
    }

    CUDA_CALLABLE cuda::std::tuple<const FMB&, const float&> operator[](const uint32_t i) const {
        return cuda::std::tie(fmbs_[i], log_weights_[i]);
    }

    class Iterator {
    private:
        FMB* fmb_ptr_;
        float* log_weight_ptr_;

    public:
        CUDA_CALLABLE Iterator(FMB* const fmb_ptr, float* const log_weight_ptr)
            : fmb_ptr_{fmb_ptr}, log_weight_ptr_{log_weight_ptr} {}
        CUDA_CALLABLE cuda::std::tuple<FMB&, float&> operator*() {
            return cuda::std::tie(*fmb_ptr_, *log_weight_ptr_);
        }
        CUDA_CALLABLE bool operator!=(const Iterator& other) const {
            return fmb_ptr_ != other.fmb_ptr_ || log_weight_ptr_ != other.log_weight_ptr_;
        }
        CUDA_CALLABLE Iterator& operator++() {
            fmb_ptr_++, log_weight_ptr_++;
            return *this;
        }
    };

    class ConstIterator {
    private:
        const FMB* fmb_ptr_;
        const float* log_weight_ptr_;

    public:
        CUDA_CALLABLE ConstIterator(const FMB* const fmb_ptr, const float* const log_weight_ptr)
            : fmb_ptr_{fmb_ptr}, log_weight_ptr_{log_weight_ptr} {}
        CUDA_CALLABLE cuda::std::tuple<const FMB&, const float&> operator*() const {
            return cuda::std::tie(*fmb_ptr_, *log_weight_ptr_);
        }
        CUDA_CALLABLE bool operator!=(const ConstIterator& other) const {
            return fmb_ptr_ != other.fmb_ptr_ || log_weight_ptr_ != other.log_weight_ptr_;
        }
        CUDA_CALLABLE ConstIterator& operator++() {
            fmb_ptr_++, log_weight_ptr_++;
            return *this;
        }
    };

    CUDA_CALLABLE Iterator begin() {
        return Iterator(fmbs_, log_weights_);
    }
    CUDA_CALLABLE Iterator end() {
        return Iterator(fmbs_ + size_, log_weights_ + size_);
    }
    CUDA_CALLABLE ConstIterator begin() const {
        return ConstIterator(fmbs_, log_weights_);
    }
    CUDA_CALLABLE ConstIterator end() const {
        return ConstIterator(fmbs_ + size_, log_weights_ + size_);
    }
    CUDA_CALLABLE const FMB& get_fmb(uint32_t idx) const {
        return fmbs_[idx];
    }
    CUDA_CALLABLE size_t size() const {
        return size_;
    }
};

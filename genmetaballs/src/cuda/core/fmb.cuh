#pragma once

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
    FMB(const Pose& pose, float x_extent, float y_extent, float z_extent) noexcept(false)
        : pose_{pose} {
        if (x_extent <= 0 || y_extent <= 0 || z_extent <= 0)
            throw std::domain_error("a metaball cannot have negative extent");
        extent_ = {x_extent, y_extent, z_extent};
    }

    Pose get_pose() const {
        return pose_;
    }
    float3 get_extent() const {
        return extent_;
    }

    CUDA_CALLABLE float quadratic_form(const Vec3D) const;
};

/*template <typename containter_template>
class FMBScene {
private:
    containter_template<FMB> fmbs_;
    containter_template<float> log_weights_;

public:
    FMBs(uint32_t size) : fmbs_(size), log_weights_(size) {
        // TODO: set all log_weights_ to 0
    }
};*/

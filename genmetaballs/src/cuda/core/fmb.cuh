#pragma once

#include "geometry.cuh"

struct FMB {
    Pose pose;
    float3 extent;

    FMB(const Pose _pose, const float3 _extent) : pose{_pose}, extent{_extent} {}

    float quadratic_form(const Vec3D) const;
};

template <typename containter_template>
class FMBs {
private:
    containter_template<FMB> fmbs_;
    containter_template<float> log_weights_;

public:
    FMBs(uint32_t size) : fmbs_(size), log_weights_(size) {
        // TODO: set all log_weights_ to 0
    }
};

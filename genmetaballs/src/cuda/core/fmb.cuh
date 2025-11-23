#pragma once

#include "geometry.cuh"

struct FMB {
    Pose pose; // mean + orientation
    float3 extent;
};

template <template <typename> class containter_template>
class FMBs {
private:
    containter_template<FMB> fmbs_;
    containter_template<float> log_weights_;

public:
    FMBs(uint32_t size) : fmbs_(size), log_weights_(size) {
        // TODO: set all log_weights_ to 0
    }
    CUDA_CALLABLE const containter_template<FMB>& get_all_fmbs() const {
        return fmbs_;
    }
    CUDA_CALLABLE const FMB& get_fmb(uint32_t idx) const {
        return fmbs_[idx];
    }
};

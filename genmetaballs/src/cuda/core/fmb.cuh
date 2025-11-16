#pragma once

#include "geometry.cuh"

struct FMB {
    Pose pose; // mean + orientation
    float3 extent;
};

template <typename containter_template>
class FMBs {
private:
    containter_template<FMB> fmbs_;
    containter_template<float> log_weights_;

public:
    FMBs(uint32_t size) : fmbs_(size), log_weights_(size), {
        // TODO: set all log_weights_ to 0
    }
};

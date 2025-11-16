#pragma once

#include <cmath>

struct TwoParameterConfidence {
    float beta4;
    float beta5;

    __host__ __device__ __forceinline__ float get_confidence(float sumexpd) {
        return 0;
    } // TODO
};

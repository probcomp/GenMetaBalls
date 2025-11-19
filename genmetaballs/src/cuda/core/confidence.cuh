#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#include "utils.cuh"

struct TwoParameterConfidence {

    float beta4;
    float beta5;
    CUDA_CALLABLE __forceinline__ float get_confidence(float sumexpd) const {
        return sigmoid(beta4 * sumexpd + beta5);
    }
};

struct ZeroParameterConfidence {

    CUDA_CALLABLE __forceinline__ float get_confidence(float sumexpd) const {
        return 1.0f - expf(-sumexpd);
    }
};
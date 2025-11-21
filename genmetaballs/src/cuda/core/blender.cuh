#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "utils.cuh"

struct FourParameterBlender {
    float beta1;
    float beta2;
    float beta3;
    float eta;

    CUDA_CALLABLE __forceinline__ float blend(float t, float d) const {
        return expf((beta1 * d * sigmoid((beta3 / eta) * t)) - ((beta2 / eta) * t));
    }
};

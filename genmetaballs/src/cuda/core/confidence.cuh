#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#include "utils.cuh"

namespace nb = nanobind;

struct FiveParameterConfidence {

    float beta4;
    float beta5;
    __host__ __device__ __forceinline__ float get_confidence(float sumexpd) const {
        return sigmoid(beta4 * sumexpd + beta5);
    }
};

struct ThreeParameterConfidence {

    __host__ __device__ __forceinline__ float get_confidence(float sumexpd) const {
        return 1.0f - expf(-sumexpd);
    }
};

// According to the FMB+ code, the zero parameter confidence is the same as the three parameter
// confidence. Look at
// https://github.com/leonidk/fmb-plus/blob/235a078a402968554186a2ca752fb13afffb84f8/zpfm_render.py#L59
using ZeroParameterConfidence = ThreeParameterConfidence;
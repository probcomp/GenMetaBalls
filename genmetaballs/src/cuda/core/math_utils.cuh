#pragma once

#include <cmath>
#include <cuda_runtime.h>

__host__ __device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
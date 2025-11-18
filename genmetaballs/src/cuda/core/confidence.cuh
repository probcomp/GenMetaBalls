#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#include "math_utils.cuh"
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

// Generic CUDA kernel for computing confidence values
template <typename Confidence>
__global__ void confidence_kernel(const float* sumexpd, float* confidences, uint32_t n,
                                  Confidence confidence) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        confidences[i] = confidence.get_confidence(sumexpd[i]);
    }
}

// Generic GPU function to call the CUDA kernel for arbitrary Confidence type
template <uint32_t grid_dim, uint32_t block_dim, typename Confidence>
std::vector<float> gpu_get_confidence(const std::vector<float>& sumexpd_vec,
                                      Confidence confidence) {
    uint32_t n = sumexpd_vec.size();
    uint32_t nbytes = n * sizeof(float);
    float *d_sumexpd = nullptr, *d_confidences = nullptr;
    std::vector<float> result(n);

    CUDA_CHECK(cudaMalloc(&d_sumexpd, nbytes));
    CUDA_CHECK(cudaMalloc(&d_confidences, nbytes));

    CUDA_CHECK(cudaMemcpy(d_sumexpd, sumexpd_vec.data(), nbytes, cudaMemcpyHostToDevice));

    confidence_kernel<Confidence><<<grid_dim, block_dim>>>(d_sumexpd, d_confidences, n, confidence);

    CUDA_CHECK(cudaMemcpy(result.data(), d_confidences, nbytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_sumexpd));
    CUDA_CHECK(cudaFree(d_confidences));

    return result;
}
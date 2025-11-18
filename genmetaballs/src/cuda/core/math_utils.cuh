#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#include "utils.cuh"

template <typename T>
__host__ __device__ __forceinline__ float sigmoid(T x) {
    float x_float = static_cast<float>(x);
    if (isnan(x_float)) {
        return x_float;
    }
    return 1.0f / (1.0f + expf(-x_float));
}

// Generic CUDA kernel for computing sigmoid element-wise
template <typename T>
__global__ void sigmoid_kernel(const T* x, T* result, uint32_t n) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        result[i] = sigmoid(x[i]);
    }
}

// GPU function to compute sigmoid for a vector
template <uint32_t grid_dim, uint32_t block_dim, typename T>
std::vector<T> gpu_sigmoid(const std::vector<T>& x_vec) {
    uint32_t n = x_vec.size();
    uint32_t nbytes = n * sizeof(T);
    T *d_x = nullptr, *d_result = nullptr;
    std::vector<T> result(n);

    CUDA_CHECK(cudaMalloc(&d_x, nbytes));
    CUDA_CHECK(cudaMalloc(&d_result, nbytes));

    CUDA_CHECK(cudaMemcpy(d_x, x_vec.data(), nbytes, cudaMemcpyHostToDevice));

    sigmoid_kernel<T><<<grid_dim, block_dim>>>(d_x, d_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), d_result, nbytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}
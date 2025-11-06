#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#include "utils.h"

__global__ void add_kernel(
    float const *a, 
    float const *b, 
    const uint32_t n,
    float *sum
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n)
        sum[i] = a[i] + b[i];
}

template<uint32_t grid_dim, uint32_t block_dim>
std::vector<float> gpu_add(
    const std::vector<float> a_vec,
    const std::vector<float> b_vec
) {
    const uint32_t n = a_vec.size();
    const uint32_t nbytes = n * sizeof(float);
    float *a, *b, *sum;
    std::vector<float> sum_vec(n);
    CUDA_CHECK(cudaMalloc(&a, nbytes));
    CUDA_CHECK(cudaMalloc(&b, nbytes));
    CUDA_CHECK(cudaMalloc(&sum, nbytes));

    CUDA_CHECK(cudaMemcpy(a, a_vec.data(), nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, b_vec.data(), nbytes, cudaMemcpyHostToDevice));
    add_kernel<<<grid_dim, block_dim>>>(a, b, n, sum);
    CUDA_CHECK(cudaMemcpy(sum_vec.data(), sum, nbytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(sum));

    return sum_vec;
}

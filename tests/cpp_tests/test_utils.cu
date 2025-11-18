// #include <cstdint>
// #include <cuda_runtime.h>
// #include <gtest/gtest.h>
// #include <vector>

// #include "core/utils.cuh"

// // CUDA kernel for computing sigmoid element-wise
// __global__ void sigmoid_kernel(const float* x, float* result, uint32_t n) {
//     uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
//     if (i < n) {
//         result[i] = sigmoid(x[i]);
//     }
// }

// // GPU function to compute sigmoid for a vector (float only)
// template <uint32_t grid_dim, uint32_t block_dim>
// std::vector<float> gpu_sigmoid(const std::vector<float>& x_vec) {
//     uint32_t n = x_vec.size();
//     uint32_t nbytes = n * sizeof(float);
//     float *d_x = nullptr, *d_result = nullptr;
//     std::vector<float> result(n);

//     CUDA_CHECK(cudaMalloc(&d_x, nbytes));
//     CUDA_CHECK(cudaMalloc(&d_result, nbytes));

//     CUDA_CHECK(cudaMemcpy(d_x, x_vec.data(), nbytes, cudaMemcpyHostToDevice));

//     sigmoid_kernel<<<grid_dim, block_dim>>>(d_x, d_result, n);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaMemcpy(result.data(), d_result, nbytes, cudaMemcpyDeviceToHost));

//     CUDA_CHECK(cudaFree(d_x));
//     CUDA_CHECK(cudaFree(d_result));

//     return result;
// }
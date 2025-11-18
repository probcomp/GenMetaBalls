#include "math_utils.cuh"

// CUDA kernel for computing sigmoid element-wise (float only)
__global__ void sigmoid_kernel(const float* x, float* result, uint32_t n) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        result[i] = sigmoid(x[i]);
    }
}

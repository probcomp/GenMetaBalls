#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CALLABLE __host__ __device__

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cuda_check((x), __FILE__, __LINE__);                                                       \
    } while (0)

void cuda_check(cudaError_t code, const char* file, int line);

__host__ __device__ __forceinline__ float sigmoid(float x) {
    if (isnan(x)) {
        return x;
    }
    return 1.0f / (1.0f + expf(-x));
}

// XXX container_t should be a thrust container type
template <typename container_t>
class Array2D {
private:
    // XXX TODO: make sure this works
    container_t data_;

public:
    __host__ __device__ __forceinline__ container_t& at(const uint32_t i, const uint32_t j) {
        return data_;
        // return data_[i * width + j];
    }

    __host__ __device__ __forceinline__ const container_t& at(const uint32_t i,
                                                              const uint32_t j) const {
        return data_;
        // return data_[i * width + j];
    }

    __host__ __device__ constexpr uint32_t size() const {
        return 0;
        // return width * height;
    }
};

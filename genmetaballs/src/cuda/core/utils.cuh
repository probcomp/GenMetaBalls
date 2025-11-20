#pragma once

#include <cmath>
#include <cstdint>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>

#define CUDA_CALLABLE __host__ __device__

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cuda_check((x), __FILE__, __LINE__);                                                       \
    } while (0)

void cuda_check(cudaError_t code, const char* file, int line);

CUDA_CALLABLE __forceinline__ float sigmoid(float x) {
    if (isnan(x)) {
        return x;
    }
    return 1.0f / (1.0f + expf(-x));
}

// Non-owning 2D view into a contiguous array in either host or device memory
template <typename T>
class Array2D {
private:
    cuda::std::mdspan<
        T, cuda::std::extents<uint32_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>>
        data_view_;

public:
    // constructor
    __host__ __device__ constexpr Array2D(T* data, uint32_t rows, uint32_t cols)
        : data_view_(data, rows, cols) {}

    // accessor methods
    __host__ __device__ constexpr T& operator()(uint32_t row, uint32_t col) {
        return data_view_(row, col);
    }
    __host__ __device__ constexpr T operator()(uint32_t row, uint32_t col) const {
        return data_view_(row, col);
    }
    // size methods
    __host__ __device__ constexpr auto num_rows() const noexcept {
        return data_view_.extent(0);
    }
    __host__ __device__ constexpr auto num_cols() const noexcept {
        return data_view_.extent(1);
    }

    __host__ __device__ constexpr auto rank() const noexcept {
        return data_view_.rank();
    }
    __host__ __device__ constexpr auto size() const noexcept {
        return data_view_.size();
    }
};

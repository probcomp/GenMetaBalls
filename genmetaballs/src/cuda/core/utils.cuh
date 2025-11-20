#pragma once

#include <cmath>
#include <cstdint>
#include <cuda/std/mdspan>
#include <cuda/std/span>
#include <cuda_runtime.h>

#define CUDA_CALLABLE __host__ __device__

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cuda_check((x), __FILE__, __LINE__);                                                       \
    } while (0)

void cuda_check(cudaError_t code, const char* file, int line);

CUDA_CALLABLE __forceinline__ float sigmoid(float x) {
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
    CUDA_CALLABLE constexpr Array2D(T* data_ptr, uint32_t rows, uint32_t cols)
        : data_view_(data_ptr, rows, cols) {}

    // getting a 1D view of a specific row
    // this supports array2d[row][col] access pattern and range-based for loops
    // e.g., for (auto val : array2d[row]) { ... }
    CUDA_CALLABLE constexpr auto operator[](uint32_t row) const {
        return cuda::std::span<T>(data_view_.data_handle() + row * num_cols(), num_cols());
    }

    // size methods
    CUDA_CALLABLE constexpr auto num_rows() const noexcept {
        return data_view_.extent(0);
    }
    CUDA_CALLABLE constexpr auto num_cols() const noexcept {
        return data_view_.extent(1);
    }

    CUDA_CALLABLE constexpr auto rank() const noexcept {
        return data_view_.rank();
    }
    CUDA_CALLABLE constexpr auto size() const noexcept {
        return data_view_.size();
    }
}; // class Array2D

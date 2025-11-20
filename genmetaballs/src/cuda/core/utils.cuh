#pragma once

#include <cstdint>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cuda_check((x), __FILE__, __LINE__);                                                       \
    } while (0)

void cuda_check(cudaError_t code, const char* file, int line);

// Non-owning 2D view into a contiguous array in either host or device memory
template <typename T>
using Array2D = cuda::std::mdspan<
    T, cuda::std::extents<uint32_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>>;

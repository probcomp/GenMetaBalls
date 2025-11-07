#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

void cuda_check(cudaError_t code, const char *file, int line);

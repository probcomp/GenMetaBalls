#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#include "utils.h"

void cuda_check(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(code)
                  << std::endl;
        exit(1);
    }
}

#include <cstdint>
#include <vector>

#include "utils.h"

__global__ void add_kernel(
    float const *a, 
    float const *b, 
    const uint32_t n,
    float *sum
);

template<uint32_t grid_dim, uint32_t block_dim>
std::vector<float> gpu_add(
    const std::vector<float> a_vec,
    const std::vector<float> b_vec,
);

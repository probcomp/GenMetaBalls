#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#include "add.h"

int main()
{
    constexpr uint32_t N = 4096;
    constexpr uint32_t NBYTES = N * sizeof(float);
    constexpr uint32_t block_dim = 1024;
    constexpr uint32_t grid_dim = 4;

    std::vector<float> a_vec(N), b_vec(N), cpu_sum_vec(N);
    for(uint32_t i = 0; i < N; i++) {
        a_vec[i] = 2 * i * 3.14;
        b_vec[i] = (2 * i + 1) * 2.71;
        cpu_sum_vec[i] = a_vec[i] + b_vec[i];
    }

    auto gpu_sum_vec = gpu_add<grid_dim, block_dim>(a_vec, b_vec);

    for(uint32_t i = 0; i < N; i++)
        if(cpu_sum_vec[i] != gpu_sum_vec[i])
            return 1;

    return 0;
}

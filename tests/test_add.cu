#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "core/add.cuh"

TEST(GpuAddTest, BasicAddition) {
    constexpr uint32_t N = 4096;
    constexpr uint32_t block_dim = 1024;
    constexpr uint32_t grid_dim = 4;

    std::vector<float> a_vec(N), b_vec(N), cpu_sum_vec(N);
    for (uint32_t i = 0; i < N; i++) {
        a_vec[i] = 2 * i * 3.14;
        b_vec[i] = (2 * i + 1) * 2.71;
        cpu_sum_vec[i] = a_vec[i] + b_vec[i];
    }

    auto gpu_sum_vec = gpu_add<grid_dim, block_dim>(a_vec, b_vec);

    for (uint32_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(cpu_sum_vec[i], gpu_sum_vec[i]) << "Mismatch at index " << i;
    }
}

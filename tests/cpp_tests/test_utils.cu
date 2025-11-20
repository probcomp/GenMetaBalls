#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "core/utils.cuh"

TEST(Array2DTest, CreateAndAccessArray2DOnCPU) {
    uint32_t rows = 4;
    uint32_t cols = 6;
    // Array2D should work with any container
    thrust::host_vector<float> data(rows * cols);
    std::iota(data.begin(), data.end(), 0);

    // create 2D view into the underlying data on CPU
    auto array2d = Array2D<float>(data.data(), rows, cols);

    EXPECT_EQ(array2d.size(), rows * cols);
    EXPECT_EQ(array2d.extent(0), rows);
    EXPECT_EQ(array2d.extent(1), cols);

    // check the data is correct
    for (auto i = 0; i < array2d.extent(0); i++) {
        for (auto j = 0; j < array2d.extent(1); j++) {
            // in C++23, we will be able to use array2d[i, j] to access the element :)
            EXPECT_FLOAT_EQ(array2d(i, j), i * cols + j);
        }
    }
}

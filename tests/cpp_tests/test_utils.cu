#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>
#include <vector>

#include "core/utils.cuh"

// CUDA kernel to fill Array2D with sequential values
__global__ void fill_array2d_kernel(Array2D<float> array2d) {
    uint32_t i = threadIdx.x;
    uint32_t j = threadIdx.y;

    if (i < array2d.num_rows() && j < array2d.num_cols()) {
        array2d(i, j) = i * array2d.num_cols() + j;
    }
}

template <typename Container>
class Array2DTestFixture : public ::testing::Test {};

using ContainerTypes = ::testing::Types<std::vector<float>, thrust::device_vector<float>>;

TYPED_TEST_SUITE(Array2DTestFixture, ContainerTypes);

TYPED_TEST(Array2DTestFixture, CreateAndAccessArray2D) {
    uint32_t rows = 4;
    uint32_t cols = 6;

    auto data = TypeParam(rows * cols);
    // create 2D view into the underlying data on host or device
    auto array2d = Array2D<float>(thrust::raw_pointer_cast(data.data()), rows, cols);

    if constexpr (std::is_same_v<TypeParam, std::vector<float>>) {
        for (auto i = 0; i < rows; i++) {
            for (auto j = 0; j < cols; j++) {
                array2d(i, j) = i * cols + j;
            }
        }
    } else {
        // Launch kernel to fill Array2D on device
        // Note: we could've simply use thrust::sequence to fill the device vector,
        // but this is a simple example to demonstrate how to pass an Array2D to a kernel.
        fill_array2d_kernel<<<1, dim3(rows, cols)>>>(array2d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    EXPECT_EQ(array2d.size(), rows * cols);
    EXPECT_EQ(array2d.num_rows(), rows);
    EXPECT_EQ(array2d.num_cols(), cols);
    EXPECT_EQ(array2d.rank(), 2); // 2D array

    // create host vector to verify the data
    // for std::vector, this simply duplicate the vector.
    // for thrust::device_vector, it will copy the data to the host.
    thrust::host_vector<float> host_data = data;
    for (auto idx = 0; idx < rows * cols; idx++) {
        EXPECT_FLOAT_EQ(host_data[idx], idx);
    }
}

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

template <typename Container>
class HostArray2DTestFixture : public ::testing::Test {};

using HostContainerTypes = ::testing::Types<std::vector<float>, std::array<float, 24>, // 4 * 6 = 24
                                            thrust::host_vector<float>>;

TYPED_TEST_SUITE(HostArray2DTestFixture, HostContainerTypes);

TYPED_TEST(HostArray2DTestFixture, CreateAndAccessArray2DOnHost) {
    uint32_t rows = 4;
    uint32_t cols = 6;

    // Initialize container - std::array is fixed size, others use size constructor
    TypeParam data;
    if constexpr (std::is_same_v<TypeParam, std::array<float, 24>>) {
        data = std::array<float, 24>{};
    } else {
        data = TypeParam(rows * cols);
    }
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

// CUDA kernel to fill Array2D with sequential values
__global__ void fill_array2d(Array2D<float> array2d) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < array2d.extent(0) && j < array2d.extent(1)) {
        array2d(i, j) = i * array2d.extent(1) + j;
    }
}

TEST(Array2DTest, CreateAndAccessArray2DOnDevice) {
    uint32_t rows = 3;
    uint32_t cols = 5;
    uint32_t size = rows * cols;

    // Initialize device vector
    thrust::device_vector<float> device_data(size);

    // create 2D view into the underlying data on device
    auto array2d = Array2D<float>(thrust::raw_pointer_cast(device_data.data()), rows, cols);

    EXPECT_EQ(array2d.size(), rows * cols);
    EXPECT_EQ(array2d.extent(0), rows);
    EXPECT_EQ(array2d.extent(1), cols);

    // Launch kernel to fill Array2D on device
    // Note: we could've simply use thrust::sequence to fill the device vector,
    // but this is a simple example to demonstrate how to pass an Array2D to a kernel.
    dim3 block_size(16, 16);
    dim3 grid_size((rows + block_size.x - 1) / block_size.x,
                   (cols + block_size.y - 1) / block_size.y);
    fill_array2d<<<grid_size, block_size>>>(array2d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy data back to host to verify
    thrust::host_vector<float> host_data = device_data;
    for (auto i = 0; i < rows; i++) {
        for (auto j = 0; j < cols; j++) {
            EXPECT_FLOAT_EQ(host_data[i * cols + j], i * cols + j);
        }
    }
}

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

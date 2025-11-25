#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>
#include <vector>

#include "core/utils.cuh"

namespace test_utils_gpu {

// CUDA kernel for computing sigmoid element-wise
__global__ void sigmoid_kernel(const float* x, float* result, uint32_t n) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        result[i] = sigmoid(x[i]);
    }
}

// GPU function to compute sigmoid for a vector
template <uint32_t grid_dim, uint32_t block_dim>
std::vector<float> gpu_sigmoid(const std::vector<float>& x_vec) {
    uint32_t n = x_vec.size();
    uint32_t nbytes = n * sizeof(float);
    float *d_x = nullptr, *d_result = nullptr;
    std::vector<float> result(n);

    CUDA_CHECK(cudaMalloc(&d_x, nbytes));
    CUDA_CHECK(cudaMalloc(&d_result, nbytes));

    CUDA_CHECK(cudaMemcpy(d_x, x_vec.data(), nbytes, cudaMemcpyHostToDevice));

    sigmoid_kernel<<<grid_dim, block_dim>>>(d_x, d_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), d_result, nbytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

} // namespace test_utils_gpu

// Parameters matching the removed Python test
constexpr int NUM_RNG_SEEDS_PER_TEST = 5;
constexpr int NUM_N_VALUES_PER_TEST = 5;
constexpr int SEED_MASTER = 0;

// Helper: Generate test sizes [16, 32, 64, 128, 256]
static std::vector<int> sigmoid_test_sizes() {
    std::vector<int> sizes;
    for (int k = 0; k < NUM_N_VALUES_PER_TEST; ++k)
        sizes.push_back(1 << (4 + k)); // 2^(4+k)
    return sizes;
}

TEST(GpuSigmoidTest, SigmoidGPUWithinBounds) {
    // Generate seeds
    std::mt19937 master_gen(SEED_MASTER);
    std::uniform_int_distribution<uint32_t> seed_dist(0, std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> seeds(NUM_RNG_SEEDS_PER_TEST);
    for (auto& s : seeds)
        s = seed_dist(master_gen);

    auto sizes = sigmoid_test_sizes();

    for (size_t size_idx = 0; size_idx < sizes.size(); ++size_idx) {
        int N = sizes[size_idx];
        for (uint32_t seed : seeds) {
            // Create reproducible random numbers in [-10, 10]
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
            std::vector<float> x_vec(N);
            for (int i = 0; i < N; ++i)
                x_vec[i] = dist(rng);

            // Run on GPU
            constexpr uint32_t block_dim = 256;
            std::vector<float> actual = test_utils_gpu::gpu_sigmoid<1024, block_dim>(x_vec);

            // Check [0, 1] bounds
            ASSERT_TRUE(std::all_of(actual.begin(), actual.end(),
                                    [](float v) { return v >= 0.0f && v <= 1.0f; }))
                << "Sigmoid output out of [0,1] range for N=" << N << " seed=" << seed;
        }
    }
}

namespace test_utils_gpu {
// CUDA kernel to fill Array2D with sequential values
__global__ void fill_array2d_kernel(Array2D<float, DeviceType::GPU> array2d) {
    uint32_t i = threadIdx.x;
    uint32_t j = threadIdx.y;

    if (i < array2d.num_rows() && j < array2d.num_cols()) {
        if (i == array2d.num_rows() - 1) {
            array2d[i][j] = -1.0f; // last row set to -1
        } else {
            array2d[i][j] = i * array2d.num_cols() + j;
        }
    }
}
} // namespace test_utils_gpu

template <typename Container>
class Array2DTestFixture : public ::testing::Test {};

using ContainerTypes = ::testing::Types<std::vector<float>, thrust::device_vector<float>>;

TYPED_TEST_SUITE(Array2DTestFixture, ContainerTypes);

TYPED_TEST(Array2DTestFixture, CreateAndAccessArray2D) {
    uint32_t rows = 4;
    uint32_t cols = 6;

    auto data = TypeParam(rows * cols);
    constexpr auto device_type =
        std::is_same_v<TypeParam, thrust::device_vector<float>> ? DeviceType::GPU : DeviceType::CPU;
    // create 2D view into the underlying data on host or device
    auto array2d = Array2D<float, device_type>(data.data(), rows, cols);

    if constexpr (std::is_same_v<TypeParam, std::vector<float>>) {
        for (auto i = 0; i < rows - 1; i++) {
            for (auto j = 0; j < cols; j++) {
                array2d[i][j] = i * cols + j;
            }
        }
        for (auto& val : array2d[rows - 1]) {
            val = -1.0f; // setting last row to -1 with range-based for loop
        }
    } else {
        // Launch kernel to fill Array2D on device
        // Note: we could've simply use thrust::sequence to fill the device vector,
        // but this is a simple example to demonstrate how to pass an Array2D to a kernel.
        test_utils_gpu::fill_array2d_kernel<<<1, dim3(rows, cols)>>>(array2d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    EXPECT_EQ(array2d.size(), rows * cols);
    EXPECT_EQ(array2d.num_rows(), rows);
    EXPECT_EQ(array2d.num_cols(), cols);
    EXPECT_EQ(array2d.ndim(), 2); // 2D array

    // create host vector to verify the data
    // for std::vector, this simply duplicate the vector.
    // for thrust::device_vector, it will copy the data to the host.
    thrust::host_vector<float> host_data = data;
    for (auto idx = 0; idx < (rows - 1) * cols; idx++) {
        EXPECT_FLOAT_EQ(host_data[idx], idx);
    }
    for (auto idx = (rows - 1) * cols; idx < rows * cols; idx++) {
        EXPECT_FLOAT_EQ(host_data[idx], -1.0f);
    }
}

// Test that modifications through view affect underlying data
TYPED_TEST(Array2DTestFixture, ViewModifiesUnderlyingData) {
    if constexpr (std::is_same_v<TypeParam, std::vector<float>>) {
        uint32_t rows = 3;
        uint32_t cols = 4;
        auto data = TypeParam(rows * cols, 0.0f);
        constexpr auto device_type = std::is_same_v<TypeParam, thrust::device_vector<float>>
                                         ? DeviceType::GPU
                                         : DeviceType::CPU;
        auto array2d = Array2D<float, device_type>(data.data(), rows, cols);

        // Modify through view
        array2d[1][2] = 42.5f;
        // Verify underlying data changed
        EXPECT_FLOAT_EQ(data[1 * cols + 2], 42.5f);

        // Modify underlying data directly
        data[0 * cols + 1] = 99.9f;
        // Verify view reflects change
        EXPECT_FLOAT_EQ(array2d[0][1], 99.9f);
    }
}

// Test multiple views of the same data
TYPED_TEST(Array2DTestFixture, MultipleViewsOfSameData) {
    if constexpr (std::is_same_v<TypeParam, std::vector<float>>) {
        uint32_t rows = 2;
        uint32_t cols = 3;
        auto data = TypeParam(rows * cols, 0.0f);
        constexpr auto device_type = std::is_same_v<TypeParam, thrust::device_vector<float>>
                                         ? DeviceType::GPU
                                         : DeviceType::CPU;
        auto view1 = Array2D<float, device_type>(data.data(), rows, cols);
        auto view2 = Array2D<float, device_type>(data.data(), rows, cols);

        // Modify through view1
        view1[0][0] = 100.0f;
        // Verify view2 sees the change
        EXPECT_FLOAT_EQ(view2[0][0], 100.0f);

        // Modify through view2
        view2[1][2] = 200.0f;
        // Verify view1 sees the change
        EXPECT_FLOAT_EQ(view1[1][2], 200.0f);
    }
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "core/utils.cuh"

namespace test_utils_gpu {

// CUDA kernel for computing sigmoid element-wise (relies on __device__ sigmoid in utils.cuh)
__global__ void sigmoid_kernel(const float* x, float* result, uint32_t n) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        result[i] = sigmoid(x[i]);
    }
}

// GPU function to compute sigmoid for a vector (float only)
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

// Host sigmoid for reference
inline float host_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
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

TEST(GpuSigmoidTest, SigmoidVectorCorrectness) {
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

            // Compute expected (host)
            std::vector<float> expected(N);
            for (int i = 0; i < N; ++i)
                expected[i] = test_utils_gpu::host_sigmoid(x_vec[i]);

            // Compute actual (GPU)
            constexpr uint32_t block_dim = 256;
            uint32_t grid_dim = (N + block_dim - 1) / block_dim;
            std::vector<float> actual = test_utils_gpu::gpu_sigmoid<1024, block_dim>(x_vec);

            // Compare
            ASSERT_EQ(actual.size(), expected.size());
            for (int i = 0; i < N; ++i) {
                ASSERT_NEAR(actual[i], expected[i], 1e-5)
                    << "at idx=" << i << " for N=" << N << " seed=" << seed;
            }

            // Check [0, 1] bounds
            ASSERT_TRUE(std::all_of(actual.begin(), actual.end(),
                                    [](float v) { return v >= 0.0f && v <= 1.0f; }))
                << "Sigmoid output out of [0,1] range for N=" << N << " seed=" << seed;
        }
    }
}
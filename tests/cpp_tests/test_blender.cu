#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

#include "core/blender.cuh"
#include "core/utils.cuh"

template <typename Blender>
__global__ void blender_kernel(const float* t, const float* d, float* blended, uint32_t n,
                               Blender blender) {
    uint32_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < n) {
        blended[i] = blender.blend(t[i], d[i]);
    }
}

constexpr uint32_t GRID_DIM = 256;
constexpr uint32_t BLOCK_DIM = 1024;

template <typename Blender>
std::vector<float> gpu_blend(const std::vector<float>& t_vec, const std::vector<float>& d_vec,
                             Blender blender) {
    auto n = static_cast<uint32_t>(t_vec.size());
    auto nbytes = n * sizeof(float);
    float *d_t = nullptr, *d_d = nullptr, *d_blended = nullptr;
    std::vector<float> result(n);

    CUDA_CHECK(cudaMalloc(&d_t, nbytes));
    CUDA_CHECK(cudaMalloc(&d_d, nbytes));
    CUDA_CHECK(cudaMalloc(&d_blended, nbytes));
    CUDA_CHECK(cudaMemcpy(d_t, t_vec.data(), nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, d_vec.data(), nbytes, cudaMemcpyHostToDevice));

    auto block_dim = BLOCK_DIM;
    auto grid_dim = (n + block_dim - 1) / block_dim;
    if (grid_dim > GRID_DIM)
        grid_dim = GRID_DIM;

    blender_kernel<Blender><<<grid_dim, block_dim>>>(d_t, d_d, d_blended, n, blender);

    CUDA_CHECK(cudaMemcpy(result.data(), d_blended, nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_blended));
    return result;
}

constexpr int NUM_RNG_SEEDS_PER_TEST = 5;
constexpr int NUM_N_VALUES_PER_TEST = 5;
constexpr uint32_t MASTER_SEED = 0;

static std::vector<int> blender_test_sizes() {
    std::vector<int> sizes;
    for (int k = 0; k < NUM_N_VALUES_PER_TEST; ++k)
        sizes.push_back(1 << (4 + k)); // 2^(4+k): [16, 32, 64, 128, 256]
    return sizes;
}

struct BlenderCase {
    float beta1, beta2, beta3, eta;
    const char* name;
};

static std::vector<BlenderCase> blender_cases() {
    return {
        {1.0F, 0.5F, 0.2F, 2.0F, "case1"},
        {-2.0F, 1.0F, -1.0F, 1.5F, "case2"},
        {0.0F, 0.0F, 1.0F, 1.0F, "case3"},
        {0.5F, -0.5F, 0.8F, 0.5F, "case4"},
    };
}

TEST(GpuBlenderTest, Blender_GPU_Smoke_FourParameter) {
    auto sizes = blender_test_sizes();
    std::mt19937 master_gen(MASTER_SEED);
    std::uniform_int_distribution<uint32_t> seed_dist(0, std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> seeds(NUM_RNG_SEEDS_PER_TEST);
    for (auto& s : seeds)
        s = seed_dist(master_gen);

    for (int size_idx = 0; size_idx < static_cast<int>(sizes.size()); ++size_idx) {
        int N = sizes[size_idx];

        for (const auto& blend_case : blender_cases()) {
            for (uint32_t test_seed : seeds) {
                SCOPED_TRACE(testing::Message() << "N=" << N << ", seed=" << test_seed
                                                << ", blend_type=" << blend_case.name);

                std::mt19937 rng(test_seed);
                std::uniform_real_distribution<float> tdist(0.0F, 10.0F);
                std::uniform_real_distribution<float> ddist(0.0F, 10.0F);

                std::vector<float> t_vec(N), d_vec(N);
                for (int i = 0; i < N; ++i) {
                    t_vec[i] = tdist(rng);
                    d_vec[i] = ddist(rng);
                }

                FourParameterBlender blender{blend_case.beta1, blend_case.beta2, blend_case.beta3,
                                             blend_case.eta};

                // smoke testing to see if this gpu kernel can run, i'm not gong to test for
                // correctness here since t was done in python
                std::vector<float> actual = gpu_blend(t_vec, d_vec, blender);

                ASSERT_EQ(actual.size(), static_cast<size_t>(N));
                // Optionally verify all are finite
                ASSERT_TRUE(std::all_of(actual.begin(), actual.end(),
                                        [](float v) { return std::isfinite(v); }));
            }
        }
    }
}

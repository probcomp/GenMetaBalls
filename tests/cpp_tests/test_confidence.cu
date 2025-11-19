#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "core/confidence.cuh"

// Helper: Python ground truth, as in test_confidence.py
inline float ground_truth_expit(float x) {
    return 1.0F / (1.0F + std::exp(-x));
}
float ground_truth_two_parameter_confidence(float beta4, float beta5, float sumexpd) {
    return ground_truth_expit((beta4 * sumexpd) + beta5);
}
float ground_truth_zero_parameter_confidence(float sumexpd) {
    return 1.0F - std::exp(-sumexpd);
}

template <typename Confidence>
__global__ void confidence_kernel(const float* sumexpd, float* confidences, uint32_t n,
                                  Confidence confidence) {
    uint32_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < n) {
        confidences[i] = confidence.get_confidence(sumexpd[i]);
    }
}

constexpr uint32_t GRID_DIM = 256;
constexpr uint32_t BLOCK_DIM = 1024;

template <typename Confidence>
std::vector<float> gpu_get_confidence(const std::vector<float>& sumexpd_vec,
                                      Confidence confidence) {
    auto n = static_cast<uint32_t>(sumexpd_vec.size());
    auto nbytes = n * sizeof(float);
    float *d_sumexpd = nullptr, *d_confidences = nullptr;
    std::vector<float> result(n);

    CUDA_CHECK(cudaMalloc(&d_sumexpd, nbytes));
    CUDA_CHECK(cudaMalloc(&d_confidences, nbytes));
    CUDA_CHECK(cudaMemcpy(d_sumexpd, sumexpd_vec.data(), nbytes, cudaMemcpyHostToDevice));

    auto block_dim = BLOCK_DIM;
    auto grid_dim = (n + block_dim - 1) / block_dim;
    if (grid_dim > GRID_DIM)
        grid_dim = GRID_DIM;

    confidence_kernel<Confidence><<<grid_dim, block_dim>>>(d_sumexpd, d_confidences, n, confidence);

    CUDA_CHECK(cudaMemcpy(result.data(), d_confidences, nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sumexpd));
    CUDA_CHECK(cudaFree(d_confidences));
    return result;
}

constexpr int NUM_RNG_SEEDS_PER_TEST = 5;
constexpr int NUM_N_VALUES_PER_TEST = 5;
constexpr uint32_t MASTER_SEED = 0;

static std::vector<int> confidence_test_sizes() {
    std::vector<int> sizes;
    for (int k = 0; k < NUM_N_VALUES_PER_TEST; ++k)
        sizes.push_back(1 << (4 + k)); // 2^(4+k): [16, 32, 64, 128, 256]
    return sizes;
}

// Define simple struct to match python CONFIDENCE_TEST_CASES
struct ConfidenceCase {
    std::string name;
    float beta4 = 0.0F;
    float beta5 = 0.0F;
    bool is_two_param;
};
static std::vector<ConfidenceCase> confidence_cases() {
    return {
        {"zero_param", 0.0F, 0.0F, false},
        {"two_param_0.5_-1", 0.5F, -1.0F, true},
        {"two_param_1_0", 1.0F, 0.0F, true},
        {"two_param_-0.5_2", -0.5F, 2.0F, true},
    };
}

TEST(GpuConfidenceTest, ConfidenceMultipleValuesGPU_AllTypes) {
    using test_float = float;
    constexpr float rtol = 1e-6F;

    auto sizes = confidence_test_sizes();
    std::mt19937 master_gen(MASTER_SEED);
    std::uniform_int_distribution<uint32_t> seed_dist(0, std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> seeds(NUM_RNG_SEEDS_PER_TEST);
    for (auto& s : seeds)
        s = seed_dist(master_gen);

    for (int size_idx = 0; size_idx < static_cast<int>(sizes.size()); ++size_idx) {
        int N = sizes[size_idx];

        for (const auto& conf_case : confidence_cases()) {
            for (uint32_t test_seed : seeds) {
                SCOPED_TRACE(testing::Message() << "N=" << N << ", seed=" << test_seed
                                                << ", conf_type=" << conf_case.name);

                // Use float32 min as lower and float32 max as upper bound
                std::mt19937 rng(test_seed);
                std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(),
                                                           std::numeric_limits<float>::max());
                std::vector<float> sumexpd_vec(N);
                for (int i = 0; i < N; ++i)
                    sumexpd_vec[i] = dist(rng);

                std::vector<float> expected(N);
                if (conf_case.is_two_param) {
                    for (int i = 0; i < N; ++i)
                        expected[i] = ground_truth_two_parameter_confidence(
                            conf_case.beta4, conf_case.beta5, sumexpd_vec[i]);
                } else {
                    for (int i = 0; i < N; ++i)
                        expected[i] = ground_truth_zero_parameter_confidence(sumexpd_vec[i]);
                }

                std::vector<float> actual;
                if (conf_case.is_two_param) {
                    TwoParameterConfidence conf(conf_case.beta4, conf_case.beta5);
                    actual = gpu_get_confidence(sumexpd_vec, conf);
                } else {
                    ZeroParameterConfidence conf;
                    actual = gpu_get_confidence(sumexpd_vec, conf);
                }

                ASSERT_EQ(actual.size(), expected.size());
                for (int i = 0; i < N; ++i) {
                    ASSERT_NEAR(actual[i], expected[i], 1e-6F)
                        << "at idx=" << i << " N=" << N << " conf_type=" << conf_case.name
                        << " exp=" << expected[i] << " act=" << actual[i];
                }
                // Ensure all actual values are in [0, 1]
                ASSERT_TRUE(std::all_of(actual.begin(), actual.end(),
                                        [](float v) { return v >= 0.0F && v <= 1.0F; }))
                    << "out-of-bounds value(s) detected for conf_type=" << conf_case.name;
            }
        }
    }
}
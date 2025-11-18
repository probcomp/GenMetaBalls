// #include <cstdint>
// #include <cuda_runtime.h>
// #include <gtest/gtest.h>
// #include <vector>

// #include "core/confidence.cuh"

// // CUDA kernel for computing confidence values
// // Template kernel definition - safe in header since each instantiation is a different function
// template <typename Confidence>
// __global__ void confidence_kernel(const float* sumexpd, float* confidences, uint32_t n,
//                                   Confidence confidence) {
//     uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
//     if (i < n) {
//         confidences[i] = confidence.get_confidence(sumexpd[i]);
//     }
// }

// // Generic GPU function to call the CUDA kernel for arbitrary Confidence type
// template <uint32_t grid_dim, uint32_t block_dim, typename Confidence>
// std::vector<float> gpu_get_confidence(const std::vector<float>& sumexpd_vec,
//                                       Confidence confidence) {
//     uint32_t n = sumexpd_vec.size();
//     uint32_t nbytes = n * sizeof(float);
//     float *d_sumexpd = nullptr, *d_confidences = nullptr;
//     std::vector<float> result(n);

//     CUDA_CHECK(cudaMalloc(&d_sumexpd, nbytes));
//     CUDA_CHECK(cudaMalloc(&d_confidences, nbytes));

//     CUDA_CHECK(cudaMemcpy(d_sumexpd, sumexpd_vec.data(), nbytes, cudaMemcpyHostToDevice));

//     confidence_kernel<Confidence><<<grid_dim, block_dim>>>(d_sumexpd, d_confidences, n,
//     confidence);

//     CUDA_CHECK(cudaMemcpy(result.data(), d_confidences, nbytes, cudaMemcpyDeviceToHost));

//     CUDA_CHECK(cudaFree(d_sumexpd));
//     CUDA_CHECK(cudaFree(d_confidences));

//     return result;
// }

// TEST(GpuConfidenceTest, ThreeParameterConfidence) {
//     constexpr uint32_t N = 4096;
//     constexpr uint32_t block_dim = 1024;
//     constexpr uint32_t grid_dim = 4;

//     std::vector<float> a_vec(N), b_vec(N), cpu_sum_vec(N);
//     for (uint32_t i = 0; i < N; i++) {
//         a_vec[i] = 2 * i * 3.14;
//         b_vec[i] = (2 * i + 1) * 2.71;
//         cpu_sum_vec[i] = a_vec[i] + b_vec[i];
//     }

//     auto gpu_sum_vec = gpu_add<grid_dim, block_dim>(a_vec, b_vec);

//     for (uint32_t i = 0; i < N; i++) {
//         EXPECT_FLOAT_EQ(cpu_sum_vec[i], gpu_sum_vec[i]) << "Mismatch at index " << i;
//     }
// }
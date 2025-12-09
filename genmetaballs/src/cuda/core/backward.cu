#include <cstdint>
#include <cuda_runtime.h>

#include "backward.cuh"
#include "utils.cuh"

__host__ FMBSceneGradient::FMBSceneGradient(
    const FMBScene<MemoryLocation::DEVICE>& fmbs, const Intrinsics& intr, const Pose& extr
): tau_grad_{}, rho_grad_{}, mu_grad_{}, pi_grad_{}, fmbs_{fmbs}, intr_{intr},
   extr_{extr_}
{
    const auto H = intr.height;
    const auto W = intr.width;
    const auto N = fmbs.size();

    CUDA_CHECK(cudaMalloc(&tau_buf_, H * W * sizeof(Vec3D)));
    CUDA_CHECK(cudaMalloc(&rho_buf_, H * W * sizeof(Mat33)));
    CUDA_CHECK(cudaMalloc(&lambda_buf_, H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mu_buf_, H * W * N * sizeof(Vec3D)));
    CUDA_CHECK(cudaMalloc(&pi_buf_, H * W * N * sizeof(Mat33)));

    CUDA_CHECK(cudaMemset(tau_buf_, 0, H * W * sizeof(Vec3D)));
    CUDA_CHECK(cudaMemset(rho_buf_, 0, H * W * sizeof(Mat33)));
    CUDA_CHECK(cudaMemset(lambda_buf_, 0,  H * W * sizeof(float)));
    CUDA_CHECK(cudaMemset(mu_buf_, 0, H * W * N * sizeof(Vec3D)));
    CUDA_CHECK(cudaMemset(pi_buf_, 0, H * W * N * sizeof(Mat33)));

}

__host__ FMBSceneGradient::~FMBSceneGradient()
{
    CUDA_CHECK(cudaFree(tau_buf_));
    CUDA_CHECK(cudaFree(rho_buf_));
    CUDA_CHECK(cudaFree(lambda_buf_));
    CUDA_CHECK(cudaFree(mu_buf_));
    CUDA_CHECK(cudaFree(pi_buf_));
}


CUDA_CALLABLE void FMBSceneGradient::accumulate_cam_pose_grads(
    uint32_t i, uint32_t j, uint32_t k, float d, float q, float lambda
) {

}

CUDA_CALLABLE void FMBSceneGradient::accumulate_fmb_grads(
    uint32_t i, uint32_t j, uint32_t k, float d, float q, float lambda
) {
}

CUDA_CALLABLE void FMBSceneGradient::adjust_cam_pose_grads(
    uint32_t i, uint32_t j, uint32_t k, float c, float y
) {
}

CUDA_CALLABLE void FMBSceneGradient::adjust_fmb_grads(
    uint32_t i, uint32_t j, uint32_t k, float c, float y
) {
}

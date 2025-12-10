#include <cstdint>
#include <cuda_runtime.h>

#include "backward.cuh"
#include "utils.cuh"

__host__ FMBSceneGradient::FMBSceneGradient(
    const FMBScene<MemoryLocation::DEVICE>& fmbs, const Intrinsics& intr, const Pose& extr
): tau_grad_{}, rho_grad_{}, lambda_grad_{}, mu_grad_{}, pi_grad_{}, fmbs_{fmbs}, intr_{intr},
   extr_{extr_}
{
    const auto H = intr.height;
    const auto W = intr.width;
    const auto N = fmbs.size();

    CUDA_CHECK(cudaMalloc(&tau_buf_,    H * W * N * sizeof(Vec3D)));
    CUDA_CHECK(cudaMalloc(&rho_buf_,    H * W * N * sizeof(Mat33)));
    CUDA_CHECK(cudaMalloc(&lambda_buf_, H * W * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mu_buf_,     H * W * N * sizeof(Vec3D)));
    CUDA_CHECK(cudaMalloc(&pi_buf_,     H * W * N * sizeof(Mat33)));

    CUDA_CHECK(cudaMalloc(&tau_grad_, sizeof(Vec3D)));
    CUDA_CHECK(cudaMalloc(&rho_grad_, sizeof(Mat33)));
    CUDA_CHECK(cudaMalloc(&lambda_grad_, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mu_grad_, N * sizeof(Vec3D)));
    CUDA_CHECK(cudaMalloc(&pi_grad_, N * sizeof(Mat33)));


    CUDA_CHECK(cudaMemset(tau_buf_, 0,    H * W * N * sizeof(Vec3D)));
    CUDA_CHECK(cudaMemset(rho_buf_, 0,    H * W * N * sizeof(Mat33)));
    CUDA_CHECK(cudaMemset(lambda_buf_, 0, H * W * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(mu_buf_, 0,     H * W * N * sizeof(Vec3D)));
    CUDA_CHECK(cudaMemset(pi_buf_, 0,     H * W * N * sizeof(Mat33)));

    CUDA_CHECK(cudaMemset(tau_grad_, 0, sizeof(Vec3D)));
    CUDA_CHECK(cudaMemset(rho_grad_, 0, sizeof(Mat33)));
    CUDA_CHECK(cudaMemset(lambda_grad_, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(mu_grad_, 0, N * sizeof(Vec3D)));
    CUDA_CHECK(cudaMemset(pi_grad_, 0, N * sizeof(Mat33)));
}

__host__ FMBSceneGradient::~FMBSceneGradient()
{
    CUDA_CHECK(cudaFree(tau_buf_));
    CUDA_CHECK(cudaFree(rho_buf_));
    CUDA_CHECK(cudaFree(lambda_buf_));
    CUDA_CHECK(cudaFree(mu_buf_));
    CUDA_CHECK(cudaFree(pi_buf_));

    CUDA_CHECK(cudaFree(tau_grad_));
    CUDA_CHECK(cudaFree(rho_grad_));
    CUDA_CHECK(cudaFree(lambda_grad_));
    CUDA_CHECK(cudaFree(mu_grad_));
    CUDA_CHECK(cudaFree(pi_grad_));
}


__device__ void FMBSceneGradient::buffer_grads(
    uint32_t i, uint32_t j, uint32_t k, float d, float q
) {

    const auto W = intr_.width;
    const auto N = fmbs_.size();
    const auto ij = i * W + j;
    const auto ijk = ij * N + k;

    const auto [fmb, lambda] = fmbs_[k];
    const auto mu = fmb.get_mean();
    const auto tau = extr_.get_tran();
    const auto vij = intr_.get_ray_direction(j, i);
    const auto vijt = extr_.get_rot().apply(vij);

    const auto pi_vijt = fmb.cov_inv_apply(vijt);
    const auto pi_mu_minus_tau = fmb.cov_inv_apply(mu - tau);
    const auto vijt_pi_vijt = dot(vijt, fmb.cov_inv_apply(vijt));
    const auto big_square = dot(vijt, pi_mu_minus_tau)/(vijt_pi_vijt * vijt_pi_vijt);

    const auto dc_dq_stem = -0.5f * exp(lambda - q/2);
    const auto dq_dd = -2 * dot(mu - tau - d*vijt, pi_vijt);
    const auto dd_dmu = pi_vijt / vijt_pi_vijt;
    const auto dd_dtau = -dd_dmu;
    const auto dq_dmu = 2 * fmb.cov_inv_apply(mu - tau - d*vijt);
    const auto dq_dtau = -dd_dmu;
    const auto dq_dvijt = -d * dd_dmu;
    const auto dd_dvijt = -2 * big_square * pi_vijt + pi_mu_minus_tau / vijt_pi_vijt;
    const auto dq_dpi = outer(mu - tau - d*vijt,  mu - tau - d*vijt);
    const auto dd_dpi = outer(mu - tau, vijt)/vijt_pi_vijt - outer(vijt, vijt)*big_square;


    tau_buf_[ijk] = dc_dq_stem * (dq_dd * dd_dtau + dq_dtau);
    rho_buf_[ijk] = dc_dq_stem * outer(dq_dd * dd_dvijt + dq_dvijt, vij);

    lambda_buf_[ijk] = -2 * dc_dq_stem;
    mu_buf_[ijk]     = dc_dq_stem * (dq_dmu + dq_dd * dd_dmu);
    pi_buf_[ijk]     = dc_dq_stem * (dq_dpi + dq_dd * dd_dpi);
}

__device__ void FMBSceneGradient::accumulate_grads(
    uint32_t i, uint32_t j, uint32_t k, float c, float y
) {
    const auto W = intr_.width;
    const auto N = fmbs_.size();

    const auto ij = i * W + j;
    const auto ijk = ij * N + k;

    const auto dL_dc = (1 - y)/(1 - c) - y/c;
    const auto adjustment = (1 - c) * dL_dc;

    componentwise_atomic_add(*tau_grad_, adjustment * tau_buf_[ijk]);
    componentwise_atomic_add(*rho_grad_, adjustment * rho_buf_[ijk]);
    atomicAdd(&lambda_grad_[k], adjustment * lambda_buf_[ijk]);
    componentwise_atomic_add(mu_grad_[k], adjustment * mu_buf_[ijk]);
    componentwise_atomic_add(pi_grad_[k], adjustment * pi_buf_[ijk]);
}


#pragma once

#include <stdexcept>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>


#include "geometry.cuh"
#include "utils.cuh"

class FMB {
private:
    // In Gaussian terms:
    // - mean: pose.tran
    // - cov: pose.rot.mat().inv() * diag(extent) * pose.rot.mat()
    Pose pose_;
    float3 extent_;

public:
    FMB(): pose_{}, extent_{1.0f, 1.0f, 1.0f} {};

    FMB(const Pose& pose, float x_extent, float y_extent, float z_extent) noexcept(false)
        : pose_{pose} {
        if (x_extent <= 0 || y_extent <= 0 || z_extent <= 0)
            throw std::domain_error("a metaball cannot have negative extent");
        extent_ = {x_extent, y_extent, z_extent};
    }

    CUDA_CALLABLE Pose get_pose() const {
        return pose_;
    }
    CUDA_CALLABLE float3 get_extent() const {
        return extent_;
    }

    CUDA_CALLABLE float quadratic_form(const Vec3D) const;
};


class FMBScene {
private:
    thrust::device_vector<FMB> fmbs_;
    thrust::device_vector<float> log_weights_;

    friend class FMBSceneHostView;

public:
    
    // constructors 
    __host__ FMBScene(size_t size): fmbs_(size), log_weights_(size) {}

    __device__ thrust::device_vector<FMB> &get_fmbs() { return fmbs_; }
    __device__ thrust::device_vector<FMB> get_fmbs() const { return fmbs_; }

    __device__ thrust::device_vector<float> &get_log_weights() { return log_weights_; }
    __device__ thrust::device_vector<float> get_log_weights() const { return log_weights_; }

    __device__ thrust::tuple<FMB, float> operator[] (const uint32_t i) {
        return thrust::make_tuple(fmbs_[i], log_weights_[i]);
    }

    __device__ auto begin() {
        return thrust::make_zip_iterator(fmbs_.begin(), log_weights_.begin());
    }

    __device__ auto end() {
        return thrust::make_zip_iterator(fmbs_.end(), log_weights_.end());
    }

};

class FMBSceneHostView {
private:
    thrust::host_vector<FMB> fmbs_;
    thrust::host_vector<float> log_weights_;

public:
    FMBSceneHostView(FMBScene &scene) :
        fmbs_{scene.fmbs_}, log_weights_{scene.log_weights_} {}

    __device__ thrust::device_vector<FMB> get_fmbs() { return fmbs_; }

    __device__ thrust::device_vector<float> get_log_weights() const { return log_weights_; }

    __device__ thrust::tuple<FMB, float> operator[] (const uint32_t i) {
        return thrust::make_tuple(fmbs_[i], log_weights_[i]);
    }

    __device__ auto begin() {
        return thrust::make_zip_iterator(fmbs_.begin(), log_weights_.begin());
    }

    __device__ auto end() {
        return thrust::make_zip_iterator(fmbs_.end(), log_weights_.end());
    }

};


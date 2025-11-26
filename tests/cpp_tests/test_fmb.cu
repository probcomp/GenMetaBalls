#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "core/fmb.cuh"

__global__ void dummy_kernel(FMBScene &scene, float *tot_extent_x) {

    float par_extent_x = 0;
    
    for(auto [fmb, w] : scene) {
        par_extent_x += fmb.get_extent().x;
    }

    *tot_extent_x = par_extent_x;

}

TEST(FMBTests, KernelRangeBasedForLoopSmokeTest) {

    FMBScene dummy_scene(10);
    thrust::device_vector<float> device_res(1);

    dummy_kernel<<<1, 1>>>(dummy_scene, thrust::raw_pointer_cast(device_res.data()));

    thrust::host_vector<float> host_res = device_res;

    EXPECT_EQ(host_res[0], 10.0f);

}

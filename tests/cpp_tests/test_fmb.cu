#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "core/fmb.cuh"
#include "core/utils.cuh"

__global__ void dummy_kernel(FMBScene<MemoryLocation::DEVICE>& scene, int* num_fmbs) {

    int _num_fmbs = 0;

    for (auto [fmb, w] : scene) {
        _num_fmbs += 1;
    }

    *num_fmbs = _num_fmbs;
}

TEST(FMBTests, KernelRangeBasedForLoopSmokeTest) {

    FMBScene<MemoryLocation::DEVICE> dummy_scene(10);
    thrust::device_vector<int> device_res(1);

    dummy_kernel<<<1, 1>>>(dummy_scene, thrust::raw_pointer_cast(device_res.data()));

    thrust::host_vector<int> host_res = device_res;

    EXPECT_EQ(host_res[0], 10);
}

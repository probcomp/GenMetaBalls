#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "core/camera.cuh"
#include "core/geometry.cuh"
#include "core/utils.cuh"

namespace test_camera_gpu {

// CUDA kernel to call get_ray_directions on device with multiple threads
// Each thread processes one row of the image
__global__ void get_ray_directions_kernel(Intrinsics intrinsics,
                                          Array2D<Vec3D, DeviceType::GPU> ray_buffer) {
    uint32_t row_start = threadIdx.x * 2;
    uint32_t row_end = max(row_start + 2, intrinsics.height);
    uint32_t col_start = threadIdx.y * 2;
    uint32_t col_end = max(col_start + 2, intrinsics.width);
    intrinsics.get_ray_directions(ray_buffer, row_start, row_end, col_start, col_end);
}

} // namespace test_camera_gpu

// Test get_ray_directions on GPU (device)
TEST(CameraTest, GetRayDirectionsDevice) {
    // Create a small camera intrinsics
    Intrinsics intrinsics{4, 6, 100.0f, 100.0f, 3.0f, 2.0f};

    // Create Array2D buffer on device
    thrust::device_vector<Vec3D> data(intrinsics.height * intrinsics.width);
    Array2D<Vec3D, DeviceType::GPU> ray_buffer(data.data(), intrinsics.height, intrinsics.width);

    // Launch kernel with multiple threads -- divide into 2x2 tiles
    test_camera_gpu::
        get_ray_directions_kernel<<<1, dim3(intrinsics.height / 2, intrinsics.width / 2)>>>(
            intrinsics, ray_buffer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy data back to host for sanity check
    thrust::host_vector<Vec3D> ray_data = data;

    // Sanity check: adjacent rays should be different
    constexpr float eps = 1e-6f;
    for (auto i = 0; i < data.size() - 1; ++i) {
        auto diff = ray_data[i + 1] - ray_data[i];
        float diff_mag = sqrtf(dot(diff, diff));
        EXPECT_GT(diff_mag, eps) << "Adjacent rays are too similar at index " << i;
    }
}

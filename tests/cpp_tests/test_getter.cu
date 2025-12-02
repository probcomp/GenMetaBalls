#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>
#include <vector>

#include "core/fmb.cuh"
#include "core/geometry.cuh"
#include "core/getter.cuh"
#include "core/utils.cuh"

TEST(AllGetterTest, AllGetterHostTest) {
    constexpr uint32_t num_fmbs = 40;
    FMBScene<MemoryLocation::HOST> scene(num_fmbs);

    Pose extr;
    AllGetter<MemoryLocation::HOST> getter(scene, extr);

    // Create test rays
    std::vector<Ray> rays = {
        Ray{Vec3D{0.0f, 0.0f, 0.0f}, Vec3D{1.0f, 0.0f, 0.0f}},
        Ray{Vec3D{1.0f, 1.0f, 1.0f}, Vec3D{0.0f, 1.0f, 0.0f}},
        Ray{Vec3D{-1.0f, -1.0f, -1.0f}, Vec3D{0.0f, 0.0f, 1.0f}},
        Ray{Vec3D{2.5f, -3.1f, 0.2f}, Vec3D{-0.5f, 0.6f, 0.0f}},
        Ray{Vec3D{4.4f, 0.0f, -0.9f}, Vec3D{0.3f, -0.2f, 1.0f}},
        Ray{Vec3D{5.0f, 2.2f, 1.1f}, Vec3D{-1.0f, 2.0f, 0.2f}},
        Ray{Vec3D{0.0f, 7.0f, 6.0f}, Vec3D{0.0f, -1.0f, -1.0f}},
        Ray{Vec3D{-2.0f, 0.0f, 0.0f}, Vec3D{0.2f, 1.1f, 0.7f}},
        Ray{Vec3D{9.1f, -0.3f, 2.7f}, Vec3D{-0.3f, 0.1f, 0.0f}},
        Ray{Vec3D{1.2f, 8.8f, -4.5f}, Vec3D{1.0f, 0.0f, 1.0f}},
    };

    // Get reference to all FMBs from the original FMBs object

    // Test on host - AllGetter should return the same container for all rays
    for (const auto& ray : rays) {
        const auto& fmbs_returned = getter.get_metaballs(ray);

        // Verify that we get the same container reference (both are host objects)
        EXPECT_EQ(&fmbs_returned, &scene)
            << "AllGetter should return the same FMBs container for all rays";

        // Verify container sizes match
        EXPECT_EQ(fmbs_returned.size(), scene.size())
            << "Returned FMBs container size must match all_fmbs size";
    }
}

__global__ void test_get_metaballs_kernel_device(const AllGetter<MemoryLocation::DEVICE> fmb_getter,
                                                 const Ray* rays, int num_rays, int* out_sizes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto& fmbs_returned = fmb_getter.get_metaballs(rays[idx]);
    out_sizes[idx] = static_cast<int>(fmbs_returned.size());
}

TEST(AllGetterTest, AllGetterDeviceTest) {
    constexpr uint32_t num_fmbs = 40;
    FMBScene<MemoryLocation::DEVICE> device_scene(num_fmbs);
    Pose extr;

    AllGetter<MemoryLocation::DEVICE> getter(device_scene, extr);

    // Create test rays
    std::vector<Ray> rays = {
        Ray{Vec3D{0.0f, 0.0f, 0.0f}, Vec3D{1.0f, 0.0f, 0.0f}},
        Ray{Vec3D{1.0f, 1.0f, 1.0f}, Vec3D{0.0f, 1.0f, 0.0f}},
        Ray{Vec3D{-1.0f, -1.0f, -1.0f}, Vec3D{0.0f, 0.0f, 1.0f}},
        Ray{Vec3D{2.5f, -3.1f, 0.2f}, Vec3D{-0.5f, 0.6f, 0.0f}},
        Ray{Vec3D{4.4f, 0.0f, -0.9f}, Vec3D{0.3f, -0.2f, 1.0f}},
        Ray{Vec3D{5.0f, 2.2f, 1.1f}, Vec3D{-1.0f, 2.0f, 0.2f}},
        Ray{Vec3D{0.0f, 7.0f, 6.0f}, Vec3D{0.0f, -1.0f, -1.0f}},
        Ray{Vec3D{-2.0f, 0.0f, 0.0f}, Vec3D{0.2f, 1.1f, 0.7f}},
        Ray{Vec3D{9.1f, -0.3f, 2.7f}, Vec3D{-0.3f, 0.1f, 0.0f}},
        Ray{Vec3D{1.2f, 8.8f, -4.5f}, Vec3D{1.0f, 0.0f, 1.0f}},
    };

    // Test on GPU for device containers
    int num_rays = static_cast<int>(rays.size());
    thrust::device_vector<int> device_sizes(num_rays);

    // Launch kernel that constructs getter and calls get_metaballs on the device
    test_get_metaballs_kernel_device<<<1, num_rays>>>(
        getter, thrust::raw_pointer_cast(rays.data()), num_rays,
        thrust::raw_pointer_cast(device_sizes.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::host_vector<int> host_sizes = device_sizes;

    // Verify that get_metaballs returns the correct size for all rays
    for (int i = 0; i < num_rays; ++i) {
        EXPECT_EQ(static_cast<size_t>(host_sizes[i]), device_scene.size())
            << "Device get_metaballs returned correct size for ray " << i;
    }
}

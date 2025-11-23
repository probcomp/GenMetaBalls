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

// a whacky helper to get the container type from the template parameter
template <typename Container>
struct template_container_of;

template <template <typename...> class Template, typename... Args>
struct template_container_of<Template<Args...>> {
    template <typename T>
    using type = Template<T>;
};

template <typename InstantiatedContainer>
struct GetterTestTypes;

template <template <typename...> class Template, typename... Args>
struct GetterTestTypes<Template<Args...>> {
    template <typename T>
    using ContainerTemplate = typename template_container_of<Template<Args...>>::template type<T>;

    using FMBsType = FMBs<ContainerTemplate>;
    using AllGetterType = AllGetter<ContainerTemplate>;
};

// Template fixture for all container types
template <typename Container>
class AllGetterTestFixture : public ::testing::Test {};

using ContainerTypes = ::testing::Types<std::vector<FMB>, thrust::device_vector<FMB>>;
TYPED_TEST_SUITE(AllGetterTestFixture, ContainerTypes);

// CUDA kernel that constructs AllGetter and calls get_metaballs
template <typename AllGetterType, typename FMBsType>
__global__ void test_get_metaballs_kernel_device(const FMBsType* fmbs, const Pose* extr,
                                                 const Ray* rays, int num_rays, int* out_sizes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    AllGetterType getter(*fmbs, *extr);
    const auto& fmbs_returned = getter.get_metaballs(rays[idx]);
    out_sizes[idx] = static_cast<int>(fmbs_returned.size());
}

TYPED_TEST(AllGetterTestFixture, ReturnsAllFMBsForAnyRay) {
    // Extract types from TypeParam (std::vector<FMB> or thrust::device_vector<FMB>)
    using FMBsType = typename GetterTestTypes<TypeParam>::FMBsType;
    using AllGetterType = typename GetterTestTypes<TypeParam>::AllGetterType;

    constexpr uint32_t num_fmbs = 40;
    FMBsType fmbs(num_fmbs);

    Pose extr = Pose();
    AllGetterType getter(fmbs, extr);

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
    const auto& all_fmbs_ref = fmbs.get_all_fmbs();

    // Test on host - AllGetter should return the same container for all rays
    for (const auto& ray : rays) {
        const auto& fmbs_returned = getter.get_metaballs(ray);

        // Verify that we get the same container reference (both are host objects)
        EXPECT_EQ(&fmbs_returned, &all_fmbs_ref)
            << "AllGetter should return the same FMBs container for all rays";

        // For thrust::device_vector, also verify device pointers match
        if constexpr (std::is_same_v<TypeParam, thrust::device_vector<FMB>>) {
            EXPECT_EQ(thrust::raw_pointer_cast(fmbs_returned.data()),
                      thrust::raw_pointer_cast(all_fmbs_ref.data()))
                << "Device pointers should match for thrust::device_vector";
        }

        // Verify container sizes match
        EXPECT_EQ(fmbs_returned.size(), all_fmbs_ref.size())
            << "Returned FMBs container size must match all_fmbs size";
    }

    // Test on GPU for device containers
    // Construct AllGetter on device and call get_metaballs
    if constexpr (std::is_same_v<TypeParam, thrust::device_vector<FMB>>) {
        Ray* d_rays = nullptr;
        FMBsType* d_fmbs = nullptr;
        Pose* d_extr = nullptr;
        int* d_sizes = nullptr;
        int num_rays = static_cast<int>(rays.size());

        CUDA_CHECK(cudaMalloc(&d_rays, num_rays * sizeof(Ray)));
        CUDA_CHECK(cudaMalloc(&d_fmbs, sizeof(FMBsType)));
        CUDA_CHECK(cudaMalloc(&d_extr, sizeof(Pose)));
        CUDA_CHECK(cudaMalloc(&d_sizes, num_rays * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_rays, rays.data(), num_rays * sizeof(Ray), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fmbs, &fmbs, sizeof(FMBsType), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_extr, &extr, sizeof(Pose), cudaMemcpyHostToDevice));

        // Launch kernel that constructs getter and calls get_metaballs on the device
        test_get_metaballs_kernel_device<AllGetterType, FMBsType>
            <<<1, num_rays>>>(d_fmbs, d_extr, d_rays, num_rays, d_sizes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> host_sizes(num_rays);
        CUDA_CHECK(
            cudaMemcpy(host_sizes.data(), d_sizes, num_rays * sizeof(int), cudaMemcpyDeviceToHost));

        // Verify that get_metaballs returns the correct size for all rays
        for (int i = 0; i < num_rays; ++i) {
            EXPECT_EQ(static_cast<size_t>(host_sizes[i]), all_fmbs_ref.size())
                << "Device get_metaballs returned correct size for ray " << i;
        }

        CUDA_CHECK(cudaFree(d_rays));
        CUDA_CHECK(cudaFree(d_fmbs));
        CUDA_CHECK(cudaFree(d_extr));
        CUDA_CHECK(cudaFree(d_sizes));
    }
}

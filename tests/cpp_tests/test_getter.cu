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

// Helper to extract template template parameter and provide types for TYPED_TEST
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

// CUDA kernel to test get_metaballs on device
// For device vectors, we pass the raw device pointer and size directly
// since we cannot pass AllGetter with thrust::device_vector references to device code
__global__ void test_get_metaballs_kernel_device(const FMB* fmbs_data, uint32_t fmbs_size,
                                                 const Ray* rays, int num_rays, int* out_sizes,
                                                 void** out_ptrs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < num_rays; ++i) {
            // AllGetter returns all FMBs, so we just verify the pointer and size
            out_sizes[i] = static_cast<int>(fmbs_size);
            // Store pointer value for comparison (we only read, so const_cast is safe here)
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            out_ptrs[i] = const_cast<void*>(static_cast<const void*>(fmbs_data));
        }
    }
}

TYPED_TEST(AllGetterTestFixture, ReturnsAllFMBsForAnyRay) {
    // Extract types from TypeParam (std::vector<FMB> or thrust::device_vector<FMB>)
    using FMBsType = typename GetterTestTypes<TypeParam>::FMBsType;
    using AllGetterType = typename GetterTestTypes<TypeParam>::AllGetterType;

    constexpr uint32_t num_fmbs = 5;
    FMBsType fmbs(num_fmbs);

    Pose extr = Pose();
    AllGetterType getter(fmbs, extr);

    // Create test rays
    std::vector<Ray> rays = {
        Ray{Vec3D{0.0f, 0.0f, 0.0f}, Vec3D{1.0f, 0.0f, 0.0f}},
        Ray{Vec3D{1.0f, 1.0f, 1.0f}, Vec3D{0.0f, 1.0f, 0.0f}},
        Ray{Vec3D{-1.0f, -1.0f, -1.0f}, Vec3D{0.0f, 0.0f, 1.0f}},
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
    // Note: We cannot pass AllGetter with thrust::device_vector references to device code
    // because thrust::device_vector is a host-side object. Instead, we pass the raw device
    // pointer and verify that AllGetter would return the same data.
    if constexpr (std::is_same_v<TypeParam, thrust::device_vector<FMB>>) {
        Ray* d_rays = nullptr;
        int* d_sizes = nullptr;
        void** d_ptrs = nullptr;
        int num_rays = static_cast<int>(rays.size());
        CUDA_CHECK(cudaMalloc(&d_rays, num_rays * sizeof(Ray)));
        CUDA_CHECK(cudaMalloc(&d_sizes, num_rays * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ptrs, num_rays * sizeof(void*)));
        CUDA_CHECK(cudaMemcpy(d_rays, rays.data(), num_rays * sizeof(Ray), cudaMemcpyHostToDevice));

        // Get the raw device pointer and size from the FMBs container
        const FMB* fmbs_data = thrust::raw_pointer_cast(all_fmbs_ref.data());
        auto fmbs_size = static_cast<uint32_t>(all_fmbs_ref.size());

        // Launch kernel with raw device pointer instead of getter object
        test_get_metaballs_kernel_device<<<1, 1>>>(fmbs_data, fmbs_size, d_rays, num_rays, d_sizes,
                                                   d_ptrs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> host_sizes(num_rays);
        std::vector<void*> host_ptrs(num_rays);
        CUDA_CHECK(
            cudaMemcpy(host_sizes.data(), d_sizes, num_rays * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(
            cudaMemcpy(host_ptrs.data(), d_ptrs, num_rays * sizeof(void*), cudaMemcpyDeviceToHost));

        // Get reference device pointer on host for comparison
        const void* ref_ptr = static_cast<const void*>(fmbs_data);

        for (int i = 0; i < num_rays; ++i) {
            EXPECT_EQ(static_cast<size_t>(host_sizes[i]), all_fmbs_ref.size())
                << "Device kernel returned correct size for ray " << i;
            EXPECT_EQ(host_ptrs[i], ref_ptr)
                << "Device kernel should return same device pointer for ray " << i;
        }

        CUDA_CHECK(cudaFree(d_rays));
        CUDA_CHECK(cudaFree(d_sizes));
        CUDA_CHECK(cudaFree(d_ptrs));
    }
}

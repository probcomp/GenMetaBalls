#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "core/forward.cuh"
#include "core/utils.cuh"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

namespace get_pixel_coords_tests {
// A simple kernel that fills an Array2D with 1.0f in parallel
__global__ void fill_with_ones_kernel(Array2D<float, MemoryLocation::DEVICE> output,
                                      const Intrinsics& intr) {
    auto pixel_coords = get_pixel_coords(threadIdx, blockIdx, blockDim, gridDim, intr);
    for (const auto [px, py] : pixel_coords) {
        output[py][px] = 1.0f;
    }
}
} // namespace get_pixel_coords_tests

// Test if fmb::get_pixel_coords correctly covers all image pixels
TEST(ForwardTest, GetPixelCoordsCoverage) {
    const auto intrinsic =
        Intrinsics{.height = 100, .width = 200, .fx = 1.0f, .fy = 1.0f, .cx = 50.0f, .cy = 100.0f};
    auto buffer = thrust::device_vector<float>(intrinsic.height * intrinsic.width, 0.0f);
    auto array2d =
        Array2D<float, MemoryLocation::DEVICE>(buffer.data(), intrinsic.height, intrinsic.width);
    constexpr dim3 block_dim(12, 8);
    constexpr dim3 grid_dim(16, 24);
    get_pixel_coords_tests::fill_with_ones_kernel<<<grid_dim, block_dim>>>(array2d, intrinsic);
    auto host_buffer = thrust::host_vector<float>(buffer);
    for (size_t i = 0; i < host_buffer.size(); ++i) {
        EXPECT_EQ(host_buffer[i], 1.0f);
    }
}

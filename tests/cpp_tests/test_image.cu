#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include "core/image.cuh"

namespace test_image_gpu {

/* A simple kernel that changes the image data based on pixel coordinate
 * Note that while we allocate the image with "Image" type, we pass it to the kernel as "ImageView"
 * via as_view() method.
 */
__global__ void manipulate_image_kernel(ImageView<MemoryLocation::DEVICE> img) {
    uint32_t row = threadIdx.y;
    uint32_t col = threadIdx.x;

    if (row < img.num_rows() && col < img.num_cols()) {
        img.confidence[row][col] = static_cast<float>(row);
        img.depth[row][col] = static_cast<float>(col);
    }
}

} // namespace test_image_gpu

TEST(TestImage, ImageCreationHost) {
    constexpr uint32_t height = 128;
    constexpr uint32_t width = 256;

    // Create an image in host memory
    Image<MemoryLocation::HOST> img_buffer(height, width);
    auto img = img_buffer.as_view();

    // Check dimensions
    EXPECT_EQ(img.num_rows(), height);
    EXPECT_EQ(img.num_cols(), width);

    // Check that confidence and depth are initialized to zero
    for (uint32_t r = 0; r < height; ++r) {
        for (uint32_t c = 0; c < width; ++c) {
            EXPECT_FLOAT_EQ(img.confidence[r][c], 0.0f);
            EXPECT_FLOAT_EQ(img.depth[r][c], 0.0f);
        }
    }
}

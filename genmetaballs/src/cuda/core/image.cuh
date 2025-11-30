#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils.cuh"

/* Non-owning view into an image */
template <MemoryLocation location>
class ImageView {
public:
    Array2D<float, location> confidence;
    Array2D<float, location> depth;

    CUDA_CALLABLE auto num_rows() const noexcept {
        return confidence.num_rows();
    }
    CUDA_CALLABLE auto num_cols() const noexcept {
        return confidence.num_cols();
    }
};

/* The image buffer which handles the allocation & deallocation of memory with RAII.
 * While the underlying data may be stored in either host or device memory, this
 * class is always managed from the host side.
 */
template <MemoryLocation location>
class Image {
private:
    // Host memory -> thrust::host_vector
    // Device memory -> thrust::device_vector
    template <typename T>
    using vector_t = std::conditional_t<location == MemoryLocation::HOST, thrust::host_vector<T>,
                                        thrust::device_vector<T>>;

    // RAII storage for the image data
    vector_t<float> confidence_data_;
    vector_t<float> depth_data_;

    uint32_t height_;
    uint32_t width_;

public:
    /* Allocate the memory for a new image & default initialize with zeros
     * The "height" of the image correponds to "rows" in the 2D array, whereas
     * the "width" of the image corresponds to "columns".
     */
    __host__ Image(uint32_t height, uint32_t width)
        : height_(height), width_(width), confidence_data_(height * width),
          depth_data_(height * width) {}

    /* Create a view of this image which points to the internal data */
    CUDA_CALLABLE ImageView<location> as_view() {
        return ImageView<location>{
            Array2D<float, location>(confidence_data_.data(), height_, width_),
            Array2D<float, location>(depth_data_.data(), height_, width_)};
    }

    CUDA_CALLABLE auto num_rows() const noexcept {
        return height_;
    }
    CUDA_CALLABLE auto num_cols() const noexcept {
        return width_;
    }
};

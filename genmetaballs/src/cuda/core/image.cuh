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

    CUDA_CALLABLE constexpr auto num_rows() const noexcept {
        return confidence.num_rows();
    }
    CUDA_CALLABLE constexpr auto num_cols() const noexcept {
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

    const uint32_t height_;
    const uint32_t width_;

    // Make all Image instantiations friends so they can access each other's private members
    template <MemoryLocation other_location>
    friend class Image;

public:
    /* Allocate the memory for a new image & default initialize with zeros
     * The "height" of the image correponds to "rows" in the 2D array, whereas
     * the "width" of the image corresponds to "columns".
     */
    __host__ Image(uint32_t height, uint32_t width)
        : height_(height), width_(width), confidence_data_(height * width),
          depth_data_(height * width) {}

    /* Copy constructor from a Image which may reside in a different memory location */
    template <MemoryLocation other_location>
    __host__ Image(const Image<other_location>& other)
        : height_(other.num_rows()), width_(other.num_cols()),
          confidence_data_(other.confidence_data_), depth_data_(other.depth_data_) {}

    /* Create a view of this image which points to the internal data */
    CUDA_CALLABLE auto as_view() {
        return ImageView<location>{{confidence_data_.data(), height_, width_},
                                   {depth_data_.data(), height_, width_}};
    }

    CUDA_CALLABLE constexpr auto num_rows() const noexcept {
        return height_;
    }
    CUDA_CALLABLE constexpr auto num_cols() const noexcept {
        return width_;
    }
};

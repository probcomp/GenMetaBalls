#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils.cuh"

/* Non-owning view into temporary buffers for FMB chunk processing */
template <MemoryLocation location>
class TempBufferView {
private:
    float* data_ptr_;
    uint32_t num_fmb_chunks_;
    uint32_t height_;
    uint32_t width_;

public:
    // Default constructor (needed for bindings)
    CUDA_CALLABLE TempBufferView()
        : data_ptr_(nullptr), num_fmb_chunks_(0), height_(0), width_(0) {}

    CUDA_CALLABLE TempBufferView(float* data, uint32_t num_chunks, uint32_t height, uint32_t width)
        : data_ptr_(data), num_fmb_chunks_(num_chunks), height_(height), width_(width) {}

    // Get buffer index for a specific chunk and type
    // type: 0=depth_numer, 1=depth_denom, 2=conf_tmp
    CUDA_CALLABLE constexpr uint32_t get_buffer_idx(uint32_t fmb_chunk_idx, uint32_t type) const {
        return fmb_chunk_idx * 3 + type;
    }

    // Get pointer offset for a specific buffer
    CUDA_CALLABLE constexpr float* get_buffer_ptr(uint32_t fmb_chunk_idx, uint32_t type) const {
        uint32_t idx = get_buffer_idx(fmb_chunk_idx, type);
        return data_ptr_ + idx * height_ * width_;
    }

    // Access a specific buffer as Array2D
    CUDA_CALLABLE constexpr Array2D<float, location> get_buffer(uint32_t fmb_chunk_idx,
                                                                uint32_t type) const {
        return Array2D<float, location>(get_buffer_ptr(fmb_chunk_idx, type), height_, width_);
    }

    CUDA_CALLABLE constexpr auto num_rows() const noexcept {
        return height_;
    }

    CUDA_CALLABLE constexpr auto num_cols() const noexcept {
        return width_;
    }

    CUDA_CALLABLE constexpr auto num_fmb_chunks() const noexcept {
        return num_fmb_chunks_;
    }
};

/* The temporary buffer which handles the allocation & deallocation of memory with RAII.
 * Stores num_fmb_chunks * 3 buffers (depth_numer, depth_denom, conf_tmp for each chunk).
 * While the underlying data may be stored in either host or device memory, this
 * class is always managed from the host side.
 */
template <MemoryLocation location>
class TempBuffer {
private:
    // Host memory -> thrust::host_vector
    // Device memory -> thrust::device_vector
    template <typename T>
    using vector_t = std::conditional_t<location == MemoryLocation::HOST, thrust::host_vector<T>,
                                        thrust::device_vector<T>>;

    // RAII storage for all buffer data
    vector_t<float> buffer_data_;

    const uint32_t height_;
    const uint32_t width_;
    const uint32_t num_fmb_chunks_;

    // Make all TempBuffer instantiations friends so they can access each other's private members
    template <MemoryLocation other_location>
    friend class TempBuffer;

public:
    /* Allocate the memory for temporary buffers & default initialize with zeros
     * The "height" of the buffer corresponds to "rows" in the 2D array, whereas
     * the "width" of the buffer corresponds to "columns".
     */
    __host__ TempBuffer(uint32_t height, uint32_t width, uint32_t num_fmb_chunks)
        : height_(height), width_(width), num_fmb_chunks_(num_fmb_chunks),
          buffer_data_(num_fmb_chunks * 3 * height * width) {}

    /* Copy constructor from a TempBuffer which may reside in a different memory location */
    template <MemoryLocation other_location>
    __host__ TempBuffer(const TempBuffer<other_location>& other)
        : height_(other.num_rows()), width_(other.num_cols()),
          num_fmb_chunks_(other.num_fmb_chunks_), buffer_data_(other.buffer_data_) {}

    /* Create a view of this temp buffer which points to the internal data */
    __host__ auto as_view() {
        // as_view() is called from host code, so we can safely use raw_pointer_cast
        return TempBufferView<location>(thrust::raw_pointer_cast(buffer_data_.data()),
                                        num_fmb_chunks_, height_, width_);
    }

    CUDA_CALLABLE constexpr auto num_rows() const noexcept {
        return height_;
    }

    CUDA_CALLABLE constexpr auto num_cols() const noexcept {
        return width_;
    }

    CUDA_CALLABLE constexpr auto num_fmb_chunks() const noexcept {
        return num_fmb_chunks_;
    }
};

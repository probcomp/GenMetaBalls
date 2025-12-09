#include <cstdint>
#include <cuda_runtime.h>

#include "core/camera.cuh"
#include "core/kernels/kernel1.cuh"
#include "core/temp_buffer.cuh"
#include "core/utils.cuh"

// Simple sum operation for parallel reduction
struct SumOp {
    using Data = float;

    static __device__ __forceinline__ Data identity() {
        return 0.0f;
    }

    static __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }
};

// Kernel 1b: Reduce across FMB chunks using parallel sum (inspired by scan implementation)
// Launch: 3D block where z-dimension = num_fmb_chunks
// Each thread loads one chunk's value, then we use parallel sum across z-dimension
__global__ void render_kernel_fmb_reduce(TempBufferView<MemoryLocation::DEVICE> temp_buffers,
                                         uint32_t num_fmb_chunks, const Intrinsics& intr) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    const int chunk_idx = threadIdx.z; // Chunk index from z-dimension
    const int pixel_in_block = threadIdx.y * blockDim.x + threadIdx.x; // Unique pixel ID in block

    if (px >= intr.width || py >= intr.height || chunk_idx >= (int)num_fmb_chunks)
        return;

    const int img_row = intr.height - py - 1;
    const uint32_t pixel_offset = img_row * intr.width + px;

    // Load one chunk's value for this pixel
    float depth_numer = temp_buffers.get_buffer_ptr(chunk_idx, 0)[pixel_offset];
    float depth_denom = temp_buffers.get_buffer_ptr(chunk_idx, 1)[pixel_offset];
    float conf_tmp = temp_buffers.get_buffer_ptr(chunk_idx, 2)[pixel_offset];

    // Use shared memory for parallel reduction across z-dimension (chunks)
    // Shared memory layout: [pixel_in_block][chunk_idx]
    extern __shared__ __align__(16) char shmem_raw[];
    float* shmem_numer = reinterpret_cast<float*>(shmem_raw);
    float* shmem_denom = shmem_numer + blockDim.x * blockDim.y * blockDim.z;
    float* shmem_conf = shmem_denom + blockDim.x * blockDim.y * blockDim.z;

    const int shmem_idx = pixel_in_block * blockDim.z + chunk_idx;

    // Store values in shared memory
    shmem_numer[shmem_idx] = depth_numer;
    shmem_denom[shmem_idx] = depth_denom;
    shmem_conf[shmem_idx] = conf_tmp;
    __syncthreads();

    // Parallel sum across chunks for this pixel using reduction pattern
    // Reduce across z-dimension: threads with same pixel_in_block but different chunk_idx
    float val_numer = depth_numer;
    float val_denom = depth_denom;
    float val_conf = conf_tmp;

    // Reduction phase: build sum tree (inspired by scan up-sweep)
    for (uint32_t delta = 1; delta < blockDim.z; delta <<= 1) {
        if (chunk_idx >= delta) {
            float partial_numer = shmem_numer[pixel_in_block * blockDim.z + chunk_idx - delta];
            float partial_denom = shmem_denom[pixel_in_block * blockDim.z + chunk_idx - delta];
            float partial_conf = shmem_conf[pixel_in_block * blockDim.z + chunk_idx - delta];

            val_numer = SumOp::combine(partial_numer, val_numer);
            val_denom = SumOp::combine(partial_denom, val_denom);
            val_conf = SumOp::combine(partial_conf, val_conf);
        }
        __syncthreads();

        if (chunk_idx >= delta) {
            shmem_numer[shmem_idx] = val_numer;
            shmem_denom[shmem_idx] = val_denom;
            shmem_conf[shmem_idx] = val_conf;
        }
        __syncthreads();
    }

    // After reduction, thread with highest chunk_idx has the total sum
    // We write it to chunk 0's buffer (overwriting it)
    if (chunk_idx == blockDim.z - 1) {
        temp_buffers.get_buffer_ptr(0, 0)[pixel_offset] = val_numer;
        temp_buffers.get_buffer_ptr(0, 1)[pixel_offset] = val_denom;
        temp_buffers.get_buffer_ptr(0, 2)[pixel_offset] = val_conf;
    }
}

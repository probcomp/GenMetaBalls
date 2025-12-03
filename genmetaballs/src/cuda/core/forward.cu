#include <cstdint>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "fmb.cuh"
#include "geometry.cuh"
#include "image.cuh"

// TODO: tune this number
constexpr auto NUM_BLOCKS = dim3(4, 4);
constexpr auto THREADS_PER_BLOCK = dim3(16, 16);

namespace fmb {

CUDA_CALLABLE PixelCoordRange get_pixel_coords(const dim3 thread_idx, const dim3 block_idx,
                                               const dim3 block_dim, const dim3 grid_dim,
                                               const Intrinsics& intr) {
    // compute the number of pixels each thread should process
    const auto num_pixels_x = int_ceil_div(intr.height, grid_dim.x * block_dim.x);
    const auto num_pixels_y = int_ceil_div(intr.width, grid_dim.y * block_dim.y);
    const auto start_x = (block_idx.x * block_dim.x + thread_idx.x) * num_pixels_x;
    const auto start_y = (block_idx.y * block_dim.y + thread_idx.y) * num_pixels_y;
    return PixelCoordRange{.px_start = start_x,
                           .px_end = min(start_x + num_pixels_x, intr.height),
                           .py_start = start_y,
                           .py_end = min(start_y + num_pixels_y, intr.width)};
}

template <typename Getter, typename Intersector, typename Blender, typename Confidence>
__global__ void render_kernel(const Getter fmb_getter, const Blender blender,
                              Confidence const* confidence, Intrinsics const intr, Pose const* extr,
                              ImageView<MemoryLocation::DEVICE> img) {
    auto pixel_coords = get_pixel_coords(threadIdx, blockIdx, blockDim, gridDim, intr);

    for (const auto& [px, py] : pixel_coords) {
        float w0 = 0.0f, tf = 0.0f, sumexpd = 0.0f;
        auto ray = intr.get_ray_direction(px, py);
        for (const auto& fmb : fmb_getter->get_metaballs(ray)) {
            const auto& [t, d] = Intersector::intersect(fmb, ray, extr);
            auto w = blender->blend(t, d, fmb, ray);
            sumexpd += exp(d); // numerically unstable. use logsumexp
            tf += t;
            w0 += w;
        }
        img.confidence[px][py] = confidence->get_confidence(sumexpd);
        img.depth[px][py] = tf / w0;
    }
}

template <typename Getter, typename Intersector, typename Blender, typename Confidence>
void render_fmbs(const FMBScene<MemoryLocation::DEVICE>& fmbs, const Intrinsics& intr,
                 const Pose& extr) {
    // initialize the fmb_getter
    auto fmb_getter = Getter(fmbs, extr);
    auto& kernel = render_kernel<Getter, Intersector, Blender, Confidence>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(fmb_getter, fmbs, intr, extr);
}

}; // namespace fmb

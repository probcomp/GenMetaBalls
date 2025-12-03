#include <cstdint>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "fmb.cuh"
#include "geometry.cuh"
#include "image.cuh"

constexpr auto NUM_BLOCKS = dim3(10); // XXX madeup
constexpr auto THREADS_PER_BLOCK = dim3(10);

namespace fmb {

CUDA_CALLABLE auto get_pixel_coords(const dim3 thread_idx, const dim3 block_idx,
                                    const dim3 block_dim, const dim3 grid_dim,
                                    Intrinsics const* intr, Pose const* extr) {
    std::vector<PixelCoord> res;

    uint32_t i_beg = 0; // XXX TODO
    uint32_t i_end = 0; // XXX TODO

    for (int i = i_beg; i < i_end; i += blockDim.x) {
        //...
    }

    return res;
}

template <typename Getter, typename Intersector, typename Blender, typename Confidence>
__global__ void render_kernel(const Getter fmb_getter, const Blender blender,
                              Confidence const* confidence, Intrinsics const* intr,
                              Pose const* extr, ImageView<MemoryLocation::DEVICE> img) {
    auto pixel_coords = get_pixel_coords(threadIdx, blockIdx, blockDim, gridDim, intr, extr);

    for (const auto& [px, py] : pixel_coords) {
        float w0 = 0.0f, tf = 0.0f, sumexpd = 0.0f;
        auto ray = intr->get_ray_direction(px, py);
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
    typename Getter::Getter fmb_getter(fmbs, extr);
    auto& kernel = render_kernel<Getter, Intersector, Blender, Confidence>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(fmb_getter, fmbs, intr, extr);
}

}; // namespace fmb

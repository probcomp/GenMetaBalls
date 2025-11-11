#include <cstdint>

#include <cuda_runtime.h>

constexpr NUM_BLOCKS dim3(10); //XXX madeup
constexpr THREADS_PER_BLOCK dim3(10);

namespace FMB {

__device__ __host__
std::vector<std::pair<PixelCoord, Ray>>
get_pixel_coords_and_rays(const dim3 thread_idx, const dim3 block_idx)
{
    std::vector<std::pair<PixelCoord, Ray>> res;

    uint32_t i_beg = 0; // XXX TODO
    uint32_t i_end = 0; // XXX TODO

    for(int i = i_beg; i < i_end; i+=blockDim.x) {
        //...
    }

    return res;
}

template<class Getter, class Intersector, class Blender, class Confidence>
__global__
render_kernel(
    const typename Getter::Getter &fmb_getter,
    const Intrinsics &intr,
    const Pose &extr,
    Image *img
) {
    // TODO how to find the relevant chunk of computation from threadIdx,
    // blockIdx, etc
    auto pixel_coords_and_rays = get_pixel_coords_and_rays(threadIdx, blockIdx, ...);

    for(const auto &[pixel_coords, ray]: pixel_coords_and_rays) {
        float w0 = 0.0f, tf = 0.0f, confidence = 0.0f;
        for(const auto &fmb: fmb_getter.get_metaballs(ray)) {
             t = Intersector::intersect(fmb, ray);
             w = Blender::blend(t, fmb, ray);
             confidence = Confidence::update(confidence, t, w);
             tf += t;
             w0 += w;
        }
        img.confidence.at(pixel_coords) = confidence;
        img.depth.at(pixel_coords) = tf / w0;
    }
}

template<class Getter, class Intersector, class Blender, class Confidence>
void
render_fmbs(const FMBs &fmbs, const Intrinsics &intr, const Pose &extr)
{
     // initialize the fmb_getter
    typename Getter::Getter fmb_getter(fmbs, intr, extr);
    auto kernel = render_kernel<Getter, Intersector, Blender, Confidence>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(fmb_getter, fmbs, intr, extr);
}

};

#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "fmb.cuh"
#include "geometry.cuh"
#include "utils.cuh"

// This is the dummy version of getter, where all FMBs are relevant to any ray
template <typename containter_template>
struct AllGetter {
    FMBs<containter_template> fmbs;
    Pose extr; // Current assumption: rays are in camera frame

    // It does not bother using the ray, because it simply returns all FMBs
    CUDA_CALLABLE const containter_template<FMB>& get_metaballs(const Ray& ray) const {
        return fmbs.get_all_fmbs();
    }
};

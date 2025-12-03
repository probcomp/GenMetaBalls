#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "camera.cuh"
#include "fmb.cuh"
#include "geometry.cuh"
#include "utils.cuh"

// This is the dummy version of getter, where all FMBs are relevant to any ray
template <MemoryLocation location>
struct AllGetter {
    const FMBScene<location>& scene;
    const Pose& extr; // Current assumption: rays are in camera frame

    CUDA_CALLABLE AllGetter(const FMBScene<location>& scene, const Pose& extr)
        : scene(scene), extr(extr) {}

    // It does not bother using the ray, because it simply returns all FMBs
    CUDA_CALLABLE const FMBScene<location>& get_metaballs(const Vec3D& ray) const {
        return scene;
    }
};

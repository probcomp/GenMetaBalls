#pragma once

#include <utility>

#include "fmb.h"
#include "geometry.h"

// implement equation (6) in the paper
class LinearIntersector {

    static __device__ __host__
    std::pair<float, float>
    intersect(const FMB &fmb, const Ray &ray) const;
};

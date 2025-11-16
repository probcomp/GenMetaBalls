#pragma once

#include "fmb.h"
#include "geometry.h"


struct ThreeParameterBlender {
    float beta1;
    float beta2;
    float eta;

    __host__ __device__ __forceinline__ // TODO inline?
    float blend(float t, float d, const FMB &fmb, const Ray &ray) const;
};

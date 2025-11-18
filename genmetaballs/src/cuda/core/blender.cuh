#pragma once

#include "fmb.h"
#include "geometry.h"

struct ThreeParameterBlender {
    float beta1;
    float beta2;
    float eta;

    CUDA_CALLABLE __forceinline__ // TODO inline?
        float
        blend(float t, float d, const FMB& fmb, const Ray& ray) const;
};

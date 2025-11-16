#pragma once

#include <cstdint>

struct Intrinsics {
    uint32_t height;
    uint32_t width;
    float fx;
    float fy;
    float cx;
    float cy;
    float near;
    float far;
};

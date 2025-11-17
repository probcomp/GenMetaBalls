#pragma once

#include <cstdint>

#include "geometry.cuh"
#include "utils.cuh"

template <uint32_t width, uint32_t height>
struct Image {
    Array2D<float, width, height> confidence;
    Array2D<float, width, height> depth;
};

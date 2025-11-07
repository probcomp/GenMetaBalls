#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "core/add.cuh"

constexpr uint32_t GRID_DIM = 4096;
constexpr uint32_t BLOCK_DIM = 1024;

namespace nb = nanobind;

NB_MODULE(_genmetaballs_bindings, m) {
    m.def(
        "gpu_add",
        &gpu_add<GRID_DIM, BLOCK_DIM>,
        "Add two lists elementwise on the GPU",
        nb::arg("a"), nb::arg("b")
    );
}

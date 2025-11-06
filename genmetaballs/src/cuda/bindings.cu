//TODO: add whatever is needed for the CUDA bindings to work

#include <cstdint>

#include <nanobind/nanobind.h>

#include "core/add.cuh"

constexpr uint32_t GRID_DIM = 4096;
constexpr uint32_t BLOCK_DIM = 1024;

NB_MODULE(_genmetaballs_bindings, m) {
    m.def(
        "gpu_add",
        &gpu_add<GRID_DIM, BLOCK_DIM>,
        "Add two lists elementwise on the GPU"
    );
}

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "core/math_utils.cuh"

constexpr uint32_t GRID_DIM = 4096;
constexpr uint32_t BLOCK_DIM = 1024;

namespace nb = nanobind;

// Initialize the math_utils submodule (called from main.cu)
void init_math_utils_submodule(nb::module_& m) {
    // Expose sigmoid function for single values
    m.def("sigmoid", sigmoid, nb::arg("x"), "Compute the sigmoid function: 1 / (1 + exp(-x))");

    // GPU version -- accepts Python list, returns vector<float>
    m.def("sigmoid_vector", gpu_sigmoid<GRID_DIM, BLOCK_DIM>, nb::arg("x"),
          "Compute the sigmoid function element-wise for a vector on GPU");
}

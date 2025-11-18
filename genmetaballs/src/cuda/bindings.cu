#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

#include "core/add.cuh"
#include "core/confidence.cuh"
#include "core/geometry.cuh"
#include "core/utils.cuh"

constexpr uint32_t GRID_DIM = 4096;
constexpr uint32_t BLOCK_DIM = 1024;

namespace nb = nanobind;

// Initialize the confidence submodule (called from main module)
void init_confidence_submodule(nb::module_& m) {
    // Note: ZeroParameterConfidence is a type alias for ThreeParameterConfidence,
    // so we only need to register ThreeParameterConfidence to the bindings.
    nb::class_<ThreeParameterConfidence>(m, "ThreeParameterConfidence")
        .def(nb::init<>())
        .def("get_confidence", &ThreeParameterConfidence::get_confidence);

    // I am exposing ZeroParameterConfidence as an alias in Python too
    m.attr("ZeroParameterConfidence") = m.attr("ThreeParameterConfidence");

    nb::class_<FiveParameterConfidence>(m, "FiveParameterConfidence")
        .def(nb::init<float, float>())
        .def("get_confidence", &FiveParameterConfidence::get_confidence);
}

// Initialize the utils submodule (called from main module)
void init_utils_submodule(nb::module_& m) {
    // Expose sigmoid function for single values
    m.def("sigmoid", sigmoid, nb::arg("x"), "Compute the sigmoid function: 1 / (1 + exp(-x))");
}

NB_MODULE(_genmetaballs_bindings, m) {

    // simple add kernel
    m.def("gpu_add", &gpu_add<GRID_DIM, BLOCK_DIM>, "Add two lists elementwise on the GPU",
          nb::arg("a"), nb::arg("b"));

    // exposing Vec3D
    nb::class_<Vec3D>(m, "Vec3D")
        .def(nb::init<>())
        .def(nb::init<float, float, float>())
        .def_rw("x", &Vec3D::x)
        .def_rw("y", &Vec3D::y)
        .def_rw("z", &Vec3D::z)
        .def("__add__", &operator+)
        .def("__sub__", &operator-)
        .def("__repr__",
             [](const Vec3D& v) { return nb::str("Vec3D({}, {}, {})").format(v.x, v.y, v.z); });

    // Create confidence submodule
    nb::module_ confidence_submodule = m.def_submodule("confidence");
    init_confidence_submodule(confidence_submodule);

    // Create utils submodule
    nb::module_ utils_submodule = m.def_submodule("utils");
    init_utils_submodule(utils_submodule);
}

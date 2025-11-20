#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

#include "core/confidence.cuh"
#include "core/geometry.cuh"
#include "core/utils.cuh"

constexpr uint32_t GRID_DIM = 4096;
constexpr uint32_t BLOCK_DIM = 1024;

namespace nb = nanobind;

NB_MODULE(_genmetaballs_bindings, m) {

    // exposing Vec3D
    nb::class_<Vec3D>(m, "Vec3D")
        .def(nb::init<>())
        .def(nb::init<float, float, float>())
        .def_rw("x", &Vec3D::x)
        .def_rw("y", &Vec3D::y)
        .def_rw("z", &Vec3D::z)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def("__repr__",
             [](const Vec3D& v) { return nb::str("Vec3D({}, {}, {})").format(v.x, v.y, v.z); });

    // confidence submodule
    nb::module_ confidence = m.def_submodule("confidence");
    nb::class_<ZeroParameterConfidence>(confidence, "ZeroParameterConfidence")
        .def(nb::init<>())
        .def("get_confidence", &ZeroParameterConfidence::get_confidence);

    nb::class_<TwoParameterConfidence>(confidence, "TwoParameterConfidence")
        .def(nb::init<float, float>())
        .def("get_confidence", &TwoParameterConfidence::get_confidence);

    // utils submodule
    nb::module_ utils = m.def_submodule("utils");
    utils.def("sigmoid", sigmoid, nb::arg("x"), "Compute the sigmoid function: 1 / (1 + exp(-x))");

} // NB_MODULE(_genmetaballs_bindings)

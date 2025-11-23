#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

#include "core/add.cuh"
#include "core/confidence.cuh"
#include "core/geometry.cuh"
#include "core/utils.cuh"

constexpr uint32_t GRID_DIM = 4096;
constexpr uint32_t BLOCK_DIM = 1024;

namespace nb = nanobind;

NB_MODULE(_genmetaballs_bindings, m) {

    // simple add kernel
    m.def("gpu_add", &gpu_add<GRID_DIM, BLOCK_DIM>, "Add two lists elementwise on the GPU",
          nb::arg("a"), nb::arg("b"));

    /*
     * Geometry module bindings
     */

    nb::module_ geometry = m.def_submodule("geometry", "Geometry helpers for GenMetaballs");

    nb::class_<Vec3D>(geometry, "Vec3D")
        .def(nb::init<>())
        .def(nb::init<float, float, float>())
        .def_ro("x", &Vec3D::x)
        .def_ro("y", &Vec3D::y)
        .def_ro("z", &Vec3D::z)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(-nb::self)
        .def(nb::self * float())
        .def(float() * nb::self)
        .def(nb::self / float())
        .def("__repr__",
             [](const Vec3D& v) { return nb::str("Vec3D({}, {}, {})").format(v.x, v.y, v.z); });

    geometry.def("dot", &dot, "Dot product of two `Vec3D`s", nb::arg("a"), nb::arg("b"));
    geometry.def("cross", &cross, "Cross product of two `Vec3D`s", nb::arg("a"), nb::arg("b"));

    nb::class_<Rotation>(geometry, "Rotation")
        .def(nb::init<>())
        .def_static("from_quat", &Rotation::from_quat, "Create rotation from quaternion",
                    nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"))
        .def("apply", &Rotation::apply, "Apply rotation to vector", nb::arg("vec"))
        .def("compose", &Rotation::compose, "Compose with another rotation", nb::arg("rot"))
        .def("inv", &Rotation::inv, "Inverse rotation");

    nb::class_<Pose>(geometry, "Pose")
        .def(nb::init<>())
        .def_static("from_components", &Pose::from_components,
                    "Create rotation from a rotation and a translation", nb::arg("rot"),
                    nb::arg("tran"))
        .def_prop_ro("rot", &Pose::get_rot, "get the rotation component")
        .def_prop_ro("tran", &Pose::get_tran, "get the translation component")
        .def("apply", &Pose::apply, "Apply pose to vector", nb::arg("vec"))
        .def("compose", &Pose::compose, "Compose with another pose", nb::arg("pose"))
        .def("inv", &Pose::inv, "Inverse pose");

    nb::class_<Ray>(geometry, "Ray")
        .def(nb::init<>())
        .def_rw("start", &Ray::start)
        .def_rw("direction", &Ray::direction);

    /*
     * Confidence module bindings
     */

    nb::module_ confidence = m.def_submodule("confidence");
    nb::class_<ZeroParameterConfidence>(confidence, "ZeroParameterConfidence")
        .def(nb::init<>())
        .def("get_confidence", &ZeroParameterConfidence::get_confidence);

    nb::class_<TwoParameterConfidence>(confidence, "TwoParameterConfidence")
        .def(nb::init<float, float>())
        .def("get_confidence", &TwoParameterConfidence::get_confidence);

    /*
     * Utils module bindings
     */

    nb::module_ utils = m.def_submodule("utils");
    utils.def("sigmoid", sigmoid, nb::arg("x"), "Compute the sigmoid function: 1 / (1 + exp(-x))");

} // NB_MODULE(_genmetaballs_bindings)

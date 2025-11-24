#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>

#include "core/confidence.cuh"
#include "core/geometry.cuh"
#include "core/utils.cuh"

namespace nb = nanobind;

NB_MODULE(_genmetaballs_bindings, m) {

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

    nb::class_<Array2D<float>>(utils, "FloatArray2D")
        // TODO: switch to the array_api in future nanobind release
        // https://nanobind.readthedocs.io/en/latest/api_extra.html#_CPPv4N8nanobind9array_apiE
        .def(
            "numpy",
            [](const Array2D<float>& self) {
                return nb::ndarray<float, nb::numpy, nb::c_contig>(
                    self.data(), {self.num_rows(), self.num_cols()});
            },
            nb::rv_policy::reference_internal)
        .def_static("from_array",
                    [](const nb::ndarray<float, nb::ndim<2>, nb::c_contig>& array) {
                        return Array2D<float>(array.data(), array.shape(0), array.shape(1));
                    })
        .def_prop_ro("num_rows", &Array2D<float>::num_rows)
        .def_prop_ro("num_cols", &Array2D<float>::num_cols)
        .def_prop_ro("ndim", &Array2D<float>::ndim)
        .def_prop_ro("size", &Array2D<float>::size);

} // NB_MODULE(_genmetaballs_bindings)

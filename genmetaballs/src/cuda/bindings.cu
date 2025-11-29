#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>

#include "core/blender.cuh"
#include "core/confidence.cuh"
#include "core/geometry.cuh"
#include "core/utils.cuh"

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
        .def("get_confidence", &ZeroParameterConfidence::get_confidence, nb::arg("sumexpd"),
             "Get the confidence value for a given sumexpd")
        .def("__repr__",
             [](const ZeroParameterConfidence& c) { return nb::str("ZeroParameterConfidence()"); });

    nb::class_<TwoParameterConfidence>(confidence, "TwoParameterConfidence")
        .def(nb::init<float, float>())
        .def_ro("beta4", &TwoParameterConfidence::beta4)
        .def_ro("beta5", &TwoParameterConfidence::beta5)
        .def("get_confidence", &TwoParameterConfidence::get_confidence, nb::arg("sumexpd"),
             "Get the confidence value for a given sumexpd")
        .def("__repr__", [](const TwoParameterConfidence& c) {
            return nb::str("TwoParameterConfidence(beta4={}, beta5={})").format(c.beta4, c.beta5);
        });

    // utils submodule
    nb::module_ utils = m.def_submodule("utils");
    utils.def("sigmoid", sigmoid, nb::arg("x"), "Compute the sigmoid function: 1 / (1 + exp(-x))");

    // blender submodule
    nb::module_ blender = m.def_submodule("blender");
    nb::class_<FourParameterBlender>(blender, "FourParameterBlender")
        .def(nb::init<float, float, float, float>())
        .def_ro("beta1", &FourParameterBlender::beta1)
        .def_ro("beta2", &FourParameterBlender::beta2)
        .def_ro("beta3", &FourParameterBlender::beta3)
        .def_ro("eta", &FourParameterBlender::eta)
        .def("blend", &FourParameterBlender::blend, nb::arg("t"), nb::arg("d"),
             "Blend two values with (t,d)")
        .def("__repr__", [](const FourParameterBlender& b) {
            return nb::str("FourParameterBlender(beta1={}, beta2={}, beta3={}, eta={})")
                .format(b.beta1, b.beta2, b.beta3, b.eta);
        });

    nb::class_<ThreeParameterBlender>(blender, "ThreeParameterBlender")
        .def(nb::init<float, float, float>())
        .def_ro("beta1", &ThreeParameterBlender::beta1)
        .def_ro("beta2", &ThreeParameterBlender::beta2)
        .def_ro("eta", &ThreeParameterBlender::eta)
        .def("blend", &ThreeParameterBlender::blend, nb::arg("t"), nb::arg("d"),
             "Blend two values with (t,d)")
        .def("__repr__", [](const ThreeParameterBlender& b) {
            return nb::str("ThreeParameterBlender(beta1={}, beta2={}, eta={})")
                .format(b.beta1, b.beta2, b.eta);
        });
} // NB_MODULE(_genmetaballs_bindings)

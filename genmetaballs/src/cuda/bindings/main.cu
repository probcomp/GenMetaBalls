#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "core/add.cuh"
#include "core/confidence.cuh"
#include "core/geometry.cuh"

constexpr uint32_t GRID_DIM = 4096;
constexpr uint32_t BLOCK_DIM = 1024;

namespace nb = nanobind;

// Forward declaration - will be defined in confidence.cu
void init_confidence_submodule(nb::module_& m);

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
}

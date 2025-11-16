#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "core/add.cuh"
#include "core/geometry.cuh"

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

    nb::class_<Vec3D>(m, "Vec3D")
        .def(nb::init<>())
        .def(nb::init<float, float, float>())
        .def_rw("x", &Vec3D::x)
        .def_rw("y", &Vec3D::y)
        .def_rw("z", &Vec3D::z)
        .def("__add__", &operator+)
        .def("__repr__", [](const Vec3D &v) {
            return "Vec3D(" + std::to_string(v.x) + ", " +
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

}

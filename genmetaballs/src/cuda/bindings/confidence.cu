#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

#include "core/confidence.cuh"

constexpr uint32_t GRID_DIM = 4096;
constexpr uint32_t BLOCK_DIM = 1024;

namespace nb = nanobind;

// helper for type based dispatch of confidence functors
template <typename F>
auto dispatch_confidence(const nb::object& conf_obj, F f) {
    if (nb::isinstance<ThreeParameterConfidence>(conf_obj)) {
        return f(nb::cast<ThreeParameterConfidence>(conf_obj));
    }
    if (nb::isinstance<FiveParameterConfidence>(conf_obj)) {
        return f(nb::cast<FiveParameterConfidence>(conf_obj));
    }
    throw std::runtime_error("Unsupported confidence type");
}

// Initialize the confidence submodule (called from main.cu)
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

    // GPU version -- accepts Python list, returns vector<float>
    m.def(
        "gpu_get_confidence",
        [&](const std::vector<float>& sumexpd, nb::object conf_obj) {
            return dispatch_confidence(conf_obj, [&](auto conf) {
                return gpu_get_confidence<GRID_DIM, BLOCK_DIM>(sumexpd, conf);
            });
        },
        nb::arg("sumexpd"), nb::arg("confidence"));
}

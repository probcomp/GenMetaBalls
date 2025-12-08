#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <tuple>

#include "core/blender.cuh"
#include "core/camera.cuh"
#include "core/confidence.cuh"
#include "core/fmb.cuh"
#include "core/forward.cuh"
#include "core/geometry.cuh"
#include "core/getter.cuh"
#include "core/image.cuh"
#include "core/intersector.cuh"
#include "core/utils.cuh"

namespace nb = nanobind;

template <typename T, MemoryLocation location>
void bind_array2d(nb::module_& m, const char* name);
template <MemoryLocation location>
void bind_image(nb::module_& m, const char* name);
template <MemoryLocation location>
void bind_image_view(nb::module_& m, const char* name);
template <MemoryLocation location>
void bind_fmb_scene(nb::module_& m, const char* name);
template <typename Blender, typename Confidence>
void bind_render_fmbs(nb::module_& m, const char* name);

NB_MODULE(_genmetaballs_bindings, m) {

    /*
     * Confidence module bindings
     */

    nb::module_ confidence = m.def_submodule("confidence");
    nb::class_<ZeroParameterConfidence>(confidence, "ZeroParameterConfidence")
        .def(nb::init<>())
        .def("get_confidence", &ZeroParameterConfidence::get_confidence, nb::arg("sumexpd"),
             "Get the confidence value for a given sumexpd")
        .def("__repr__",
             [](const ZeroParameterConfidence& c) { return nb::str("ZeroParameterConfidence()"); });

    nb::class_<TwoParameterConfidence>(confidence, "TwoParameterConfidence")
        .def(nb::init<float, float>(), nb::arg("beta4"), nb::arg("beta5"))
        .def_ro("beta4", &TwoParameterConfidence::beta4)
        .def_ro("beta5", &TwoParameterConfidence::beta5)
        .def("get_confidence", &TwoParameterConfidence::get_confidence, nb::arg("sumexpd"),
             "Get the confidence value for a given sumexpd")
        .def("__repr__", [](const TwoParameterConfidence& c) {
            return nb::str("TwoParameterConfidence(beta4={}, beta5={})").format(c.beta4, c.beta5);
        });

    /*
     * FMB module bindings
     */

    nb::module_ fmb = m.def_submodule("fmb", "Fuzzy meta ball data types");

    nb::class_<FMB>(fmb, "FMB")
        .def(nb::init<Pose, float, float, float>())
        .def_prop_ro("pose", &FMB::get_pose)
        .def_prop_ro("extent",
                     [](const FMB& self) {
                         auto extent = self.get_extent();
                         return std::tuple{extent.x, extent.y, extent.z};
                     })
        .def("cov_inv_apply", &FMB::cov_inv_apply,
             "apply the inverse covariance matrix to the given vector", nb::arg("vec"))
        .def("quadratic_form", &FMB::quadratic_form,
             "Evaluate the associated quadratic form at the given vector", nb::arg("vec"))
        .def("__repr__", [](const FMB& self) {
                return nb::str("FMB(pose={}, extent={})").format(self.get_pose(), self.get_extent());
            });
    bind_fmb_scene<MemoryLocation::HOST>(fmb, "CPUFMBScene");
    bind_fmb_scene<MemoryLocation::DEVICE>(fmb, "GPUFMBScene");

    /*
     * Forward (rendering) module bindings
     */
    nb::module_ forward = m.def_submodule("forward", "Forward rendering of FMBs");
    bind_render_fmbs<FourParameterBlender, ZeroParameterConfidence>(
        forward, "render_fmbs_four_param_zero_confidence");
    bind_render_fmbs<ThreeParameterBlender, TwoParameterConfidence>(
        forward, "render_fmbs_three_param_two_confidence");
    bind_render_fmbs<ThreeParameterBlender, ZeroParameterConfidence>(
        forward, "render_fmbs_three_param_zero_confidence");
    bind_render_fmbs<FourParameterBlender, TwoParameterConfidence>(
        forward, "render_fmbs_four_param_two_confidence");

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
        .def_prop_ro("quat", [](const Rotation& self) {
            auto quat = self.get_quat();
            return std::tuple{quat.x, quat.y, quat.z, quat.w};
        }, "Get quaternion components as (x, y, z, w)")
        .def("apply", &Rotation::apply, "Apply rotation to vector", nb::arg("vec"))
        .def("compose", &Rotation::compose, "Compose with another rotation", nb::arg("rot"))
        .def("inv", &Rotation::inv, "Inverse rotation")
        .def("__repr__", [](const Rotation& self) {
                auto quat = self.get_quat();
                return nb::str("Rotation(x={}, y={}, z={}, w={})").format(quat.x, quat.y, quat.z, quat.w);
            });

    nb::class_<Pose>(geometry, "Pose")
        .def(nb::init<>())
        .def_static("from_components", &Pose::from_components,
                    "Create rotation from a rotation and a translation", nb::arg("rot"),
                    nb::arg("tran"))
        .def_prop_ro("rot", &Pose::get_rot, "get the rotation component")
        .def_prop_ro("tran", &Pose::get_tran, "get the translation component")
        .def("apply", &Pose::apply, "Apply pose to vector", nb::arg("vec"))
        .def("compose", &Pose::compose, "Compose with another pose", nb::arg("pose"))
        .def("inv", &Pose::inv, "Inverse pose")
        .def("__repr__", [](const Pose& self) {
                return nb::str("Pose(rot={}, tran={})").format(self.get_rot(), self.get_tran());
            });
    /*
     * Camera module bindings
     */
    nb::module_ camera = m.def_submodule("camera", "Camera intrinsics and extrinsics");
    nb::class_<Intrinsics>(camera, "Intrinsics")
        .def(nb::init<uint32_t, uint32_t, float, float, float, float>(), nb::arg("width"),
             nb::arg("height"), nb::arg("fx"), nb::arg("fy"), nb::arg("cx"), nb::arg("cy"))
        .def_ro("height", &Intrinsics::height)
        .def_ro("width", &Intrinsics::width)
        .def_ro("fx", &Intrinsics::fx)
        .def_ro("fy", &Intrinsics::fy)
        .def_ro("cx", &Intrinsics::cx)
        .def_ro("cy", &Intrinsics::cy)
        .def("get_ray_direction", &Intrinsics::get_ray_direction,
             "Get the direction of the ray going through pixel (px, py) in camera frame",
             nb::arg("px"), nb::arg("py"))
        .def("__repr__", [](const Intrinsics& self) {
                return nb::str("Intrinsics(width={}, height={}, fx={}, fy={}, cx={}, cy={})").format(self.width, self.height, self.fx, self.fy, self.cx, self.cy);
            });

    /*
     * Image module bindings
     */
    nb::module_ image = m.def_submodule("image", "Image data structure for GenMetaballs");
    bind_image_view<MemoryLocation::HOST>(image, "CPUImageView");
    bind_image<MemoryLocation::HOST>(image, "CPUImage");
    bind_image_view<MemoryLocation::DEVICE>(image, "GPUImageView");
    bind_image<MemoryLocation::DEVICE>(image, "GPUImage");

    /*
     * Intersector module bindings
     */

    nb::module_ intersector = m.def_submodule("intersector");
    intersector.def(
        "linear_intersect",
        [](const FMB& fmb, const Vec3D& ray, const Pose& cam_pose) {
            auto [t, d] = LinearIntersector::intersect(fmb, ray, cam_pose);
            return std::make_tuple(t, d);
        },
        "Linear intersection of ray and FMB.", nb::arg("fmb"), nb::arg("ray"), nb::arg("cam_pose"));

    /*
     * Utils module bindings
     */

    nb::module_ utils = m.def_submodule("utils");
    utils.def("sigmoid", sigmoid, nb::arg("x"), "Compute the sigmoid function: 1 / (1 + exp(-x))");

    // blender submodule
    nb::module_ blender = m.def_submodule("blender");
    nb::class_<FourParameterBlender>(blender, "FourParameterBlender")
        .def(nb::init<float, float, float, float>(), nb::arg("beta1"), nb::arg("beta2"),
             nb::arg("beta3"), nb::arg("eta"))
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
        .def(nb::init<float, float, float>(), nb::arg("beta1"), nb::arg("beta2"), nb::arg("eta"))
        .def_ro("beta1", &ThreeParameterBlender::beta1)
        .def_ro("beta2", &ThreeParameterBlender::beta2)
        .def_ro("eta", &ThreeParameterBlender::eta)
        .def("blend", &ThreeParameterBlender::blend, nb::arg("t"), nb::arg("d"),
             "Blend two values with (t,d)")
        .def("__repr__", [](const ThreeParameterBlender& b) {
            return nb::str("ThreeParameterBlender(beta1={}, beta2={}, eta={})")
                .format(b.beta1, b.beta2, b.eta);
        });
    bind_array2d<float, MemoryLocation::HOST>(utils, "CPUFloatArray2D");
    bind_array2d<float, MemoryLocation::DEVICE>(utils, "GPUFloatArray2D");

} // NB_MODULE(_genmetaballs_bindings)

template <typename T, MemoryLocation location>
void bind_array2d(nb::module_& m, const char* name) {
    using nb_device_type =
        std::conditional_t<location == MemoryLocation::HOST, nb::device::cpu, nb::device::cuda>;
    nb::class_<Array2D<T, location>>(m, name)
        .def_static("from_array",
                    [](const nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb_device_type>& array) {
                        return Array2D<T, location>(array.data(), array.shape(0), array.shape(1));
                    })
        // TODO: switch to the array_api in future nanobind release
        // https://nanobind.readthedocs.io/en/latest/api_extra.html#_CPPv4N8nanobind9array_apiE
        .def(
            "as_numpy",
            [](const Array2D<T, location>& self) {
                return nb::ndarray<T, nb::numpy, nb::c_contig, nb_device_type>(
                    self.data(), {self.num_rows(), self.num_cols()});
            },
            nb::rv_policy::reference_internal)
        .def(
            "as_jax",
            [](const Array2D<T, location>& self) {
                return nb::ndarray<T, nb::jax, nb::c_contig, nb_device_type>(
                    self.data(), {self.num_rows(), self.num_cols()});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro("num_rows", &Array2D<T, location>::num_rows)
        .def_prop_ro("num_cols", &Array2D<T, location>::num_cols)
        .def_prop_ro("ndim", &Array2D<T, location>::ndim)
        .def_prop_ro("size", &Array2D<T, location>::size);
}

template <MemoryLocation location>
void bind_image_view(nb::module_& m, const char* name) {
    nb::class_<ImageView<location>>(m, name)
        .def(nb::init<const Array2D<float, location>&, const Array2D<float, location>&>(),
             nb::arg("confidence"), nb::arg("depth"))
        .def_prop_ro("confidence", [](const ImageView<location>& view) { return view.confidence; })
        .def_prop_ro("depth", [](const ImageView<location>& view) { return view.depth; })
        .def_prop_ro("num_rows", &ImageView<location>::num_rows)
        .def_prop_ro("num_cols", &ImageView<location>::num_cols)
        .def("__repr__", [=](const ImageView<location>& view) {
            return nb::str("{}(height={}, width={})")
                .format(name, view.num_rows(), view.num_cols());
        });
}

template <MemoryLocation location>
void bind_image(nb::module_& m, const char* name) {
    nb::class_<Image<location>>(m, name)
        .def(nb::init<uint32_t, uint32_t>(), nb::arg("height"), nb::arg("width"))
        .def_prop_ro("num_rows", &Image<location>::num_rows)
        .def_prop_ro("num_cols", &Image<location>::num_cols)
        .def("as_view", &Image<location>::as_view, "Get a view of the image data as ImageView")
        .def("__repr__", [=](const Image<location>& img) {
            return nb::str("{}(height={}, width={})").format(name, img.num_rows(), img.num_cols());
        });
}

template <MemoryLocation location>
void bind_fmb_scene(nb::module_& m, const char* name) {
    nb::class_<FMBScene<location>>(m, name)
        .def(nb::init<size_t>(), nb::arg("size"))
        .def(nb::init<const std::vector<FMB>&, const std::vector<float>&>(), nb::arg("fmbs"),
             nb::arg("log_weights"),
             "Construct FMBScene from a list of FMBs and corresponding log weights")
        .def_prop_ro("size", &FMBScene<location>::size)
        .def("__len__", &FMBScene<location>::size)
        .def(
            "__getitem__",
            // Convert cuda::std::tuple to std::tuple for nanobind
            [](const FMBScene<location>& scene, size_t idx) {
                const auto& [fmb, log_weight] = scene[idx];
                // for device data, the types would be thrust::device_reference, which cannot be
                // returned directly to Python. The static cast forces a copy (to host) to be made.
                return std::make_tuple(static_cast<const FMB&>(fmb),
                                       static_cast<const float&>(log_weight));
            },
            "Get the (FMB, log_weight) tuple at index i")
        .def("__repr__", [=](const FMBScene<location>& scene) {
            return nb::str("{}(size={})").format(name, scene.size());
        });
}

template <typename Blender, typename Confidence>
void bind_render_fmbs(nb::module_& m, const char* name) {
    m.def(name,
          &render_fmbs<AllGetter<MemoryLocation::DEVICE>, LinearIntersector, Blender, Confidence>,
          "Render the given FMB scene into the provided image view", nb::arg("fmbs"),
          nb::arg("blender"), nb::arg("confidence"), nb::arg("intr"), nb::arg("extr"),
          nb::arg("img"));
}

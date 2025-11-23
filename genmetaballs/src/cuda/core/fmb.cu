#include "fmb.cuh"
#include "geometry.cuh"
#include "utils.cuh"

CUDA_CALLABLE float FMB::quadratic_form(const Vec3D vec) const {
    const auto shftd_vec = vec - pose.get_tran();
    const auto rot_shftd_vec = pose.get_rot().apply(shftd_vec);
    const auto scaled_rot_shftd_vec =
        Vec3D(rot_shftd_vec.x / extent.x, rot_shftd_vec.y / extent.y, rot_shftd_vec.z / extent.z);
    return dot(rot_shftd_vec, scaled_rot_shftd_vec);
}

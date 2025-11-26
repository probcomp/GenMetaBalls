#pragma once

#include <cuda/std/tuple>

#include "fmb.cuh"
#include "geometry.cuh"

// implement equation (6) in the paper
class LinearIntersector {
public:
    CUDA_CALLABLE static cuda::std::tuple<float, float> intersect(const FMB& fmb, const Ray& ray) {
        auto vecdiv = [](const Vec3D& u, const Vec3D& v) {
            return Vec3D{u.x / v.x, u.y / v.y, u.z / v.z};
        };
        auto rot = fmb.get_pose().get_rot();
        auto tmp = rot.inv().apply(vecdiv(rot.apply(ray.direction), fmb.get_extent()));
        auto t = dot(fmb.get_pose().get_tran() - ray.start, tmp) / dot(ray.direction, tmp);
        return {t, fmb.quadratic_form(ray.start + t * ray.direction)};
    }
};

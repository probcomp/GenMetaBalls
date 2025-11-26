#pragma once

#include <utility>

#include "fmb.h"
#include "geometry.h"

// implement equation (6) in the paper
class LinearIntersector {

    static CUDA_CALLABLE std::pair<float, float> intersect(const FMB& fmb, const Ray& ray) const {
        auto vecdiv = [](const Vec3D& u, const Vec3D& v) {
            return Vec3D{u.x / v.x, u.y / v.y, u.z / v.z};
        };
        auto rot = fmb.get_pose().get_rot();
        auto tmp = rot.inv().apply(vecdiv(rot.apply(ray.direction), fmb.get_extent()));
        return dot(fmb.get_pose().get_tran() - ray.start, tmp) / dot(ray.direction, tmp);
    }
};

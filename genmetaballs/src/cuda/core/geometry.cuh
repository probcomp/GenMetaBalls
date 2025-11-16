#pragma once

#include <cuda_runtime.h>


typedef Vec3D = float3;


class Rotation {

    private:
        // ...
        float rotmat_[9];

    public:
        Vec3D apply(const Vec3D vec) const;
        Rotation compose(const Rotation &rot) const;
        Rotation inv() const;
};

struct Pose {
    Rotation rot;
    Vec3D tran;

    Vec3D apply(const Vec3D vec) const;
    Pose compose(const Pose &pose) const;
    Pose inv() const;
};

struct Ray {
    Vec3D start;
    Vec3D direction;
};

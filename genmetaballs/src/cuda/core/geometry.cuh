#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

using Vec3D = float3;

CUDA_CALLABLE inline Vec3D operator-(const Vec3D a)
{
    return {-a.x, -a.y, -a.z};
}

CUDA_CALLABLE inline Vec3D operator+(const Vec3D a, const Vec3D b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

CUDA_CALLABLE inline Vec3D operator-(const Vec3D a, const Vec3D b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

CUDA_CALLABLE inline Vec3D operator*(const float scalar, const Vec3D a)
{
    return {a.x * scalar, a.y * scalar, a.z * scalar};
}

CUDA_CALLABLE inline Vec3D operator*(const Vec3D a, const float scalar)
{
    return {a.x * scalar, a.y * scalar, a.z * scalar};
}

CUDA_CALLABLE inline Vec3D operator/(const Vec3D a, const float scalar)
{
    return {a.x / scalar, a.y / scalar, a.z / scalar};
}

CUDA_CALLABLE inline float dot(const Vec3D a, const Vec3D b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

CUDA_CALLABLE inline Vec3D cross(const Vec3D a, const Vec3D b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

class Rotation {
private:
    float4 unit_quat_;

    Rotation(float4 unit_quat): unit_quat_{unit_quat} {};

public:
    Rotation(): unit_quat_{0.0f, 0.0f, 0.0f, 1.0f} {};

    static Rotation from_quat(float x, float y, float z, float w);

    CUDA_CALLABLE Vec3D apply(const Vec3D vec) const;

    CUDA_CALLABLE Rotation compose(const Rotation& rot) const;

    CUDA_CALLABLE Rotation inv() const;
};

struct Pose {
    Rotation rot;
    Vec3D tran;

    CUDA_CALLABLE inline Vec3D apply(const Vec3D vec) const
    {
        return tran + rot.apply(vec);
    }

    CUDA_CALLABLE inline Pose compose(const Pose &pose) const
    {
        /*
         * If $A_i$ is the matrix corresponding to pose object `p_i`, then
         * $A_1A_2$ is the matrix corresponding to the pose object
         * `p_1.compose(p2)`.
         */
        return {rot.compose(pose.rot), rot.apply(pose.tran) + tran};
    }

    CUDA_CALLABLE inline Pose inv() const
    {
        auto rotinv = rot.inv();
        return {rotinv, -rotinv.apply(tran)};
    }
};

struct Ray {
    Vec3D start;
    Vec3D direction;
};

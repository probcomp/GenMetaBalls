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

    CUDA_CALLABLE Rotation(float4 unit_quat): unit_quat_{unit_quat} {};

public:
    CUDA_CALLABLE Rotation(): unit_quat_{0.0f, 0.0f, 0.0f, 1.0f} {};

    static CUDA_CALLABLE Rotation from_quat(float x, float y, float z, float w);

    CUDA_CALLABLE Vec3D apply(const Vec3D vec) const;

    CUDA_CALLABLE Rotation compose(const Rotation& rot) const;

    CUDA_CALLABLE Rotation inv() const;
};

class Pose {
private:
    Rotation rot_;
    Vec3D tran_;

    CUDA_CALLABLE Pose(const Rotation rot, const Vec3D tran): rot_{rot}, tran_{tran} {}

public:
    //these member functions are defined in class body to allow for possible inlining
    
    CUDA_CALLABLE Pose(): rot_{Rotation()}, tran_{0.0f, 0.0f, 0.0f} {}

    static CUDA_CALLABLE Pose from_components(const Rotation rot, const Vec3D tran)
    {
        return {rot, tran};
    }

    CUDA_CALLABLE Rotation get_rot() const
    {
        return rot_;
    }

    CUDA_CALLABLE Vec3D get_tran() const
    {
        return tran_;
    }

    CUDA_CALLABLE Vec3D apply(const Vec3D vec) const
    {
        return tran_ + rot_.apply(vec);
    }

    CUDA_CALLABLE Pose compose(const Pose &pose) const
    {
        /*
         * If $A_i$ is the matrix corresponding to pose object `p_i`, then
         * $A_1A_2$ is the matrix corresponding to the pose object
         * `p_1.compose(p2)`.
         */
        return {rot_.compose(pose.rot_), rot_.apply(pose.tran_) + tran_};
    }

    CUDA_CALLABLE Pose inv() const
    {
        auto rotinv = rot_.inv();
        return {rotinv, -rotinv.apply(tran_)};
    }
};

struct Ray {
    Vec3D start;
    Vec3D direction;
};

#pragma once

#include <cuda_runtime.h>

using Vec3D = float3;

inline Vec3D operator-(const Vec3D a)
{
    return {-a.x, -a.y, -a.z};
}

inline Vec3D operator+(const Vec3D a, const Vec3D b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3D operator-(const Vec3D a, const Vec3D b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3D operator*(const float scalar, const Vec3D a)
{
    return {a.x * scalar, a.y * scalar, a.z * scalar};
}

inline Vec3D operator*(const Vec3D a, const float scalar)
{
    return {a.x * scalar, a.y * scalar, a.z * scalar};
}

inline Vec3D operator/(const Vec3D a, const float scalar)
{
    return {a.x / scalar, a.y / scalar, a.z / scalar};
}

inline float dot(const Vec3D a, const Vec3D b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3D cross(const Vec3D a, const Vec3D b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

class Rotation {
private:
    float4 unit_quat_;

public:
    Rotation() unit_quat_{0.0f, 0.0f, 0.0f, 1.0f} {};

    __host__ __device__ Vec3D apply(const Vec3D vec) const;

    __host__ __device__ Rotation compose(const Rotation& rot) const;

    __host__ __device__ Rotation inv() const;
};

struct Pose {
    Rotation rot;
    Vec3D tran;

    __host__ __device__ inline Vec3D apply(const Vec3D vec) const
    {
        return tran + rot.apply(vec);
    }

    __host__ __device__ inline Pose compose(const Pose &pose) const
    {
        /*
         * If $A_i$ is the matrix corresponding to pose object `p_i`, then
         * $A_1A_2$ is the matrix corresponding to the pose object
         * `p_1.compose(p2)`.
         */
        return {rot.compose(pose.rot), rot.apply(pose.tran) + tran};
    }

    __host__ __device__ inline Pose inv() const
    {
        auto rotinv = rot.inv();
        return {rotinv, -rotinv.apply(tran)};
    }
};

struct Ray {
    Vec3D start;
    Vec3D direction;
};

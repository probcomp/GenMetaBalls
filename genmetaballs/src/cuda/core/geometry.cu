#include <cmath>

#include "geometry.cuh"

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
CUDA_CALLABLE Rotation Rotation::from_quat(float x, float y, float z, float w) {
    auto modulus = std::sqrt(x * x + y * y + z * z + w * w);
    return Rotation{{x / modulus, y / modulus, z / modulus, w / modulus}};
}

CUDA_CALLABLE Vec3D Rotation::apply(const Vec3D vec) const {
    // v' = q * v * q^(-1) for unit quaternions
    // where q^(-1) = (-x, -y, -z, w)
    Vec3D q = {unit_quat_.x, unit_quat_.y, unit_quat_.z};
    float w = unit_quat_.w;

    // v' = 2*(q·v)*q + (w²-|q|²)*v + 2*w*(q×v)
    float d = dot(q, vec);
    Vec3D c = cross(q, vec);

    return 2.0f * d * q + (w * w - dot(q, q)) * vec + 2.0f * w * c;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
CUDA_CALLABLE Rotation Rotation::compose(const Rotation& rot) const {
    // Quaternion multiplication: q1 * q2
    float4 q1 = unit_quat_;
    float4 q2 = rot.unit_quat_;

    return Rotation{{q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
                     q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
                     q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
                     q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z}};
}

CUDA_CALLABLE Rotation Rotation::inv() const {
    // For unit quaternions, inverse = conjugate: (-x, -y, -z, w)
    return Rotation{{-unit_quat_.x, -unit_quat_.y, -unit_quat_.z, unit_quat_.w}};
}

#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

using Vec3D = float3;

CUDA_CALLABLE inline Vec3D operator-(const Vec3D a) {
    return {-a.x, -a.y, -a.z};
}

CUDA_CALLABLE inline Vec3D operator+(const Vec3D a, const Vec3D b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

CUDA_CALLABLE inline Vec3D operator-(const Vec3D a, const Vec3D b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

CUDA_CALLABLE inline Vec3D operator*(const float scalar, const Vec3D a) {
    return {a.x * scalar, a.y * scalar, a.z * scalar};
}

CUDA_CALLABLE inline Vec3D operator*(const Vec3D a, const float scalar) {
    return {a.x * scalar, a.y * scalar, a.z * scalar};
}

CUDA_CALLABLE inline Vec3D operator/(const Vec3D a, const float scalar) {
    return {a.x / scalar, a.y / scalar, a.z / scalar};
}

CUDA_CALLABLE inline float dot(const Vec3D a, const Vec3D b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

CUDA_CALLABLE inline Vec3D cross(const Vec3D a, const Vec3D b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// CLAUDE TODO add a simple Mat33 type representing 3 x 3 matrices and
// implement basic CUDA_CALLABLE matrix-matrix and matrix-vector algebraic
// operations similar to the Vec3D class

struct Mat33 {
    float m[3][3];

    CUDA_CALLABLE Mat33() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] = 0.0f;
            }
        }
    }

    CUDA_CALLABLE Mat33(float m00, float m01, float m02,
                        float m10, float m11, float m12,
                        float m20, float m21, float m22) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
    }

    CUDA_CALLABLE static Mat33 identity() {
        return Mat33(1.0f, 0.0f, 0.0f,
                     0.0f, 1.0f, 0.0f,
                     0.0f, 0.0f, 1.0f);
    }

    CUDA_CALLABLE Mat33& operator+=(const Mat33& b) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] += b.m[i][j];
            }
        }
        return *this;
    }

    CUDA_CALLABLE Mat33& operator-=(const Mat33& b) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] -= b.m[i][j];
            }
        }
        return *this;
    }
};

// Matrix-matrix multiplication
CUDA_CALLABLE inline Mat33 operator*(const Mat33& a, const Mat33& b) {
    Mat33 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.m[i][j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                result.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    return result;
}

// Matrix-vector multiplication
CUDA_CALLABLE inline Vec3D operator*(const Mat33& a, const Vec3D v) {
    return {
        a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
        a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
        a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z
    };
}

// Scalar-matrix multiplication
CUDA_CALLABLE inline Mat33 operator*(const float scalar, const Mat33& a) {
    Mat33 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.m[i][j] = scalar * a.m[i][j];
        }
    }
    return result;
}

CUDA_CALLABLE inline Mat33 operator*(const Mat33& a, const float scalar) {
    return scalar * a;
}

// Matrix addition
CUDA_CALLABLE inline Mat33 operator+(const Mat33& a, const Mat33& b) {
    Mat33 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.m[i][j] = a.m[i][j] + b.m[i][j];
        }
    }
    return result;
}

// Matrix subtraction
CUDA_CALLABLE inline Mat33 operator-(const Mat33& a, const Mat33& b) {
    Mat33 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.m[i][j] = a.m[i][j] - b.m[i][j];
        }
    }
    return result;
}

// Matrix transpose
CUDA_CALLABLE inline Mat33 transpose(const Mat33& a) {
    return Mat33(a.m[0][0], a.m[1][0], a.m[2][0],
                 a.m[0][1], a.m[1][1], a.m[2][1],
                 a.m[0][2], a.m[1][2], a.m[2][2]);
}

// END CLAUDE TODO

class Rotation {
private:
    float4 unit_quat_;

    CUDA_CALLABLE Rotation(float4 unit_quat) : unit_quat_{unit_quat} {};

public:
    CUDA_CALLABLE Rotation() : unit_quat_{0.0f, 0.0f, 0.0f, 1.0f} {};

    static CUDA_CALLABLE Rotation from_quat(float x, float y, float z, float w);

    CUDA_CALLABLE const float4& get_quat() const;

    CUDA_CALLABLE Vec3D apply(const Vec3D vec) const;

    CUDA_CALLABLE Rotation compose(const Rotation& rot) const;

    CUDA_CALLABLE Rotation inv() const;
};

class Pose {
private:
    Rotation rot_;
    Vec3D tran_;

    CUDA_CALLABLE Pose(const Rotation rot, const Vec3D tran) : rot_{rot}, tran_{tran} {}

public:
    // these member functions are defined in class body to allow for possible inlining

    CUDA_CALLABLE Pose() : rot_{Rotation()}, tran_{0.0f, 0.0f, 0.0f} {}

    static CUDA_CALLABLE Pose from_components(const Rotation rot, const Vec3D tran) {
        return {rot, tran};
    }

    CUDA_CALLABLE const Rotation& get_rot() const {
        return rot_;
    }

    CUDA_CALLABLE const Vec3D& get_tran() const {
        return tran_;
    }

    CUDA_CALLABLE Vec3D apply(const Vec3D vec) const {
        return tran_ + rot_.apply(vec);
    }

    CUDA_CALLABLE Pose compose(const Pose& pose) const {
        /*
         * If $A_i$ is the matrix corresponding to pose object `p_i`, then
         * $A_1A_2$ is the matrix corresponding to the pose object
         * `p_1.compose(p2)`.
         */
        return {rot_.compose(pose.rot_), rot_.apply(pose.tran_) + tran_};
    }

    CUDA_CALLABLE Pose inv() const {
        auto rotinv = rot_.inv();
        return {rotinv, -rotinv.apply(tran_)};
    }
};

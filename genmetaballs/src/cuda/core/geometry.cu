#include "geometry.cuh"

Vec3D operator+(const Vec3D a, const Vec3D b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

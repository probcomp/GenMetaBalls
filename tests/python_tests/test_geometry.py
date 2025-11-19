import operator as op

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from genmetaballs._genmetaballs_bindings import geometry as geometry


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_vec3d_smoke() -> None:
    geometry.Vec3D(0, 0, 0)


def test_vec3d_repr_returns_valid_string() -> None:
    v = geometry.Vec3D(1.0, 2.0, 3.0)
    repr_str = repr(v)
    assert isinstance(repr_str, str)
    assert repr_str == "Vec3D(1.0, 2.0, 3.0)"


@pytest.mark.parametrize(
    "np_op,vec3d_op",
    [
        (np.add, op.add),
        (np.subtract, op.sub),
        (np.cross, geometry.cross),
    ]
 )
def test_vec3d_ops(rng: np.random.Generator, np_op, vec3d_op) -> None:
    _a, _b = rng.uniform(size=3), rng.uniform(size=3)
    a, b = geometry.Vec3D(*_a), geometry.Vec3D(*_b)
    c = vec3d_op(a, b)
    _c = np_op(_a, _b)
    assert np.allclose(_c, np.array([c.x, c.y, c.z]))

def test_vec3d_dot(rng: np.random.Generator) -> None:
    _a, _b = rng.uniform(size=3), rng.uniform(size=3)
    a, b = geometry.Vec3D(*_a), geometry.Vec3D(*_b)
    assert np.allclose(np.dot(_a, _b), geometry.dot(a, b))

def test_rotation_apply(rng: np.random.Generator) -> None:
    quats = rng.uniform(-1, 1, size=(100, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    vecs = rng.uniform(size=(100, 3))

    for i in range(100):
        rot_scipy = Rot.from_quat(quats[i])
        rot_geom = geometry.Rotation.from_quat(*quats[i])
        vec_geom = geometry.Vec3D(*vecs[i])

        result_scipy = rot_scipy.apply(vecs[i])
        result_geom = rot_geom.apply(vec_geom)

        assert np.allclose(result_scipy, np.array([result_geom.x, result_geom.y, result_geom.z]))

def test_rotation_compose(rng: np.random.Generator) -> None:
    quats1 = rng.uniform(-1, 1, size=(100, 4))
    quats1 /= np.linalg.norm(quats1, axis=1, keepdims=True)
    quats2 = rng.uniform(-1, 1, size=(100, 4))
    quats2 /= np.linalg.norm(quats2, axis=1, keepdims=True)
    vecs = rng.uniform(size=(50, 3))

    for i in range(100):
        rot1_scipy = Rot.from_quat(quats1[i])
        rot2_scipy = Rot.from_quat(quats2[i])
        composed_scipy = rot1_scipy * rot2_scipy

        rot1_geom = geometry.Rotation.from_quat(*quats1[i])
        rot2_geom = geometry.Rotation.from_quat(*quats2[i])
        composed_geom = rot1_geom.compose(rot2_geom)

        for j in range(50):
            vec_geom = geometry.Vec3D(*vecs[j])
            result_scipy = composed_scipy.apply(vecs[j])
            result_geom = composed_geom.apply(vec_geom)
            assert np.allclose(
               result_scipy,
               np.array([result_geom.x, result_geom.y, result_geom.z]),
               rtol=1e-5,
               atol=1e-6,
           )

def test_rotation_inv(rng: np.random.Generator) -> None:
    quats = rng.uniform(-1, 1, size=(100, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    vecs = rng.uniform(size=(50, 3))

    for i in range(100):
        rot = geometry.Rotation.from_quat(*quats[i])
        rotinv = rot.inv()
        composed = rot.compose(rotinv)

        for j in range(50):
            vec = geometry.Vec3D(*vecs[j])
            vec_ = composed.apply(vec)
            assert np.allclose(
               np.array([vec.x, vec.y, vec.z]),
               np.array([vec_.x, vec_.y, vec_.z]),
               rtol=1e-5,
               atol=1e-6,
           )

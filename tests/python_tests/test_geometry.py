import operator as op

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RigidTransform as Rigid

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
    _a, _b = rng.uniform(size=(100, 3)), rng.uniform(size=(100, 3))
    for i in range(100):
        a, b = geometry.Vec3D(*_a[i]), geometry.Vec3D(*_b[i])
        c = vec3d_op(a, b)
        _c = np_op(_a[i], _b[i])
        assert np.allclose(_c, np.array([c.x, c.y, c.z]))

def test_vec3d_dot(rng: np.random.Generator) -> None:
    _a, _b = rng.uniform(size=(100, 3)), rng.uniform(size=(100, 3))
    for i in range(100):
        a, b = geometry.Vec3D(*_a[i]), geometry.Vec3D(*_b[i])
        assert np.allclose(np.dot(_a[i], _b[i]), geometry.dot(a, b))

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

def test_pose_apply(rng: np.random.Generator) -> None:
    exp_coords = rng.uniform(size=(100, 6))
    vecs = rng.uniform(size=(100, 3))

    for i in range(100):
        pose_scipy = Rigid.from_exp_coords(exp_coords[i])

        pose_geom = geometry.Pose.from_components(
            rot=geometry.Rotation.from_quat(*pose_scipy.rotation.as_quat()),
            tran=geometry.Vec3D(*pose_scipy.translation),
        )
        vec_geom = geometry.Vec3D(*vecs[i])

        result_scipy = pose_scipy.apply(vecs[i])
        result_geom = pose_geom.apply(vec_geom)

        assert np.allclose(result_scipy, np.array([result_geom.x, result_geom.y, result_geom.z]))

def test_pose_compose(rng: np.random.Generator) -> None:
    exp_coords1 = rng.uniform(size=(100, 6))
    exp_coords2 = rng.uniform(size=(100, 6))
    vecs = rng.uniform(size=(50, 3))

    for i in range(100):
        pose1_scipy = Rigid.from_exp_coords(exp_coords1[i])
        pose2_scipy = Rigid.from_exp_coords(exp_coords2[i])
        composed_scipy = pose1_scipy * pose2_scipy

        pose1_geom = geometry.Pose.from_components(
            rot=geometry.Rotation.from_quat(*pose1_scipy.rotation.as_quat()),
            tran=geometry.Vec3D(*pose1_scipy.translation),
        )
        pose2_geom = geometry.Pose.from_components(
            rot=geometry.Rotation.from_quat(*pose2_scipy.rotation.as_quat()),
            tran=geometry.Vec3D(*pose2_scipy.translation),
        )
        composed_geom = pose1_geom.compose(pose2_geom)

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

def test_pose_inv(rng: np.random.Generator) -> None:
    exp_coords = rng.uniform(size=(100, 6))
    vecs = rng.uniform(size=(50, 3))

    for i in range(100):
        pose_scipy = Rigid.from_exp_coords(exp_coords[i])

        pose = geometry.Pose.from_components(
            rot=geometry.Rotation.from_quat(*pose_scipy.rotation.as_quat()),
            tran=geometry.Vec3D(*pose_scipy.translation),
        )
        poseinv = pose.inv()
        composed = pose.compose(poseinv)

        for j in range(50):
            vec = geometry.Vec3D(*vecs[j])
            vec_ = composed.apply(vec)
            assert np.allclose(
               np.array([vec.x, vec.y, vec.z]),
               np.array([vec_.x, vec_.y, vec_.z]),
               rtol=1e-5,
               atol=1e-6,
           )

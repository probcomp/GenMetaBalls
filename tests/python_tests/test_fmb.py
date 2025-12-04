import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis
from scipy.spatial.transform import Rotation as Rot

from genmetaballs.core import fmb, geometry, make_fmb_scene

FMB = fmb.FMB
Pose, Vec3D, Rotation = geometry.Pose, geometry.Vec3D, geometry.Rotation


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_fmb_cov_inv_apply(rng):
    for _ in range(100):
        quat = rng.uniform(size=4).astype(np.float32)
        tran, extent, vec = rng.uniform(size=(3, 3)).astype(np.float32)
        pose = Pose.from_components(Rotation.from_quat(*quat), Vec3D(*tran))
        scipy_rot_mat = Rot.from_quat(quat).as_matrix()
        cov = scipy_rot_mat.T @ np.diag(extent) @ scipy_rot_mat
        theirs = np.linalg.solve(cov, vec)
        ours = FMB(pose, *extent).cov_inv_apply(Vec3D(*vec))
        ourvec = np.array([ours.x, ours.y, ours.z], dtype=np.float32)
        assert np.allclose(theirs, ourvec, atol=1e-6)


def test_fmb_quadratic_form(rng):
    for _ in range(100):
        quat = rng.uniform(size=4)
        tran, extent, vec = rng.uniform(size=(3, 3))
        pose = Pose.from_components(Rotation.from_quat(*quat), Vec3D(*tran))
        scipy_rot_mat = Rot.from_quat(quat).as_matrix()
        cov = scipy_rot_mat.T @ np.diag(extent) @ scipy_rot_mat
        assert np.isclose(
            FMB(pose, *extent).quadratic_form(Vec3D(*vec)),
            mahalanobis(vec, tran, np.linalg.inv(cov)) ** 2,
        )


def test_fmb_scene_creation():
    cpu_scene = make_fmb_scene(10, device="cpu")
    assert isinstance(cpu_scene, fmb.CPUFMBScene)
    assert len(cpu_scene) == 10

    gpu_scene = make_fmb_scene(20, device="gpu")
    assert isinstance(gpu_scene, fmb.GPUFMBScene)
    assert len(gpu_scene) == 20

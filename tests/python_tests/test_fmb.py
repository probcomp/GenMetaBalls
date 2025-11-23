import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis
from scipy.spatial.transform import Rotation as Rot

from genmetaballs.core.fmb import FMB
from genmetaballs.core.geometry import Pose, Rotation, Vec3D


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_fmb_quadratic_form(rng):
    for _ in range(100):
        quat = rng.uniform(size=4)
        tran, extent, vec = rng.uniform(size=(3, 3))
        pose = Pose.from_components(Rotation.from_quat(quat), Vec3D(tran))
        scipy_rot_mat = Rot.from_quat(quat).as_matrix()
        cov = scipy_rot_mat.T @ np.diag(extent) @ scipy_rot_mat
        assert np.isclose(
            FMB(pose, Vec3D(extent)).quadratic_form(vec),
            mahalanobis(vec, tran, np.linalg.inv(cov)) ** 2,
        )

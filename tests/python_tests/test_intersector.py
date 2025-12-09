import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis
from scipy.spatial.transform import Rotation as Rot

from genmetaballs.core import fmb, geometry, intersector

FMB = fmb.FMB
Pose, Vec3D, Rotation = geometry.Pose, geometry.Vec3D, geometry.Rotation


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.mark.xfail(reason="Depends on cov_inv_apply, which is not implemented correctly")
def test_linear_intersect(rng):
    for _ in range(100):
        # sample
        cam_quat, fmb_quat = rng.uniform(size=(2, 4)).astype(np.float32)
        fmb_extent, fmb_mu = rng.uniform(size=(2, 3)).astype(np.float32)
        cam_tran, ray_dir = rng.uniform(size=(2, 3)).astype(np.float32)
        # ground truth computation
        v = Rot.from_quat(cam_quat).apply(ray_dir)
        fmb_rotmat = Rot.from_quat(fmb_quat).as_matrix()
        cov_inv = fmb_rotmat.T @ np.diag(1 / fmb_extent) @ fmb_rotmat
        t = ((fmb_mu - cam_tran) @ cov_inv @ v) / (v @ cov_inv @ v)
        d = mahalanobis(fmb_mu, cam_tran + t * v, cov_inv) ** 2
        # cuda computation
        fmb_pose = Pose.from_components(Rotation.from_quat(*fmb_quat), Vec3D(*fmb_mu))
        cam_pose = Pose.from_components(Rotation.from_quat(*cam_quat), Vec3D(*cam_tran))
        fmb = FMB(fmb_pose, *fmb_extent)
        ray = Vec3D(*ray_dir)
        t_, d_ = intersector.linear_intersect(fmb, ray, cam_pose)
        assert np.isclose(t, t_) and np.isclose(d, d_)

import numpy as np
import pytest

from genmetaballs.core import Intrinsics
from genmetaballs.fmb.utils import get_camera_rays


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def intrinsics() -> Intrinsics:
    return Intrinsics(width=640, height=480, fx=500.0, fy=520.0, cx=320.0, cy=240.0)


def test_get_ray_direction_in_camera_frame(intrinsics: Intrinsics):
    # selet a few random pixels to test
    pixel_list = np.array(
        [
            [0, 0],
            [intrinsics.width - 1, 0],
            [0, intrinsics.height - 1],
            [intrinsics.width - 1, intrinsics.height - 1],
            [intrinsics.width // 2, intrinsics.height // 2],
        ]
    )

    reference_rays = get_camera_rays(
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        np.concatenate([pixel_list, np.zeros((pixel_list.shape[0], 1))], axis=1),
    )

    for (pixel_x, pixel_y), reference_ray in zip(pixel_list, reference_rays, strict=True):
        ray_direction = intrinsics.get_ray_direction(pixel_x, pixel_y)
        assert np.allclose(
            [ray_direction.x, ray_direction.y, ray_direction.z],
            reference_ray,
        )

import numpy as np
import pytest

from genmetaballs import _genmetaballs_bindings as _gmbb


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_vec3d(rng: np.random.Generator) -> None:
    _gmbb.Vec3D(0, 0, 0)

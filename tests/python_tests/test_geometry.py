import numpy as np
import pytest

from genmetaballs import _genmetaballs_bindings.geometry as geometry


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


def test_vec3d_add(rng: np.random.Generator) -> None:
    _a, _b = rng.uniform(size=3), rng.uniform(size=3)
    a, b = geometry.Vec3D(*_a), geometry.Vec3D(*_b)
    c = a + b
    _c = _a + _b
    assert all([np.isclose(c.x, _c[0]), np.isclose(c.y, _c[1]), np.isclose(c.z, _c[2])])


def test_vec3d_sub(rng: np.random.Generator) -> None:
    _a, _b = rng.uniform(size=3), rng.uniform(size=3)
    a, b = geometry.Vec3D(*_a), geometry.Vec3D(*_b)
    c = a - b
    _c = _a - _b
    assert all([np.isclose(c.x, _c[0]), np.isclose(c.y, _c[1]), np.isclose(c.z, _c[2])])

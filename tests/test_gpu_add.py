import numpy as np
import pytest

from genmetaballs import gpu_add


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_gpu_add(rng: np.random.Generator) -> None:
    N = 8196

    a = rng.normal(size=N).astype(np.float32).tolist()
    b = rng.normal(size=N).astype(np.float32).tolist()
    c = gpu_add(a, b)
    assert all(abs(x + y - z) < 1e-6 for (x, y, z) in zip(a, b, c, strict=True))

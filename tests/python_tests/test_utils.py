import numpy as np
import pytest
from scipy.special import expit

from genmetaballs.core import sigmoid

NUM_RNG_SEEDS_PER_TEST = 5
NUM_N_VALUES_PER_TEST = 5
MASTER_SEED = 0


@pytest.mark.parametrize(
    "rng_seed", np.random.default_rng(MASTER_SEED).integers(0, 2**32, size=NUM_RNG_SEEDS_PER_TEST)
)
def test_sigmoid_single_value(rng_seed: int) -> None:
    """Test that sigmoid can be computed correctly for a single value."""
    rng = np.random.default_rng(rng_seed)
    # Test with a wide range of values
    x = rng.uniform(low=-10.0, high=10.0, size=1).astype(np.float32).item()

    # Compute expected result using scipy
    expected = expit(x)

    # Compute actual result using our implementation
    actual = sigmoid(x)

    # Compare results - use reasonable tolerance for float32
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-6)

    # Check that sigmoid output is within [0, 1]
    assert actual >= 0.0
    assert actual <= 1.0


@pytest.mark.parametrize(
    "x",
    [
        -1e30,
        -1e10,
        -1e-30,
        0.0,
        1e-30,
        1e10,
        1e30,
        float("-inf"),
        float("inf"),
        float("nan"),
    ],
)
def test_sigmoid_edge_cases(x: float) -> None:
    """Test sigmoid with edge case values."""
    expected = expit(x)
    actual = sigmoid(x)

    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert np.isclose(actual, expected, rtol=1e-5, atol=1e-6)
        assert actual >= 0.0
        assert actual <= 1.0

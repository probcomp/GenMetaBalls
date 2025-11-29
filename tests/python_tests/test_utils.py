import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit

from genmetaballs.core import array2d_float, sigmoid

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


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_array2d_float_creation_on_jax_devices(device: str):
    """Test creation of Array2D from a numpy array."""
    rows, cols = 4, 5
    data = jnp.arange(rows * cols, dtype=jnp.float32).reshape((rows, cols))
    jax_device = jax.devices(device)[0]
    data = jax.device_put(data, device=jax_device)
    array_2d = array2d_float(data, device=device)

    assert array_2d.num_rows == rows
    assert array_2d.num_cols == cols
    assert array_2d.ndim == 2

    # then try converting back to numpy array via view
    data_view = array_2d.as_jax()
    assert data_view.device == data.device
    assert jnp.allclose(data, data_view)

    # Note: we can't test writability of shared view here since JAX arrays are immutable


def test_float_array2d_view_numpy():
    """Test creation of Array2D from a numpy array."""
    rows, cols = 3, 4
    data = np.arange(rows * cols, dtype=np.float32).reshape((rows, cols))
    array_2d = array2d_float(data, device="cpu")

    assert array_2d.num_rows == rows
    assert array_2d.num_cols == cols
    assert array_2d.ndim == 2

    # then try converting back to numpy array via view
    data_view = array_2d.as_numpy()
    assert np.allclose(data, data_view)

    # check that the view is writable and changes reflect back to original data
    data_view[0, 0] = 999.0
    assert np.isclose(data[0, 0], 999.0)


def test_create_invalid_array2d():
    """Test that creating Array2D with invalid dimensions raises errors."""
    data = np.arange(12, dtype=np.float32).reshape((3, 4))

    with pytest.raises(TypeError):
        array2d_float(data.reshape((3, 4, 1)), device="cpu")  # not 2D

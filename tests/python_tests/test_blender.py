import numpy as np
import pytest
from scipy.special import expit

from genmetaballs.core import (
    FourParameterBlender,
)


def ground_truth_four_parameter_blender(
    beta1: float,
    beta2: float,
    beta3: float,
    eta: float,
    di: float | np.ndarray,
    ti: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute the four-parameter blender function for a single value or arrays.

    Implements:
        wi = exp( (beta1 * di) * sigmoid((beta3/eta) * ti) - (beta2/eta) * ti )
    """
    sig = expit((beta3 / eta) * ti)
    result = np.exp((beta1 * di * sig) - ((beta2 / eta) * ti))
    return result


NUM_RNG_SEEDS_PER_TEST = 5
NUM_N_VALUES_PER_TEST = 5
MASTER_SEED = 0

# Test data for different four-parameter blender configurations
BLENDER_TEST_CASES = [
    # Just use the parameter dict (no ground truth dict)
    {"beta1": 1.0, "beta2": 0.5, "beta3": 0.2, "eta": 2.0},
    {"beta1": -2.0, "beta2": 1.0, "beta3": -1.0, "eta": 1.5},
    {"beta1": 0.0, "beta2": 0.0, "beta3": 1.0, "eta": 1.0},
    {"beta1": 0.5, "beta2": -0.5, "beta3": 0.8, "eta": 0.5},
]


def create_blender_instance(kwargs: dict) -> FourParameterBlender:
    """Helper to instantiate FourParameterBlender with the provided parameters."""
    return FourParameterBlender(kwargs["beta1"], kwargs["beta2"], kwargs["beta3"], kwargs["eta"])


@pytest.mark.parametrize(
    "rng_seed", np.random.default_rng(MASTER_SEED).integers(0, 2**32, size=NUM_RNG_SEEDS_PER_TEST)
)
@pytest.mark.parametrize("blender_kwargs", BLENDER_TEST_CASES)
def test_blender_single_value(rng_seed: int, blender_kwargs: dict) -> None:
    """Test that FourParameterBlender computes correct blend values for a single value."""
    rng = np.random.default_rng(rng_seed)
    blender = create_blender_instance(blender_kwargs)

    di = rng.uniform(low=0, high=10.0, size=1).astype(np.float32).item()
    ti = rng.uniform(low=0, high=10.0, size=1).astype(np.float32).item()

    expected = ground_truth_four_parameter_blender(
        blender_kwargs["beta1"],
        blender_kwargs["beta2"],
        blender_kwargs["beta3"],
        blender_kwargs["eta"],
        di,
        ti,
    )

    actual = blender.blend(t=ti, d=di)

    # check that the actual and expected values are close
    assert np.isclose(actual, expected, rtol=1e-6) or (np.isnan(actual) and np.isnan(expected))
    # Check for finiteness (blend can in rare cases produce 0 or inf for extreme values)
    assert np.isfinite(actual)

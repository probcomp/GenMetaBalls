import numpy as np
import pytest
from scipy.special import expit

from genmetaballs.core import ThreeParameterConfidence, gpu_get_confidence


def ground_truth_five_parameter_confidence_cpu(
    beta4: float, beta5: float, sumexpd: float | np.ndarray
) -> float | np.ndarray:
    """CPU version: Compute the five parameter confidence for a single value or an array of values using numpy."""
    return expit((beta4 * sumexpd) + beta5)


def ground_truth_three_parameter_confidence_cpu(
    sumexpd: float | np.ndarray,
) -> float | np.ndarray:
    """CPU version: Compute the three parameter confidence for a single value or an array of values using numpy."""
    return 1.0 - np.exp(-sumexpd)


# According to the FMB+ code, the zero parameter confidence is the same as the three parameter confidence.
# Look at https://github.com/leonidk/fmb-plus/blob/235a078a402968554186a2ca752fb13afffb84f8/zpfm_render.py#L59
ground_truth_zero_parameter_confidence_cpu = ground_truth_three_parameter_confidence_cpu

NUM_RNG_SEEDS_PER_TEST = 5
NUM_N_VALUES_PER_TEST = 5
MASTER_SEED = 0


@pytest.mark.parametrize(
    "rng_seed", np.random.default_rng(MASTER_SEED).integers(0, 2**32, size=NUM_RNG_SEEDS_PER_TEST)
)
def test_three_parameter_confidence_single_value_cpu(rng_seed: int) -> None:
    """Test that the three parameter confidence can be computed correctly on the CPU for a single value."""
    rng = np.random.default_rng(rng_seed)
    confidence = ThreeParameterConfidence()
    # range of sumexpd is [tiny (smallest f32 value above 0), max (largest f32 value)] since it is the sum of exp(d) for all metaballs
    sumexpd = (
        rng.uniform(low=np.finfo(np.float32).tiny, high=np.finfo(np.float32).max, size=1)
        .astype(np.float32)
        .item()
    )
    expected = ground_truth_three_parameter_confidence_cpu(sumexpd)
    actual = confidence.get_confidence(sumexpd)
    # check that the actual and expected values are close
    assert np.isclose(actual, expected, rtol=1e-6)
    # check that all confidence values are between 0 and 1 inclusive
    assert actual >= 0.0 and actual <= 1.0


@pytest.mark.parametrize(
    "N",
    [
        2**k for k in range(4, 4 + NUM_N_VALUES_PER_TEST)
    ],  # [16, 32, 64, ...] up to 2 ** (4 + NUM_N_VALUES_PER_TEST - 1)
)
@pytest.mark.parametrize(
    "rng_seed", np.random.default_rng(MASTER_SEED).integers(0, 2**32, size=NUM_RNG_SEEDS_PER_TEST)
)
def test_three_parameter_confidence_multiple_values_gpu(rng_seed: int, N: int) -> None:
    """Test the three parameter confidence computed on GPU for multiple values and different N."""
    rng = np.random.default_rng(rng_seed)
    confidence = ThreeParameterConfidence()

    # Generate random sumexpd vector values, seeded as above
    sumexpd_vec = rng.uniform(
        low=np.finfo(np.float32).tiny, high=np.finfo(np.float32).max, size=N
    ).astype(np.float32)

    # Compute expected results using numpy
    expected = ground_truth_three_parameter_confidence_cpu(sumexpd_vec)

    # Compute actual results using GPU
    actual = np.array(gpu_get_confidence(sumexpd_vec.tolist(), confidence))

    # Compare results
    assert len(actual) == len(expected)
    assert all(np.isclose(a, e, rtol=1e-6) for a, e in zip(actual, expected, strict=True))
    # check that all confidence values are between 0 and 1 inclusive
    assert np.all((actual >= 0.0) & (actual <= 1.0))

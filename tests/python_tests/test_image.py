import jax.numpy as jnp
import pytest

from genmetaballs.core import make_image


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_image_creation(device: str) -> None:
    height, width = 480, 640
    image = make_image(height, width, device=device)
    assert image.num_rows == height
    assert image.num_cols == width

    # create views and check their types
    image_view = image.as_view()
    assert image_view.num_rows == height
    assert image_view.num_cols == width

    # check that the confidence and depth arrays can be converted to JAX arrays
    confidence = image_view.confidence.as_jax()
    depth = image_view.depth.as_jax()
    assert isinstance(confidence, jnp.ndarray)
    assert isinstance(depth, jnp.ndarray)

    # make sure the arrays are initialized to zero
    assert jnp.allclose(confidence, 0.0)
    assert jnp.allclose(depth, 0.0)

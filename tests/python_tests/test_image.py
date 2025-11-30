import jax.numpy as jnp

from genmetaballs.core import Image


def test_image_creation():
    height, width = 480, 640
    image = Image(height, width)
    assert image.shape == (height, width)
    assert image.confidence.shape == (height, width)
    assert image.depth.shape == (height, width)

    # check types
    assert isinstance(image.confidence, jnp.ndarray)
    assert isinstance(image.depth, jnp.ndarray)

    # make sure the arrays are initialized to zero
    assert jnp.allclose(image.confidence, 0.0)
    assert jnp.allclose(image.depth, 0.0)

import jax.numpy as jnp

from genmetaballs._genmetaballs_bindings import geometry
from genmetaballs._genmetaballs_bindings.blender import (
    FourParameterBlender,
    ThreeParameterBlender,
)
from genmetaballs._genmetaballs_bindings.camera import Intrinsics
from genmetaballs._genmetaballs_bindings.confidence import (
    TwoParameterConfidence,
    ZeroParameterConfidence,
)
from genmetaballs._genmetaballs_bindings.image import CPUImage, CPUImageView
from genmetaballs._genmetaballs_bindings.utils import CPUFloatArray2D, GPUFloatArray2D, sigmoid


def array2d_float(data, device) -> CPUFloatArray2D | GPUFloatArray2D:
    """Create a FloatArray2D on the specified device from an array.

    Args:
        data: A 2D array of type float32.
        device: 'cpu' or 'gpu' to specify the target device.
    """
    if device == "cpu":
        return CPUFloatArray2D.from_array(data)
    elif device == "gpu":
        return GPUFloatArray2D.from_array(data)
    else:
        raise ValueError(f"Unsupported device type: {device}")


class Image:
    def __init__(self, height: int, width: int) -> None:
        """Create an Image on CPU

        Unlike the C++ version, this Image class keep the buffer internally.
        This is because Python does reference counting and manage the memory
        automatically for us.

        Args:
            height: Number of rows in the image.
            width: Number of columns in the image.
        """
        self._image = CPUImage(height, width)
        # keep a view for easy access
        self._view: CPUImageView = self._image.as_view()

    @property
    def confidence(self) -> jnp.ndarray:
        """Get the confidence array."""
        return self._view.confidence.as_jax()

    @property
    def depth(self) -> jnp.ndarray:
        """Get the depth array."""
        return self._view.depth.as_jax()

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the image as (height, width)."""
        return (self._image.num_rows, self._image.num_cols)


__all__ = [
    "array2d_float",
    "ZeroParameterConfidence",
    "TwoParameterConfidence",
    "geometry",
    "Camera",
    "Intrinsics",
    "sigmoid",
    "FourParameterBlender",
    "ThreeParameterBlender",
    "Image",
]

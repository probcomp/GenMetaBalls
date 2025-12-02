from typing import Literal

from genmetaballs._genmetaballs_bindings import fmb, geometry, intersector
from genmetaballs._genmetaballs_bindings.blender import (
    FourParameterBlender,
    ThreeParameterBlender,
)
from genmetaballs._genmetaballs_bindings.camera import Intrinsics
from genmetaballs._genmetaballs_bindings.confidence import (
    TwoParameterConfidence,
    ZeroParameterConfidence,
)
from genmetaballs._genmetaballs_bindings.image import CPUImage, GPUImage
from genmetaballs._genmetaballs_bindings.utils import CPUFloatArray2D, GPUFloatArray2D, sigmoid

type DeviceType = Literal["cpu", "gpu"]


def array2d_float(data, device: DeviceType) -> CPUFloatArray2D | GPUFloatArray2D:
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


def make_image(height: int, width: int, device: DeviceType) -> CPUImage | GPUImage:
    """Create an Image on the specified device.

    Args:
        height: The height of the image.
        width: The width of the image.
        device: 'cpu' or 'gpu' to specify the target device.
    """
    if device == "cpu":
        return CPUImage(height, width)
    elif device == "gpu":
        return GPUImage(height, width)
    else:
        raise ValueError(f"Unsupported device type: {device}")


__all__ = [
    "array2d_float",
    "ZeroParameterConfidence",
    "TwoParameterConfidence",
    "fmb",
    "geometry",
    "Camera",
    "Intrinsics",
    "intersector",
    "sigmoid",
    "FourParameterBlender",
    "ThreeParameterBlender",
    "make_image",
]

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
]

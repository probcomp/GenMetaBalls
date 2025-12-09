from typing import Literal

from genmetaballs._genmetaballs_bindings import fmb, forward, geometry, intersector
from genmetaballs._genmetaballs_bindings.blender import (
    FourParameterBlender,
    ThreeParameterBlender,
)
from genmetaballs._genmetaballs_bindings.camera import Intrinsics
from genmetaballs._genmetaballs_bindings.confidence import (
    TwoParameterConfidence,
    ZeroParameterConfidence,
)
from genmetaballs._genmetaballs_bindings.fmb import FMB, CPUFMBScene, GPUFMBScene
from genmetaballs._genmetaballs_bindings.image import (
    CPUImage,
    GPUImage,
    GPUTempBuffer,
)
from genmetaballs._genmetaballs_bindings.utils import (
    CPUFloatArray2D,
    GPUFloatArray2D,
    dim3,
    sigmoid,
)

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


def make_temp_buffer(
    height: int, width: int, num_fmb_chunks: int, device: DeviceType
) -> GPUTempBuffer:
    """Create a TempBuffer on the specified device.

    Args:
        height: The height of the buffer.
        width: The width of the buffer.
        num_fmb_chunks: The number of FMB chunks to process in parallel.
        device: 'cpu' or 'gpu' to specify the target device (currently only 'gpu' supported).
    """
    if device == "gpu":
        return GPUTempBuffer(height, width, num_fmb_chunks)
    else:
        raise ValueError(f"TempBuffer currently only supports 'gpu' device, got: {device}")


def make_fmb_scene(size: int, device: DeviceType) -> CPUFMBScene | GPUFMBScene:
    """Create an FMBScene on the specified device.

    Args:
        size: The number of FMBs in the scene.
        device: 'cpu' or 'gpu' to specify the target device.
    """
    if device == "cpu":
        return CPUFMBScene(size)
    elif device == "gpu":
        return GPUFMBScene(size)
    else:
        raise ValueError(f"Unsupported device type: {device}")


# TODO: create a wrapper class for FMBScene and turn the factory functions into
# class methods
def make_fmb_scene_from_values(
    fmbs: list[fmb.FMB], log_weights: list[float], device: DeviceType
) -> CPUFMBScene | GPUFMBScene:
    if device == "cpu":
        return CPUFMBScene(fmbs, log_weights)
    elif device == "gpu":
        return GPUFMBScene(fmbs, log_weights)
    else:
        raise ValueError(f"Unsupported device type: {device}")


def render_fmbs(
    fmbs: GPUFMBScene,
    blender: FourParameterBlender | ThreeParameterBlender,
    confidence: TwoParameterConfidence | ZeroParameterConfidence,
    intr: Intrinsics,
    extr: geometry.Pose,
    img: GPUImage | None = None,
    grid_size: dim3 = dim3(4, 4),
    block_size: dim3 = dim3(16, 16),
    kernel_id: int = 0,
    block: bool = False,
    temp_buffer: GPUTempBuffer | None = None,
    num_fmb_chunks: int = 8,
) -> GPUImage:
    """Render the given FMB scene into the provided image view.

    If no image is provided, a new image with the dimensions specified by the intrinsics
    will be created.

    Args:
        kernel_id: Kernel selection ID:
            - 0: Original slow working kernel (for verification)
            - 1: 3-kernel FMB chunk parallelization (requires temp_buffer)
            - 2+: Reserved for future optimizations
        block: Whether to block the GPU until the render is complete.
        temp_buffer: Temporary buffer for kernel_id=1. If None and kernel_id=1, will be auto-created.
        num_fmb_chunks: Number of FMB chunks for kernel_id=1 (default: 8).
    """
    if img is None:
        img = make_image(intr.height, intr.width, device="gpu")

    # Auto-create temp_buffer for kernel_id=1 if not provided
    if kernel_id == 1 and temp_buffer is None:
        temp_buffer = make_temp_buffer(intr.height, intr.width, num_fmb_chunks, device="gpu")

    if isinstance(blender, FourParameterBlender):
        if isinstance(confidence, ZeroParameterConfidence):
            render_func = forward.render_fmbs_four_param_zero_confidence
        elif isinstance(confidence, TwoParameterConfidence):
            render_func = forward.render_fmbs_four_param_two_confidence
    elif isinstance(blender, ThreeParameterBlender):
        if isinstance(confidence, ZeroParameterConfidence):
            render_func = forward.render_fmbs_three_param_zero_confidence
        elif isinstance(confidence, TwoParameterConfidence):
            render_func = forward.render_fmbs_three_param_two_confidence
    else:
        raise TypeError("Unsupported blender and confidence combination.")

    render_func(
        fmbs,
        blender,
        confidence,
        intr,
        extr,
        img.as_view(),
        grid_size,
        block_size,
        kernel_id,
        block,
        temp_buffer,  # Pass the TempBuffer object, not the view
        num_fmb_chunks,
    )
    return img


__all__ = [
    "array2d_float",
    "fmb",
    "geometry",
    "intersector",
    "make_fmb_scene",
    "make_fmb_scene_from_values",
    "make_image",
    "make_temp_buffer",
    "render_fmbs",
    "sigmoid",
    "Camera",
    "FourParameterBlender",
    "FMB",
    "dim3",
    "Intrinsics",
    "ThreeParameterBlender",
    "TwoParameterConfidence",
    "ZeroParameterConfidence",
]

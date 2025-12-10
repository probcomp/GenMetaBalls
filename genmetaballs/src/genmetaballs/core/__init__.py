from typing import Literal

from genmetaballs._genmetaballs_bindings import backward, fmb, forward, geometry, intersector
from genmetaballs._genmetaballs_bindings.backward import FMBSceneGradient
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
) -> GPUImage:
    """Render the given FMB scene into the provided image view.

    If no image is provided, a new image with the dimensions specified by the intrinsics
    will be created.
    """
    if img is None:
        img = make_image(intr.height, intr.width, device="gpu")

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

    render_func(fmbs, blender, confidence, intr, extr, img.as_view())
    return img


def fwdbwd(
    fmbs: GPUFMBScene,
    blender: FourParameterBlender | ThreeParameterBlender,
    confidence: TwoParameterConfidence | ZeroParameterConfidence,
    intr: Intrinsics,
    extr: geometry.Pose,
    expected_img: GPUImage,
    output_img: GPUImage | None = None,
) -> tuple[FMBSceneGradient, GPUImage]:
    """Forward and backward pass for FMB scene rendering.

    Args:
        fmbs: The FMB scene to render
        blender: The blending function
        confidence: The confidence function
        intr: Camera intrinsics
        extr: Camera extrinsics (pose)
        expected_img: The expected/target image for computing gradients
        output_img: Optional output image. If None, a new image will be created

    Returns:
        A tuple of (gradient object, output image)
    """
    if output_img is None:
        output_img = make_image(intr.height, intr.width, device="gpu")

    grad = FMBSceneGradient(fmbs, intr, extr)

    if isinstance(blender, FourParameterBlender):
        if isinstance(confidence, ZeroParameterConfidence):
            fwdbwd_func = backward.fwdbwd_four_param_zero_confidence
        elif isinstance(confidence, TwoParameterConfidence):
            fwdbwd_func = backward.fwdbwd_four_param_two_confidence
    elif isinstance(blender, ThreeParameterBlender):
        if isinstance(confidence, ZeroParameterConfidence):
            fwdbwd_func = backward.fwdbwd_three_param_zero_confidence
        elif isinstance(confidence, TwoParameterConfidence):
            fwdbwd_func = backward.fwdbwd_three_param_two_confidence
    else:
        raise TypeError("Unsupported blender and confidence combination.")

    fwdbwd_func(fmbs, blender, confidence, intr, extr, expected_img.as_view(), grad, output_img.as_view())
    return grad, output_img


__all__ = [
    "array2d_float",
    "backward",
    "fmb",
    "fwdbwd",
    "geometry",
    "intersector",
    "make_fmb_scene",
    "make_fmb_scene_from_values",
    "make_image",
    "render_fmbs",
    "sigmoid",
    "Camera",
    "FMB",
    "FMBSceneGradient",
    "FourParameterBlender",
    "Intrinsics",
    "ThreeParameterBlender",
    "TwoParameterConfidence",
    "ZeroParameterConfidence",
]

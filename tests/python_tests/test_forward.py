import jax.numpy as jnp
import pytest
from jax.scipy.spatial.transform import Rotation

from genmetaballs.core import (
    FourParameterBlender,
    Intrinsics,
    ThreeParameterBlender,
    TwoParameterConfidence,
    ZeroParameterConfidence,
    geometry,
    make_fmb_scene,
    render_fmbs,
)


@pytest.mark.parametrize(
    "confidence", [TwoParameterConfidence(beta4=0.5, beta5=-1.0), ZeroParameterConfidence()]
)
@pytest.mark.parametrize(
    "blender",
    [
        ThreeParameterBlender(beta1=1.0, beta2=0.5, eta=2.0),
        FourParameterBlender(beta1=1.0, beta2=0.5, beta3=0.3, eta=2.0),
    ],
)
def test_render_fmbs_smoke(blender, confidence) -> None:
    """checks that render_fmbs can be called without errors with combinations of blender and confidence."""
    scene = make_fmb_scene(20, device="gpu")
    camera = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=120)
    rotation = Rotation.identity()
    pose = geometry.Pose.from_components(
        rot=geometry.Rotation.from_quat(*rotation.as_quat()),
        tran=geometry.Vec3D(0.0, 0.0, 5.0),
    )
    # Note that it's also possible to allocate the buffer ahead of time and pass it in,
    # which could be useful if we want to reuse the same image buffer for multiple renders.
    image = render_fmbs(scene, blender, confidence, camera, pose)
    img_view = image.as_view()

    assert img_view.num_rows == camera.height
    assert img_view.num_cols == camera.width
    depth_image = img_view.depth.as_jax()
    assert isinstance(depth_image, jnp.ndarray)

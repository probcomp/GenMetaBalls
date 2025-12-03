from jax.scipy.spatial.transform import Rotation

from genmetaballs.core import (
    Intrinsics,
    ThreeParameterBlender,
    TwoParameterConfidence,
    geometry,
    make_fmb_scene,
    make_image,
    render_fmbs,
)


def test_render_fmbs_smoke() -> None:
    """checks that render_fmbs can be called without errors"""
    scene = make_fmb_scene(20, device="gpu")
    blender = ThreeParameterBlender(beta1=1.0, beta2=0.5, eta=2.0)
    confidence = TwoParameterConfidence(beta4=0.5, beta5=-1.0)
    camera = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=120)
    rotation = Rotation.identity()
    pose = geometry.Pose.from_components(
        rot=geometry.Rotation.from_quat(*rotation.as_quat()),
        tran=geometry.Vec3D(0.0, 0.0, 5.0),
    )
    # allocate output buffer
    image = make_image(camera.height, camera.width, "gpu")

    render_fmbs(scene, blender, confidence, camera, pose, image.as_view())

#!/usr/bin/env python3
"""Comprehensive benchmark comparing CUDA render_fmbs vs JAX implementation.
Follows the exact setup from fuzzy_metaballs_demo.ipynb

Saves results to cache files to avoid re-running optimization.
"""

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for saving plots
# Setup environment
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pyrender
import trimesh
from jax.example_libraries import optimizers
from jax.scipy.spatial.transform import Rotation as Rot

import genmetaballs.fmb.fm_render as fm_render
from genmetaballs.core import (
    FMB,
    Intrinsics,
    ThreeParameterBlender,
    ZeroParameterConfidence,
    dim3,
    geometry,
    make_fmb_scene_from_values,
    make_image,
    make_temp_buffer,
    render_fmbs,
)
from genmetaballs.fmb.utils import DegradeLR, get_camera_rays

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

Pose, Vec3D, Rotation = geometry.Pose, geometry.Vec3D, geometry.Rotation

# Configuration from notebook
gmm_init_scale = 1.0
rand_sphere_size = 30
num_views = 20
vfov_degrees = 45
Nepochs = 10
# batch_size is calculated dynamically based on image dimensions
# Base: 800 for 64x64 (4096 rays), scales proportionately
initial_lr = 0.1
opt_shape_scale = 2.2
clip_alpha = 3.0e-8
random_seed = 42


@jax.jit
def cov_to_isostds_and_quaternion(cov):
    """Convert a 3D Gaussian's covariance matrix to an isotropic stds vector and a rotation quaternion."""
    eigvals, eigvecs = jnp.linalg.eigh(cov)
    vars_ = jnp.maximum(eigvals, 0)
    # Ensure deterministic eigenvector orientation
    for i in range(3):
        eigvecs = eigvecs.at[:, i].set(jnp.where(eigvecs[0, i] < 0, -eigvecs[:, i], eigvecs[:, i]))
    # Ensure proper rotation matrix (determinant +1)
    eigvecs = eigvecs.at[:, 0].set(
        jnp.where(jnp.linalg.det(eigvecs) < 0, -eigvecs[:, 0], eigvecs[:, 0])
    )
    quat = Rot.from_matrix(eigvecs).as_quat()
    return vars_, quat


def get_cache_path(project_root, num_fmbs, width, height):
    """Get path to cache file for given configuration."""
    cache_dir = project_root / "scripts" / "data" / "benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"benchmark_fmbs{num_fmbs}_size{width}x{height}.pkl"
    return cache_file


def load_or_run_optimization(
    mesh_file, num_fmbs, width, height, project_root, force_rerun=False, use_wandb=False
):
    """Load optimization results from cache or run optimization."""
    cache_file = get_cache_path(project_root, num_fmbs, width, height)

    if cache_file.exists() and not force_rerun:
        print(f"Loading cached optimization results from {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Running optimization (this may take a while)...")
    result = run_optimization(mesh_file, num_fmbs, width, height, use_wandb=use_wandb)

    # Save to cache
    print(f"Saving optimization results to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result


def run_optimization(mesh_file, num_fmbs, width, height, use_wandb=False):
    """Run the optimization from the notebook to get final parameters."""
    print("Loading mesh and setting up optimization...")

    # Load mesh
    if not mesh_file.exists():
        subprocess.run(
            [
                "bash",
                str(
                    (mesh_file.parent.parent / "scripts/data/download_cow.sh").resolve().absolute()
                ),
            ]
        )

    mesh_tri = trimesh.load(mesh_file)

    # Mesh statistics
    shape_scale = float(mesh_tri.vertices.std(0).mean()) * 3
    center = np.array(mesh_tri.vertices.mean(0))
    shape_scale_mul = opt_shape_scale / shape_scale

    print(f"  Shape scale: {shape_scale:.4f}")
    print(f"  Center: {center}")

    # Setup camera parameters
    image_size = (height, width)
    focal_length = 0.5 * image_size[0] / np.tan((np.pi / 180.0) * vfov_degrees / 2)
    cx = (image_size[1] - 1) / 2
    cy = (image_size[0] - 1) / 2

    # Calculate batch_size dynamically based on image dimensions
    # Base: 800 for 64x64 (4096 rays), scales proportionately
    base_rays = 64 * 64  # 4096 rays
    current_rays = width * height
    batch_size = int(800 * (current_rays / base_rays))

    # Generate random camera poses
    np.random.seed(random_seed)
    rand_quats = np.random.randn(num_views, 4)
    rand_quats = rand_quats / np.linalg.norm(rand_quats, axis=1, keepdims=True)

    # Render reference views using pyrender (like notebook)
    mesh = pyrender.Mesh.from_trimesh(mesh_tri)
    ref_colors = []
    ref_depths = []
    scene = pyrender.Scene()
    scene.add(mesh)

    trans = []
    for quat in rand_quats:
        R = Rot.from_quat(quat).as_matrix()
        loc = np.array([0, 0, 3 * shape_scale]) @ R + center
        trans.append(loc)
        pose = np.vstack([np.vstack([R, loc]).T, np.array([0, 0, 0, 1])])

        light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=50.0,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )
        scene.add(light, pose=pose)

        camera = pyrender.IntrinsicsCamera(
            focal_length, focal_length, cx, cy, znear=0.1 * shape_scale, zfar=100 * shape_scale
        )
        scene.add(camera, pose=pose)

        r = pyrender.OffscreenRenderer(image_size[1], image_size[0])
        color, target_depth = r.render(scene)
        target_depth[target_depth == 0] = np.nan
        ref_colors.append(color)
        ref_depths.append(target_depth)

        for node in list(scene.light_nodes):
            scene.remove_node(node)
        for node in list(scene.camera_nodes):
            scene.remove_node(node)
        r.delete()

    # Create target silhouettes (like notebook)
    target_sil = (~np.isnan(ref_depths)).astype(np.float32)

    # Initialize FMBs
    np.random.seed(random_seed)
    rand_mean = center + np.random.multivariate_normal(
        mean=[0, 0, 0], cov=1e-2 * np.identity(3) * shape_scale, size=num_fmbs
    )
    rand_weight_log = jnp.log(np.ones(num_fmbs) / num_fmbs) + jnp.log(gmm_init_scale)
    rand_prec = jnp.array(
        [np.identity(3) * rand_sphere_size / shape_scale for _ in range(num_fmbs)]
    )

    # Setup camera rays
    pixel_list = (
        (np.array(np.meshgrid(np.arange(width), height - np.arange(height) - 1, [0]))[:, :, :, 0])
        .reshape((3, -1))
        .T
    )
    camera_rays = get_camera_rays(focal_length, focal_length, cx, cy, pixel_list)
    cameras_list = []
    for tran, quat in zip(trans, rand_quats, strict=False):
        R = Rot.from_quat(quat).as_matrix()
        camera_rays2 = camera_rays @ R
        t = np.tile(tran[None], (camera_rays2.shape[0], 1))
        rays_trans = np.stack([camera_rays2, t], 1)
        cameras_list.append(rays_trans)

    # Get hyperparameters
    hyperparams = fm_render.hyperparams
    beta2 = jnp.float32(np.exp(hyperparams[0]))
    beta3 = jnp.float32(np.exp(hyperparams[1]))

    # Setup optimization (simplified - just run a few iterations for testing)
    render_jit = jax.jit(fm_render.render_func_rays)

    def objective(params, true_alpha):
        means, prec, weights_log, camera_rays, beta2, beta3 = params
        render_res = render_jit(means, prec, weights_log, camera_rays, beta2, beta3)
        est_alpha = render_res[1]
        est_alpha = jnp.clip(est_alpha, clip_alpha, 1 - clip_alpha)
        mask_loss = -((true_alpha * jnp.log(est_alpha)) + (1 - true_alpha) * jnp.log(1 - est_alpha))
        return mask_loss.mean()

    grad_render3 = jax.jit(jax.value_and_grad(objective))

    # Prepare data (like notebook)
    all_cameras = jnp.array(cameras_list).reshape((-1, 2, 3))
    all_sils = jnp.array(target_sil.ravel()).astype(jnp.float32)
    Niter_epoch = int(np.ceil(len(all_cameras) / batch_size))

    # Setup optimizer with learning rate schedule (like notebook)
    vecM = jnp.array([[1, 1, 1], [shape_scale_mul, shape_scale_mul, shape_scale_mul]])[None]

    def irc(x):
        return int(round(x))

    lr_decay_p_thresh = 0.5
    lr_decay_window = 10
    lr_decay_p_window = 5
    lr_decay_slope_less = -1.0e-4
    lr_decay_max_drops = 4

    adjust_lr = DegradeLR(
        initial_lr,
        lr_decay_p_thresh,
        irc(Niter_epoch * 0.4),
        irc(lr_decay_p_window),
        lr_decay_slope_less,
        lr_decay_max_drops,
    )
    opt_init, opt_update, opt_params = optimizers.adam(adjust_lr.step_func)
    tmp = [rand_mean * shape_scale_mul, rand_prec / shape_scale_mul, rand_weight_log]
    opt_state = opt_init(tmp)

    # Warmup gradient computation (like notebook)
    print("Warming up backward pass JIT compilation...")
    p = opt_params(opt_state)
    idx_sample = jnp.array(list(range(min(batch_size, len(all_cameras)))))

    for _ in range(3):
        val, g = grad_render3(
            [p[0], p[1], p[2], vecM * all_cameras[idx_sample], beta2 / opt_shape_scale, beta3],
            all_sils[idx_sample],
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)
    print("✓ Backward pass JIT warmup complete")

    # Run full optimization (like notebook)
    print(f"Running optimization: {Nepochs} epochs, {Niter_epoch} iterations/epoch...")
    rand_idx = np.arange(len(all_cameras))
    losses = []
    done = False
    iteration_count = 0

    opt_start_time = time.perf_counter()

    for i in range(Nepochs):
        np.random.shuffle(rand_idx)
        rand_idx_jnp = jnp.array(rand_idx)

        epoch_start = time.perf_counter()

        for j in range(Niter_epoch):
            p = opt_params(opt_state)
            idx = jax.lax.dynamic_slice(rand_idx_jnp, [j * batch_size], [batch_size])

            # Forward + backward pass
            val, g = grad_render3(
                [p[0], p[1], p[2], vecM * all_cameras[idx], beta2 / opt_shape_scale, beta3],
                all_sils[idx],
            )
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)

            opt_state = opt_update(i, g[:3], opt_state)
            val = float(val)
            losses.append(val)
            iteration_count += 1

            if adjust_lr.add(val):
                done = True
                break

        epoch_time = (time.perf_counter() - epoch_start) * 1000
        print(f"Epoch {i + 1:2d}/{Nepochs} | Loss: {losses[-1]:.4f} | Time: {epoch_time:6.1f} ms")

        if done:
            print("\n✓ Early stopping triggered")
            break

    opt_total_time = (time.perf_counter() - opt_start_time) * 1000
    print("\n📊 Optimization Complete:")
    print(f"   Total time: {opt_total_time:.1f} ms ({opt_total_time / 1000:.2f} seconds)")
    print(f"   Total iterations: {iteration_count}")
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Initial loss: {losses[0]:.6f}")
    print(f"   Loss reduction: {(1 - losses[-1] / losses[0]) * 100:.1f}%")

    # Log optimization results to wandb if enabled
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.log(
                {
                    "optimization/total_time_ms": opt_total_time,
                    "optimization/total_time_sec": opt_total_time / 1000,
                    "optimization/total_iterations": iteration_count,
                    "optimization/final_loss": losses[-1],
                    "optimization/initial_loss": losses[0],
                    "optimization/loss_reduction_pct": (1 - losses[-1] / losses[0]) * 100,
                }
            )
        except Exception as e:
            print(f"Warning: Failed to log optimization results to wandb: {e}")

    # Get final parameters
    final_params = opt_params(opt_state)
    final_means = final_params[0] / shape_scale_mul
    final_precs = final_params[1] * shape_scale_mul
    final_weight_logs = final_params[2]

    return {
        "final_means": final_means,
        "final_precs": final_precs,
        "final_weight_logs": final_weight_logs,
        "shape_scale": shape_scale,
        "center": center,
        "cameras_list": cameras_list,
        "trans": trans,
        "rand_quats": rand_quats,
        "focal_length": focal_length,
        "cx": cx,
        "cy": cy,
        "beta2": beta2,
        "beta3": beta3,
        "width": width,
        "height": height,
        "num_fmbs": num_fmbs,
        "ref_depths": ref_depths,  # Save reference depths for GT comparison
        "target_sil": target_sil,  # Save target silhouettes
    }


def benchmark_comparison(
    opt_results,
    kernel_id=0,
    warmup=10,
    save_plot=False,
    grid_size=None,
    block_size=None,
    num_fmb_chunks=4,
    use_wandb=False,
):
    """Benchmark and compare CUDA vs JAX implementations using notebook setup."""
    # Set defaults for grid_size and block_size
    if grid_size is None:
        grid_size = dim3(4, 4)
    else:
        grid_size = dim3(grid_size[0], grid_size[1])

    if block_size is None:
        block_size = dim3(16, 16)
    else:
        block_size = dim3(block_size[0], block_size[1])

    kernel_names = {
        0: "original (slow working)",
        1: "3-kernel FMB chunk parallelization",
    }
    kernel_name = kernel_names.get(kernel_id, f"kernel_{kernel_id}")

    print("=" * 80)
    print("BENCHMARK: CUDA render_fmbs vs JAX Implementation")
    print("=" * 80)
    print("Following exact setup from fuzzy_metaballs_demo.ipynb")
    print(f"  FMBs: {opt_results['num_fmbs']}")
    print(f"  Image size: {opt_results['width']}x{opt_results['height']}")
    print(f"  Views: {num_views}")
    print(f"  CUDA Kernel ID: {kernel_id} ({kernel_name})")
    print(f"  Warmup iterations: {warmup}")
    print(f"  Benchmark: {len(opt_results['cameras_list'])} camera views (one iteration per view)")
    print()

    final_means = opt_results["final_means"]
    final_precs = opt_results["final_precs"]
    final_weight_logs = opt_results["final_weight_logs"]
    shape_scale = opt_results["shape_scale"]
    cameras_list = opt_results["cameras_list"]
    trans = opt_results["trans"]
    rand_quats = opt_results["rand_quats"]
    focal_length = opt_results["focal_length"]
    cx = opt_results["cx"]
    cy = opt_results["cy"]
    beta2 = opt_results["beta2"]
    beta3 = opt_results["beta3"]
    width = opt_results["width"]
    height = opt_results["height"]

    # Create random parameters for JAX speed benchmarking (matching notebook Cell 23)
    # NOTE: Notebook uses random parameters for JAX speed benchmarking, not trained
    np.random.seed(random_seed)

    # NOTE: For plots/accuracy, we use TRAINED parameters (matching notebook Cell 20)
    # For speed benchmarking, we use TRAINED parameters for both JAX and CUDA (fair comparison)
    print("Setting up parameters...")
    print("  For accuracy/plots: using TRAINED (optimized) parameters from PKL")
    print("  For speed benchmarking: Both JAX and CUDA use trained parameters (fair comparison)")

    # Verify we have ref_depths for GT comparison
    if "ref_depths" not in opt_results:
        print("  WARNING: ref_depths not found in cache, GT depth will not be available in plots")

    # Create CUDA scene from optimized parameters (like notebook Cell 22)
    print("Creating CUDA FMB scene from optimized parameters...")
    fmbs = []
    for final_mean, final_prec in zip(final_means, final_precs, strict=True):
        prec_matrix = final_prec @ final_prec.T
        stds, quat = cov_to_isostds_and_quaternion(jnp.linalg.inv(prec_matrix))
        pose = Pose.from_components(Rotation.from_quat(*quat), Vec3D(*final_mean))
        fmbs.append(FMB(pose, *stds))

    cuda_scene = make_fmb_scene_from_values(fmbs, list(final_weight_logs), device="gpu")

    # CUDA camera setup
    cuda_intr = Intrinsics(
        fx=focal_length, fy=focal_length, cx=cx, cy=cy, width=width, height=height
    )
    cuda_blender = ThreeParameterBlender(beta1=beta3, beta2=beta2, eta=shape_scale)
    cuda_confidence = ZeroParameterConfidence()

    # JIT compile JAX render function
    render_jit = jax.jit(fm_render.render_func_rays)

    # Create alpha_results_final BEFORE benchmarking (matching notebook Cell 20)
    # This is used for the confidence/alpha plots
    print("Rendering alpha_results_final for plots (matching notebook Cell 20)...")
    alpha_results_final = []
    for camera_rays in tqdm(cameras_list, desc="Rendering alphas"):
        est_depth, est_alpha, est_norm, est_w = render_jit(
            final_means, final_precs, final_weight_logs, camera_rays, beta2 / shape_scale, beta3
        )
        est_alpha.block_until_ready()
        image_size = (height, width)
        alpha_results_final.append(est_alpha.reshape(image_size))

    # Preallocate CUDA image object to exclude allocation time from benchmark
    cuda_image = make_image(height, width, device="gpu")

    # Create temp_buffer for kernel_id=1
    cuda_temp_buffer = None
    if kernel_id == 1:
        print(f"Creating temp_buffer for kernel_id=1 (num_fmb_chunks={num_fmb_chunks})...")
        cuda_temp_buffer = make_temp_buffer(height, width, num_fmb_chunks, device="gpu")

    # Warmup each camera pose for both implementations
    print("Warming up each camera pose...")
    for view_idx in tqdm(range(len(cameras_list)), desc="Warmup"):
        # Warmup JAX for this camera (using TRAINED parameters)
        for _ in range(warmup):
            result = render_jit(
                final_means,
                final_precs,
                final_weight_logs,
                cameras_list[view_idx],
                beta2 / shape_scale,
                beta3,
            )
            result[1].block_until_ready()

        # Warmup CUDA for this camera
        cuda_extr = Pose.from_components(
            rot=Rotation.from_quat(*rand_quats[view_idx]).inv(), tran=Vec3D(*trans[view_idx])
        )
        for _ in range(warmup):
            render_fmbs(
                cuda_scene,
                cuda_blender,
                cuda_confidence,
                cuda_intr,
                cuda_extr,
                img=cuda_image,
                grid_size=grid_size,
                block_size=block_size,
                kernel_id=kernel_id,
                temp_buffer=cuda_temp_buffer,
                num_fmb_chunks=num_fmb_chunks,
            )
            _ = cuda_image.as_view().depth.as_jax().block_until_ready()

    print("\nRunning benchmarks: 1000 iterations per view (no time limit, matching notebook)...")

    # Benchmark: iterate over each camera pose, run 1000 iterations per pose (matching notebook)
    # NOTE: Notebook has NO time limit - it runs all iterations
    ITER_MULT = 1000  # Match notebook's iter_mult = 1000

    jax_times = []
    cuda_times = []
    cuda_results = []  # List of (depth, alpha) tuples for each view (saved on first iteration)

    # Benchmark JAX: all iterations, no time limit (using trained parameters for fair comparison)
    # NOTE: Changed from random to trained parameters to match CUDA benchmarking
    print("Benchmarking FMB-JAX (using trained parameters for speed, matching CUDA)...")
    for view_idx in tqdm(range(len(cameras_list)), desc="FMB-JAX"):
        camera_rays = cameras_list[view_idx]

        # Benchmark JAX for this camera pose (ITER_MULT iterations, no time limit)
        for _ in range(ITER_MULT):
            beta2_div_shape_scale = beta2 / shape_scale

            start_time = time.time_ns()  # Use time_ns like notebook
            est_depth, est_alpha, est_norm, est_w = render_jit(
                final_means,
                final_precs,
                final_weight_logs,
                camera_rays,
                beta2_div_shape_scale,
                beta3,
            )
            est_alpha.block_until_ready()
            end_time = time.time_ns()
            elapsed = (end_time - start_time) / 1e3  # microseconds, like notebook
            jax_times.append(elapsed)  # Store in microseconds like notebook

            # NOTE: For accuracy comparison, we need results with TRAINED parameters
            # Save trained results separately (not during speed benchmark)
            # We'll use alpha_results_final (already created above) for plots

    # Benchmark CUDA: all iterations, no time limit (matching notebook)
    print("\nBenchmarking GenMetaBalls CUDA...")
    kernel_timings_list = []  # Store individual kernel timings for kernel_id=1

    for view_idx in tqdm(range(len(cameras_list)), desc="GenMetaBalls"):
        # Setup camera pose
        cuda_extr = Pose.from_components(
            rot=Rotation.from_quat(*rand_quats[view_idx]).inv(), tran=Vec3D(*trans[view_idx])
        )

        # Benchmark CUDA for this camera pose (ITER_MULT iterations, no time limit)
        for iter_idx in range(ITER_MULT):
            # For kernel_id=1, collect timing info for all iterations
            if kernel_id == 1:
                timings = render_fmbs(
                    cuda_scene,
                    cuda_blender,
                    cuda_confidence,
                    cuda_intr,
                    cuda_extr,
                    img=cuda_image,
                    grid_size=grid_size,
                    block_size=block_size,
                    kernel_id=kernel_id,
                    block=True,
                    temp_buffer=cuda_temp_buffer,
                    num_fmb_chunks=num_fmb_chunks,
                    return_timings=True,
                )
                kernel_timings_list.append(timings)
                # Use total time from timings for consistency
                cuda_times.append(timings.total_us)
            else:
                # Regular timing measurement
                start_time = time.time_ns()  # Use time_ns like notebook
                render_fmbs(
                    cuda_scene,
                    cuda_blender,
                    cuda_confidence,
                    cuda_intr,
                    cuda_extr,
                    img=cuda_image,
                    grid_size=grid_size,
                    block_size=block_size,
                    kernel_id=kernel_id,
                    block=True,
                    temp_buffer=cuda_temp_buffer,
                    num_fmb_chunks=num_fmb_chunks,
                    return_timings=False,
                )
                end_time = time.time_ns()
                elapsed = (end_time - start_time) / 1e3  # microseconds, like notebook
                cuda_times.append(elapsed)  # Store in microseconds like notebook

            # Save result on first iteration - keep as JAX arrays like notebook
            # NOTE: Store RAW depth (unfiltered) - filtering happens in plotting
            if iter_idx == 0:
                img_view = cuda_image.as_view()
                depth_image = jnp.copy(img_view.depth.as_jax())  # Use jnp.copy like notebook
                confidence = jnp.copy(img_view.confidence.as_jax())
                cuda_results.append(
                    (depth_image, confidence)
                )  # Keep as JAX arrays (RAW depth, not filtered)

    # Statistics - calculate average like notebook: total_time / len(cameras_list) / iter_mult
    jax_times = np.array(jax_times)
    cuda_times = np.array(cuda_times)
    total_jax_time_us = jax_times.sum()
    total_cuda_time_us = cuda_times.sum()

    # Calculate average like notebook: total_time_us / len(cameras_list) / iter_mult
    jax_avg_time_us = total_jax_time_us / len(cameras_list) / ITER_MULT
    cuda_avg_time_us = total_cuda_time_us / len(cameras_list) / ITER_MULT

    print("\n" + "=" * 80)
    print("SPEED RESULTS:")
    print("=" * 80)

    print("FMB-JAX Implementation:")
    print(
        f"  Time taken for {len(cameras_list) * ITER_MULT} views: {total_jax_time_us:.2f} microseconds"
    )
    print(f"  FPS:               {(len(cameras_list) * ITER_MULT) / total_jax_time_us * 1e6:.2f}")
    print(f"  Average render time per view: {jax_avg_time_us:.2f} microseconds")
    print(f"  Standard deviation: {jax_times.std():.2f} microseconds")
    print()
    print(f"GenMetaBalls CUDA Implementation (kernel_id={kernel_id}, {kernel_name}):")
    print(
        f"  Time taken for {len(cameras_list) * ITER_MULT} views: {total_cuda_time_us:.2f} microseconds"
    )
    print(f"  FPS:               {(len(cameras_list) * ITER_MULT) / total_cuda_time_us * 1e6:.2f}")
    print(f"  Average render time per view: {cuda_avg_time_us:.2f} microseconds")
    print(f"  Standard deviation: {cuda_times.std():.2f} microseconds")

    # Print individual kernel timings for kernel_id=1 (averaged across all iterations)
    if kernel_id == 1 and len(kernel_timings_list) > 0:
        # Compute mean across all collected timings
        mean_1a = np.mean([t.kernel_1a_us for t in kernel_timings_list])
        mean_1b = np.mean([t.kernel_1b_us for t in kernel_timings_list])
        mean_1c = np.mean([t.kernel_1c_us for t in kernel_timings_list])
        mean_total = np.mean([t.total_us for t in kernel_timings_list])

        # Compute standard deviations
        std_1a = np.std([t.kernel_1a_us for t in kernel_timings_list])
        std_1b = np.std([t.kernel_1b_us for t in kernel_timings_list])
        std_1c = np.std([t.kernel_1c_us for t in kernel_timings_list])
        std_total = np.std([t.total_us for t in kernel_timings_list])

        print()
        print("=" * 50)
        print(
            f"INDIVIDUAL KERNEL TIMINGS (kernel_id=1) - AVERAGED OVER {len(kernel_timings_list)} ITERATIONS:"
        )
        print("=" * 50)
        print(f"  Kernel 1a (FMB chunk processing): {mean_1a:.2f} ± {std_1a:.2f} μs")
        print(f"  Kernel 1b (Reduction):            {mean_1b:.2f} ± {std_1b:.2f} μs")
        print(f"  Kernel 1c (Finalization):         {mean_1c:.2f} ± {std_1c:.2f} μs")
        print(f"  Total (1a + 1b + 1c):             {mean_total:.2f} ± {std_total:.2f} μs")
        print("  Breakdown (based on mean):")
        if mean_total > 0:
            print(f"    Kernel 1a: {mean_1a / mean_total * 100:.1f}%")
            print(f"    Kernel 1b: {mean_1b / mean_total * 100:.1f}%")
            print(f"    Kernel 1c: {mean_1c / mean_total * 100:.1f}%")
        print("=" * 50)

        # Log kernel timings to wandb if enabled
        if use_wandb:
            wandb.log(
                {
                    "kernel_timings/kernel_1a_us_mean": float(mean_1a),
                    "kernel_timings/kernel_1a_us_std": float(std_1a),
                    "kernel_timings/kernel_1b_us_mean": float(mean_1b),
                    "kernel_timings/kernel_1b_us_std": float(std_1b),
                    "kernel_timings/kernel_1c_us_mean": float(mean_1c),
                    "kernel_timings/kernel_1c_us_std": float(std_1c),
                    "kernel_timings/total_us_mean": float(mean_total),
                    "kernel_timings/total_us_std": float(std_total),
                    "kernel_timings/kernel_1a_pct": mean_1a / mean_total * 100
                    if mean_total > 0
                    else 0,
                    "kernel_timings/kernel_1b_pct": mean_1b / mean_total * 100
                    if mean_total > 0
                    else 0,
                    "kernel_timings/kernel_1c_pct": mean_1c / mean_total * 100
                    if mean_total > 0
                    else 0,
                }
            )

    print()
    speedup = jax_avg_time_us / cuda_avg_time_us
    print()
    print("=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"GenMetaBalls average: {cuda_avg_time_us:.2f} μs")
    print(f"JAX render_jit average: {jax_avg_time_us:.2f} μs")
    print(f"Speedup (JAX/GMB): {speedup:.2f}x")
    if speedup > 1:
        print(f"GenMetaBalls is {speedup:.2f}x faster than JAX")
    else:
        print(f"JAX is {1 / speedup:.2f}x faster than GenMetaBalls")
    print("=" * 50)

    # Log performance metrics to wandb if enabled
    if use_wandb:
        wandb.log(
            {
                "performance/jax_avg_time_us": jax_avg_time_us,
                "performance/jax_std_time_us": float(jax_times.std()),
                "performance/cuda_avg_time_us": cuda_avg_time_us,
                "performance/cuda_std_time_us": float(cuda_times.std()),
                "performance/speedup": speedup,
                "performance/jax_fps": (len(cameras_list) * ITER_MULT) / total_jax_time_us * 1e6,
                "performance/cuda_fps": (len(cameras_list) * ITER_MULT) / total_cuda_time_us * 1e6,
            }
        )
        # Log timing histograms
        wandb.log(
            {
                "performance/jax_times_hist": wandb.Histogram(jax_times),
                "performance/cuda_times_hist": wandb.Histogram(cuda_times),
            }
        )

    # For accuracy comparison, we need to render with TRAINED parameters
    # Create jax_results with trained parameters (matching notebook Cell 20)
    print("\nRendering JAX results with trained parameters for accuracy comparison...")
    jax_results_trained = []
    for camera_rays in tqdm(cameras_list, desc="JAX trained render"):
        est_depth, est_alpha, est_norm, est_w = render_jit(
            final_means, final_precs, final_weight_logs, camera_rays, beta2 / shape_scale, beta3
        )
        est_alpha.block_until_ready()
        image_size = (height, width)
        # Filter depth BEFORE reshaping (matching notebook Cell 20)
        est_depth_np = np.array(est_depth)
        est_alpha_np = np.array(est_alpha)
        est_depth_np[est_alpha_np < 0.5] = np.nan
        jax_results_trained.append(
            (est_depth_np.reshape(image_size), est_alpha_np.reshape(image_size))
        )

    # Accuracy comparison - aggregate over all views
    # Both JAX and CUDA use the same trained parameters
    print("\n" + "=" * 80)
    print("ACCURACY RESULTS (aggregated over all views):")
    print("=" * 80)
    print("Note: Comparing outputs from same trained parameters (both JAX and CUDA)...")

    # Collect all depth and alpha differences across all views
    all_depth_diffs = []
    all_depth_relative_errors = []
    all_alpha_diffs = []
    all_alpha_relative_errors = []
    total_valid_pixels = 0
    total_pixels = 0

    # Only compare views that we actually have results for
    num_views_with_results = min(len(jax_results_trained), len(cuda_results))

    for view_idx in range(num_views_with_results):
        jax_depth, jax_alpha = jax_results_trained[view_idx]
        cuda_depth, cuda_alpha = cuda_results[view_idx]

        # Convert JAX arrays to NumPy for comparison
        # NOTE: jax_depth is already FILTERED (has NaN where alpha < 0.5) from benchmark loop
        jax_depth_img = np.array(jax_depth)  # Already filtered
        jax_alpha_img = np.array(jax_alpha)
        cuda_depth_img = np.array(
            cuda_depth.reshape(height, width) if len(cuda_depth.shape) == 1 else cuda_depth
        )
        cuda_alpha_img = np.array(
            cuda_alpha.reshape(height, width) if len(cuda_alpha.shape) == 1 else cuda_alpha
        )

        # Compare depth following notebook logic:
        # For GenMetaBalls: filter depth where confidence < 0.5 (set to NaN)
        # For FMB-JAX: already filtered (has NaN where alpha < 0.5)
        # Then compare where both have valid (non-NaN) depths
        cuda_depth_filtered = cuda_depth_img.copy()
        cuda_depth_filtered[cuda_alpha_img < 0.5] = np.nan

        jax_depth_filtered = jax_depth_img  # Already filtered, no need to filter again

        # Compare where both have valid filtered depths (not NaN)
        valid_mask = ~(np.isnan(jax_depth_filtered) | np.isnan(cuda_depth_filtered))

        total_valid_pixels += valid_mask.sum()
        total_pixels += valid_mask.size

        if valid_mask.sum() > 0:
            depth_diff = np.abs(jax_depth_filtered[valid_mask] - cuda_depth_filtered[valid_mask])
            all_depth_diffs.extend(depth_diff.tolist())
            depth_relative_error = depth_diff / (np.abs(jax_depth_filtered[valid_mask]) + 1e-8)
            all_depth_relative_errors.extend(depth_relative_error.tolist())

        # Compare alpha/confidence
        alpha_diff = np.abs(jax_alpha_img - cuda_alpha_img)
        all_alpha_diffs.extend(alpha_diff.tolist())
        alpha_relative_error = alpha_diff / (jax_alpha_img + 1e-8)
        all_alpha_relative_errors.extend(alpha_relative_error.tolist())

    # Compute aggregate statistics
    if len(all_depth_diffs) > 0:
        all_depth_diffs = np.array(all_depth_diffs)
        all_depth_relative_errors = np.array(all_depth_relative_errors)
        depth_mae = np.mean(all_depth_diffs)
        depth_max_diff = np.max(all_depth_diffs)
        depth_relative_error = np.mean(all_depth_relative_errors)
        depth_median_relative_error = np.median(all_depth_relative_errors)

        print(
            f"Depth Comparison (valid pixels: {total_valid_pixels}/{total_pixels} across {num_views_with_results} views):"
        )
        print(f"  Mean Absolute Error (MAE):     {depth_mae:.6f}")
        print(f"  Max Absolute Error:             {depth_max_diff:.6f}")
        print(f"  Mean Relative Error:            {depth_relative_error * 100:.4f}%")
        print(f"  Relative Error (median):       {depth_median_relative_error * 100:.4f}%")
    else:
        print("Depth Comparison: No valid pixels to compare")
        depth_mae = None
        depth_relative_error = None
        depth_max_diff = None

    if len(all_alpha_diffs) > 0:
        all_alpha_diffs = np.array(all_alpha_diffs)
        all_alpha_relative_errors = np.array(all_alpha_relative_errors)
        # Filter out NaN values
        valid_alpha_diffs = all_alpha_diffs[~np.isnan(all_alpha_diffs)]
        valid_alpha_rel_errors = all_alpha_relative_errors[~np.isnan(all_alpha_relative_errors)]

        if len(valid_alpha_diffs) > 0:
            alpha_mae = np.mean(valid_alpha_diffs)
            alpha_max_diff = np.max(valid_alpha_diffs)
            alpha_relative_error = (
                np.mean(valid_alpha_rel_errors) if len(valid_alpha_rel_errors) > 0 else 0.0
            )
            alpha_median_relative_error = (
                np.median(valid_alpha_rel_errors) if len(valid_alpha_rel_errors) > 0 else 0.0
            )

            print(f"\nAlpha/Confidence Comparison (across {num_views_with_results} views):")
            print(f"  Mean Absolute Error (MAE):     {alpha_mae:.6f}")
            print(f"  Max Absolute Error:             {alpha_max_diff:.6f}")
            print(f"  Mean Relative Error:            {alpha_relative_error * 100:.4f}%")
            print(f"  Relative Error (median):        {alpha_median_relative_error * 100:.4f}%")
        else:
            alpha_mae = None
            alpha_relative_error = None
            alpha_max_diff = None
    else:
        alpha_mae = None
        alpha_relative_error = None
        alpha_max_diff = None

    # Check if results are within tolerance
    depth_tolerance = 5e-3  # 0.5% relative error
    alpha_tolerance = 1e-3  # 0.1% absolute error

    depth_ok = (
        depth_relative_error < depth_tolerance
        if (len(all_depth_diffs) > 0 and depth_relative_error is not None)
        else True
    )
    alpha_ok = alpha_mae < alpha_tolerance if alpha_mae is not None else True

    print(f"\nTolerance Check (depth: {depth_tolerance * 100}% rel, alpha: {alpha_tolerance} abs):")
    print(f"  Depth: {'✓ PASS' if depth_ok else '✗ FAIL'}")
    print(f"  Alpha: {'✓ PASS' if alpha_ok else '✗ FAIL'}")
    print("=" * 80)

    # Log accuracy metrics to wandb if enabled
    if use_wandb:
        accuracy_metrics = {
            "accuracy/depth_ok": depth_ok,
            "accuracy/alpha_ok": alpha_ok,
        }
        if depth_mae is not None:
            accuracy_metrics.update(
                {
                    "accuracy/depth_mae": depth_mae,
                    "accuracy/depth_max_diff": depth_max_diff,
                    "accuracy/depth_relative_error": depth_relative_error * 100,
                    "accuracy/depth_median_relative_error": depth_median_relative_error * 100,
                    "accuracy/valid_pixels": total_valid_pixels,
                    "accuracy/total_pixels": total_pixels,
                }
            )
        if alpha_mae is not None:
            accuracy_metrics.update(
                {
                    "accuracy/alpha_mae": alpha_mae,
                    "accuracy/alpha_max_diff": alpha_max_diff,
                    "accuracy/alpha_relative_error": alpha_relative_error * 100,
                    "accuracy/alpha_median_relative_error": alpha_median_relative_error * 100,
                }
            )
        wandb.log(accuracy_metrics)

    # Log accuracy metrics to wandb if enabled
    if use_wandb and WANDB_AVAILABLE:
        try:
            accuracy_metrics = {
                "accuracy/depth_ok": depth_ok,
                "accuracy/alpha_ok": alpha_ok,
            }
            if depth_mae is not None:
                accuracy_metrics.update(
                    {
                        "accuracy/depth_mae": depth_mae,
                        "accuracy/depth_max_diff": depth_max_diff,
                        "accuracy/depth_relative_error": depth_relative_error * 100,
                        "accuracy/depth_median_relative_error": depth_median_relative_error * 100,
                        "accuracy/valid_pixels": total_valid_pixels,
                        "accuracy/total_pixels": total_pixels,
                    }
                )
            if alpha_mae is not None:
                accuracy_metrics.update(
                    {
                        "accuracy/alpha_mae": alpha_mae,
                        "accuracy/alpha_max_diff": alpha_max_diff,
                        "accuracy/alpha_relative_error": alpha_relative_error * 100,
                        "accuracy/alpha_median_relative_error": alpha_median_relative_error * 100,
                    }
                )
            wandb.log(accuracy_metrics)
        except Exception as e:
            print(f"Warning: Failed to log accuracy metrics to wandb: {e}")

    # Create comparison plots if requested (similar to notebook)
    # Use trained parameters for plots
    if save_plot:
        print("\nGenerating comparison plots...")
        create_comparison_plots(
            jax_results_trained,
            cuda_results,
            alpha_results_final,
            opt_results,
            kernel_id,
            kernel_name,
            use_wandb=False,  # do not log plots to wandb
        )

    # Save results - convert JAX arrays to lists for JSON serialization
    jax_results_list = [
        (np.array(depth).tolist(), np.array(alpha).tolist()) for depth, alpha in jax_results_trained
    ]
    cuda_results_list = [
        (np.array(depth).tolist(), np.array(alpha).tolist()) for depth, alpha in cuda_results
    ]

    # Prepare kernel timings for results dict (averaged values)
    kernel_timings_dict = None
    if kernel_id == 1 and len(kernel_timings_list) > 0:
        mean_1a = np.mean([t.kernel_1a_us for t in kernel_timings_list])
        mean_1b = np.mean([t.kernel_1b_us for t in kernel_timings_list])
        mean_1c = np.mean([t.kernel_1c_us for t in kernel_timings_list])
        mean_total = np.mean([t.total_us for t in kernel_timings_list])
        std_1a = np.std([t.kernel_1a_us for t in kernel_timings_list])
        std_1b = np.std([t.kernel_1b_us for t in kernel_timings_list])
        std_1c = np.std([t.kernel_1c_us for t in kernel_timings_list])
        std_total = np.std([t.total_us for t in kernel_timings_list])

        kernel_timings_dict = {
            "kernel_1a_us_mean": float(mean_1a),
            "kernel_1b_us_mean": float(mean_1b),
            "kernel_1c_us_mean": float(mean_1c),
            "total_us_mean": float(mean_total),
            "kernel_1a_us_std": float(std_1a),
            "kernel_1b_us_std": float(std_1b),
            "kernel_1c_us_std": float(std_1c),
            "total_us_std": float(std_total),
            "num_samples": len(kernel_timings_list),
        }

    results = {
        "jax_times": jax_times.tolist(),
        "cuda_times": cuda_times.tolist(),
        "jax_avg_time_us": float(jax_avg_time_us),
        "cuda_avg_time_us": float(cuda_avg_time_us),
        "speedup": float(speedup),
        "depth_mae": float(depth_mae) if depth_mae is not None else None,
        "alpha_mae": float(alpha_mae) if alpha_mae is not None else None,
        "depth_max_diff": float(depth_max_diff) if depth_max_diff is not None else None,
        "alpha_max_diff": float(alpha_max_diff) if alpha_max_diff is not None else None,
        "depth_relative_error": float(depth_relative_error)
        if depth_relative_error is not None
        else None,
        "alpha_relative_error": float(alpha_relative_error)
        if alpha_relative_error is not None
        else None,
        "depth_ok": bool(depth_ok),
        "alpha_ok": bool(alpha_ok),
        "kernel_id": kernel_id,
        "kernel_name": kernel_name,
        "num_fmbs": opt_results["num_fmbs"],
        "width": width,
        "height": height,
        "num_views": len(cameras_list),
        "jax_results": jax_results_list,  # All views
        "cuda_results": cuda_results_list,  # All views
        "kernel_timings": kernel_timings_dict,
        "cuda_scene_data": {
            "fmbs": [
                {
                    "pose": {
                        "tran": (
                            float(fmb.pose.tran.x),
                            float(fmb.pose.tran.y),
                            float(fmb.pose.tran.z),
                        ),
                        "rot": tuple(fmb.pose.rot.quat),
                    },
                    "extent": tuple(fmb.extent),
                }
                for fmb in fmbs
            ],
            "log_weights": [float(w) for w in final_weight_logs],
        },
    }

    return results


def create_comparison_plots(
    jax_results,
    cuda_results,
    alpha_results_final,
    opt_results,
    kernel_id,
    kernel_name,
    use_wandb=False,
):
    """Create two comparison plots similar to notebook:
    1. Depth plot: 20 rows × 4 columns (GMB filtered, FMB-JAX filtered, GT placeholder, Diff)
    2. Confidence plot: 20 rows × 3 columns (GMB conf, FMB-JAX alpha, Diff)
    """
    width = opt_results["width"]
    height = opt_results["height"]
    num_views = len(jax_results)

    cache_dir = opt_results.get("project_root", Path.cwd()) / "scripts" / "data" / "benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data like notebook: filter depths where alpha/confidence < 0.5
    # For JAX: est_depth[est_alpha < 0.5] = np.nan (on flat array, then reshape)
    # For CUDA: depth_image = jnp.where(gmb_confidences < 0.5, jnp.nan, gmb_depths) (on reshaped image)
    jax_depths_filtered = []
    cuda_depths_filtered = []
    jax_alphas = []
    cuda_alphas = []
    cuda_depths_raw = []
    jax_depths_raw = []

    for view_idx in range(num_views):
        jax_depth, jax_alpha = jax_results[view_idx]
        cuda_depth, cuda_alpha = cuda_results[view_idx]

        # NOTE: jax_depth is already FILTERED (has NaN where alpha < 0.5) from benchmark loop
        # This matches notebook Cell 20: est_depth[est_alpha < 0.5] = np.nan before reshape
        jax_depth_np = np.array(jax_depth)  # Already filtered and reshaped
        jax_alpha_np = np.array(jax_alpha)  # Already reshaped
        jax_depths_filtered.append(jax_depth_np)  # Already filtered
        jax_alphas.append(jax_alpha_np)

        # For CUDA: reshape first, then filter (like notebook: depth_image = jnp.where(gmb_confidences < 0.5, jnp.nan, gmb_depths))
        # Use JAX operations like notebook, then convert to NumPy
        cuda_depth_jax = (
            cuda_depth.reshape(height, width) if len(cuda_depth.shape) == 1 else cuda_depth
        )
        cuda_alpha_jax = (
            cuda_alpha.reshape(height, width) if len(cuda_alpha.shape) == 1 else cuda_alpha
        )
        cuda_depth_filtered_jax = jnp.where(cuda_alpha_jax < 0.5, jnp.nan, cuda_depth_jax)
        cuda_depth_filtered_img = np.array(cuda_depth_filtered_jax)
        cuda_depths_filtered.append(cuda_depth_filtered_img)

        # Keep FILTERED depths for difference calculation (FILTERED CUDA vs FILTERED JAX)
        # NOTE: Both should be filtered to compare at the same processing stage
        # This gives a fair comparison of implementation differences, not filtering artifacts
        cuda_depths_raw.append(cuda_depth_filtered_jax)  # Store filtered CUDA depth for difference
        jax_depths_raw.append(jnp.array(jax_depth_np))  # Store filtered JAX depth for difference
        cuda_alphas.append(np.array(cuda_alpha_jax))

    # Get depth range for consistent colormap (matching notebook: uses ref_depths for vmin/vmax)
    # NOTE: Notebook Cell 20 uses: vmin = np.nanmin(np.array(ref_depths)), vmax = np.nanmax(np.array(ref_depths))
    if "ref_depths" in opt_results and len(opt_results["ref_depths"]) > 0:
        ref_depths_array = np.array(opt_results["ref_depths"])
        vmin = np.nanmin(ref_depths_array)
        vmax = np.nanmax(ref_depths_array)
    else:
        # Fallback: use raw depths if ref_depths not available
        all_raw_depths = np.concatenate([d.flatten() for d in jax_depths_raw + cuda_depths_raw])
        valid_raw = all_raw_depths[~np.isnan(all_raw_depths)]
        if len(valid_raw) > 0:
            vmin = np.min(valid_raw)
            vmax = np.max(valid_raw)
        else:
            vmin, vmax = 0.0, 1.0

    # ===== PLOT 1: DEPTH COMPARISON (20 rows × 4 columns) =====
    fig, axes = plt.subplots(num_views, 4, figsize=(16, 5 * num_views))
    if num_views == 1:
        axes = axes.reshape(1, -1)

    for view_idx in range(num_views):
        ax0, ax1, ax2, ax3 = axes[view_idx]

        # GenMetaBalls depth (filtered) - like notebook: depth_image = jnp.where(gmb_confidences < 0.5, jnp.nan, gmb_depths)
        im0 = ax0.imshow(cuda_depths_filtered[view_idx], vmin=vmin, vmax=vmax, cmap="viridis")
        ax0.set_title(f"GenMetaBalls (View {view_idx})", fontsize=10, fontweight="bold")
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0)

        # FMB-JAX depth (filtered) - like notebook: alpha_results_depth[VIEW_IDX] (already filtered)
        im1 = ax1.imshow(jax_depths_filtered[view_idx], vmin=vmin, vmax=vmax, cmap="viridis")
        ax1.set_title(f"FMB-JAX (View {view_idx})", fontsize=10, fontweight="bold")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1)

        # GT Depth (from reference renders) - like notebook uses ref_depths
        if "ref_depths" in opt_results and view_idx < len(opt_results["ref_depths"]):
            ref_depth = np.array(opt_results["ref_depths"][view_idx])
            im2 = ax2.imshow(ref_depth, vmin=vmin, vmax=vmax, cmap="viridis")
            ax2.set_title(f"GT Depth (View {view_idx})", fontsize=10, fontweight="bold")
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2)
        else:
            ax2.axis("off")
            ax2.text(
                0.5, 0.5, "N/A", ha="center", va="center", fontsize=12, transform=ax2.transAxes
            )
            ax2.set_title(f"GT Depth (View {view_idx})", fontsize=10, fontweight="bold")

        # Difference: FILTERED CUDA vs FILTERED JAX (both at same processing stage)
        # NOTE: Both depths are filtered (NaN where confidence/alpha < 0.5)
        # This gives a fair comparison of implementation differences, not filtering artifacts
        # Use JAX arrays for difference calculation, then convert to NumPy for plotting
        diff_jax = (
            cuda_depths_raw[view_idx] - jax_depths_raw[view_idx]
        )  # FILTERED CUDA - FILTERED JAX
        diff = np.array(diff_jax)
        im3 = ax3.imshow(diff, cmap="RdBu_r")
        ax3.set_title(f"Diff (GMB - FMB-JAX) (View {view_idx})", fontsize=10, fontweight="bold")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    depth_plot_file = (
        cache_dir
        / f"depth_comparison_fmbs{opt_results['num_fmbs']}_size{width}x{height}_kernel{kernel_id}.png"
    )
    plt.savefig(depth_plot_file, dpi=150, bbox_inches="tight")

    # Log plot to wandb if enabled
    if use_wandb:
        wandb.log({"plots/depth_comparison": wandb.Image(str(depth_plot_file))})

    plt.close()
    print(f"✓ Depth comparison plot saved to {depth_plot_file}")

    # ===== PLOT 2: CONFIDENCE/ALPHA COMPARISON (20 rows × 3 columns) =====
    fig, axes = plt.subplots(num_views, 3, figsize=(18, 5 * num_views))
    if num_views == 1:
        axes = axes.reshape(1, -1)

    for view_idx in range(num_views):
        ax0, ax1, ax2 = axes[view_idx]

        # GenMetaBalls confidence - like notebook: gmb_confidences[VIEW_IDX]
        # NOTE: In notebook, gmb_confidences is saved as: confidence = jnp.copy(img_view.confidence.as_jax())
        # and then used directly: im0 = ax0.imshow(gmb_confidences[VIEW_IDX], vmin=0, vmax=1)
        # The notebook passes JAX arrays directly to imshow - matplotlib handles the conversion
        # Notebook does NOT specify cmap for gmb_confidences (uses default matplotlib colormap)
        # Use confidence directly from cuda_results, matching notebook's gmb_confidences[VIEW_IDX]
        cuda_confidence = cuda_results[view_idx][
            1
        ]  # Get confidence directly from cuda_results (JAX array)
        # Reshape if needed (notebook uses JAX array directly, matplotlib auto-converts)
        if len(cuda_confidence.shape) == 1:
            cuda_confidence_img = cuda_confidence.reshape(height, width)
        else:
            cuda_confidence_img = cuda_confidence
        # Pass JAX array directly to imshow like notebook (matplotlib will convert internally)
        im0 = ax0.imshow(cuda_confidence_img, vmin=0, vmax=1)  # No cmap, matching notebook
        ax0.set_title(f"GenMetaBalls Confidence (View {view_idx})", fontsize=10, fontweight="bold")
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0)

        # FMB-JAX alpha - like notebook: alpha_results_final[VIEW_IDX]
        # Use alpha_results_final (created before benchmarking, matching notebook Cell 20)
        # NOTE: Notebook uses cmap="viridis" for alpha_results_final
        # alpha_results_final is already reshaped to (height, width) in Cell 20
        # Pass JAX array directly to imshow like notebook (matplotlib will convert internally)
        jax_alpha_final = alpha_results_final[view_idx]  # Already (height, width) shape, JAX array
        im1 = ax1.imshow(jax_alpha_final, vmin=0, vmax=1, cmap="viridis")
        ax1.set_title(f"FMB-JAX Alpha (View {view_idx})", fontsize=10, fontweight="bold")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1)

        # Difference - like notebook: diff = gmb_confidences[VIEW_IDX] - alpha_results_final[VIEW_IDX]
        # NOTE: Notebook computes difference with JAX arrays directly
        # Both arrays should already be in (height, width) shape from above
        # Compute difference with JAX arrays (preserves precision) and pass directly to imshow
        # IMPORTANT: Ensure we're comparing CUDA confidence (from CUDA rendering) with JAX alpha (from JAX rendering)
        # They should be different implementations, so there should be small precision differences
        conf_diff = cuda_confidence_img - jax_alpha_final  # JAX array subtraction

        # Convert to NumPy to compute range for colormap
        conf_diff_np = np.array(conf_diff)
        # Notebook doesn't specify vmin/vmax for diff, so let matplotlib auto-scale
        # But the differences are small (around ±0.01), so we need to see them
        # Use symmetric range based on max absolute difference
        max_abs_diff = np.max(np.abs(conf_diff_np))
        if max_abs_diff > 0:
            # Use symmetric range around the actual data range
            vmin_diff = -max_abs_diff * 1.1  # Add 10% padding
            vmax_diff = max_abs_diff * 1.1
        else:
            vmin_diff, vmax_diff = -0.01, 0.01  # Fallback range

        # Pass JAX array directly to imshow like notebook (matplotlib will convert internally)
        # NOTE: Notebook uses cmap="RdBu_r" without vmin/vmax, but we need to set them to see small differences
        im2 = ax2.imshow(conf_diff, cmap="RdBu_r", vmin=vmin_diff, vmax=vmax_diff)
        ax2.set_title(f"Diff (View {view_idx})", fontsize=10, fontweight="bold")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    conf_plot_file = (
        cache_dir
        / f"confidence_comparison_fmbs{opt_results['num_fmbs']}_size{width}x{height}_kernel{kernel_id}.png"
    )
    plt.savefig(conf_plot_file, dpi=150, bbox_inches="tight")

    # Log plot to wandb if enabled
    if use_wandb:
        wandb.log({"plots/confidence_comparison": wandb.Image(str(conf_plot_file))})

    plt.close()
    print(f"✓ Confidence comparison plot saved to {conf_plot_file}")


def save_benchmark_results(project_root, results, num_fmbs, width, height, kernel_id):
    """Save benchmark results to file."""
    cache_dir = project_root / "scripts" / "data" / "benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    kernel_suffix = f"kernel{kernel_id}"
    results_file = cache_dir / f"results_fmbs{num_fmbs}_size{width}x{height}_{kernel_suffix}.json"

    print(f"\nSaving benchmark results to {results_file}...")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CUDA vs JAX FMB rendering")
    parser.add_argument("--num-fmbs", type=int, default=40, help="Number of FMBs (default: 40)")
    parser.add_argument("--width", type=int, default=64, help="Image width (default: 64)")
    parser.add_argument("--height", type=int, default=64, help="Image height (default: 64)")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--kernel-id",
        type=int,
        default=0,
        help="CUDA kernel ID to use (0=original slow working, 1=FMB-parallelized, default: 0)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        default=[4, 4],
        metavar=("X", "Y"),
        help="Grid size for CUDA kernel (default: 4 4)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=("X", "Y"),
        help="Block size for CUDA kernel (default: 16 16)",
    )
    parser.add_argument(
        "--force-rerun", action="store_true", help="Force re-run optimization even if cache exists"
    )
    parser.add_argument("--save-plot", action="store_true", help="Save comparison plot as PNG")
    parser.add_argument(
        "--num-fmb-chunks",
        type=int,
        default=4,
        help="Number of FMB chunks for kernel_id=1 (default: 4)",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable wandb logging for benchmark results"
    )

    args = parser.parse_args()

    # Initialize wandb if requested
    use_wandb = args.use_wandb
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not installed. Install it with: pip install wandb")
            print("Continuing without wandb logging...")
            use_wandb = False
        else:
            wandb.init(
                entity="metaballers",
                project="genmetaballs-benchmark",
                config={
                    "num_fmbs": args.num_fmbs,
                    "width": args.width,
                    "height": args.height,
                    "kernel_id": args.kernel_id,
                    "grid_size": "x".join(map(str, args.grid_size)),
                    "block_size": "x".join(map(str, args.block_size)),
                    "warmup": args.warmup,
                    "num_fmb_chunks": args.num_fmb_chunks,
                    "num_views": num_views,
                },
            )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    mesh_file = PROJECT_ROOT / "data/cow/cow.obj"

    # Load or run optimization
    opt_results = load_or_run_optimization(
        mesh_file,
        args.num_fmbs,
        args.width,
        args.height,
        PROJECT_ROOT,
        force_rerun=args.force_rerun,
        use_wandb=use_wandb,
    )

    # Add project_root to opt_results for plot saving
    opt_results["project_root"] = PROJECT_ROOT

    # print the grid size and block size
    print(f"Grid size: {args.grid_size}")
    print(f"Block size: {args.block_size}")
    if args.kernel_id == 1:
        print(f"Number of FMB chunks: {args.num_fmb_chunks}")

    # Run benchmark
    results = benchmark_comparison(
        opt_results,
        kernel_id=args.kernel_id,
        warmup=args.warmup,
        save_plot=args.save_plot,
        grid_size=args.grid_size,
        block_size=args.block_size,
        num_fmb_chunks=args.num_fmb_chunks,
        use_wandb=use_wandb,
    )

    # Finish wandb run if enabled
    if use_wandb:
        wandb.finish()

    # NOTE: JSON saving disabled - only plots are saved
    # save_benchmark_results(PROJECT_ROOT, results, args.num_fmbs, args.width, args.height, args.kernel_id)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("Performance:")
    print(f"  FMB-JAX:      {results['jax_avg_time_us']:.2f} μs")
    print(
        f"  GenMetaBalls:  {results['cuda_avg_time_us']:.2f} μs (kernel_id={args.kernel_id}, {results['kernel_name']})"
    )
    print(
        f"  Speedup:      {results['speedup']:.2f}x {'(CUDA faster)' if results['speedup'] > 1 else '(JAX faster)'}"
    )
    print()
    print("Accuracy:")
    if results["depth_mae"] is not None:
        print(
            f"  Depth MAE:              {results['depth_mae']:.6f} ({'✓' if results['depth_ok'] else '✗'})"
        )
        print(
            f"  Depth Relative Error:   {results['depth_relative_error'] * 100:.4f}% ({'✓' if results['depth_ok'] else '✗'})"
        )
    print(
        f"  Alpha MAE:              {results['alpha_mae']:.6f} ({'✓' if results['alpha_ok'] else '✗'})"
    )
    print(
        f"  Alpha Relative Error:   {results['alpha_relative_error'] * 100:.4f}% ({'✓' if results['alpha_ok'] else '✗'})"
    )
    print("=" * 80)

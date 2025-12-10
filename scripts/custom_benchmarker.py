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


def benchmark_single_method(
    opt_results,
    method_name,
    kernel_id=None,
    num_fmb_chunks=None,
    warmup=10,
    grid_size=None,
    block_size=None,
):
    """Benchmark a single method (JAX or CUDA with specific kernel/chunk settings)."""
    # Set defaults for grid_size and block_size
    if grid_size is None:
        grid_size = dim3(8, 8)
    else:
        grid_size = dim3(grid_size[0], grid_size[1])

    if block_size is None:
        block_size = dim3(8, 8)
    else:
        block_size = dim3(block_size[0], block_size[1])

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

    print(f"\nBenchmarking {method_name}...")

    if method_name == "FMB-JAX":
        # JAX benchmarking
        render_jit = jax.jit(fm_render.render_func_rays)

        # Warmup
        for view_idx in range(len(cameras_list)):
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

        # Benchmark
        ITER_MULT = 1000
        jax_times = []

        for view_idx in tqdm(range(len(cameras_list)), desc=f"Benchmarking {method_name}"):
            camera_rays = cameras_list[view_idx]

            for _ in range(ITER_MULT):
                beta2_div_shape_scale = beta2 / shape_scale

                start_time = time.time_ns()
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
                elapsed = (end_time - start_time) / 1e3  # microseconds
                jax_times.append(elapsed)

        jax_times = np.array(jax_times)
        total_time_us = jax_times.sum()
        avg_time_us = total_time_us / len(cameras_list) / ITER_MULT

        return avg_time_us

    else:
        # CUDA benchmarking
        # Create CUDA scene from optimized parameters
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

        # Preallocate CUDA image object
        cuda_image = make_image(height, width, device="gpu")

        # Create temp_buffer for kernel_id=1
        cuda_temp_buffer = None
        if kernel_id == 1:
            cuda_temp_buffer = make_temp_buffer(height, width, num_fmb_chunks, device="gpu")

        # Warmup
        for view_idx in range(len(cameras_list)):
            cuda_extr = Pose.from_components(
                rot=Rotation.from_quat(*rand_quats[view_idx]).inv(), tran=Vec3D(*trans[view_idx])
            )
            for _ in range(warmup):
                render_kwargs = {
                    "fmbs": cuda_scene,
                    "blender": cuda_blender,
                    "confidence": cuda_confidence,
                    "intr": cuda_intr,
                    "extr": cuda_extr,
                    "img": cuda_image,
                    "grid_size": grid_size,
                    "block_size": block_size,
                    "kernel_id": kernel_id,
                    "temp_buffer": cuda_temp_buffer,
                }
                if num_fmb_chunks is not None:
                    render_kwargs["num_fmb_chunks"] = num_fmb_chunks
                render_fmbs(**render_kwargs)
                _ = cuda_image.as_view().depth.as_jax().block_until_ready()

        # Benchmark
        ITER_MULT = 1000
        cuda_times = []

        for view_idx in tqdm(range(len(cameras_list)), desc=f"Benchmarking {method_name}"):
            cuda_extr = Pose.from_components(
                rot=Rotation.from_quat(*rand_quats[view_idx]).inv(), tran=Vec3D(*trans[view_idx])
            )

            for _ in range(ITER_MULT):
                start_time = time.time_ns()
                render_kwargs = {
                    "fmbs": cuda_scene,
                    "blender": cuda_blender,
                    "confidence": cuda_confidence,
                    "intr": cuda_intr,
                    "extr": cuda_extr,
                    "img": cuda_image,
                    "grid_size": grid_size,
                    "block_size": block_size,
                    "kernel_id": kernel_id,
                    "block": True,
                    "temp_buffer": cuda_temp_buffer,
                    "return_timings": False,
                }
                if num_fmb_chunks is not None:
                    render_kwargs["num_fmb_chunks"] = num_fmb_chunks
                render_fmbs(**render_kwargs)
                end_time = time.time_ns()
                elapsed = (end_time - start_time) / 1e3  # microseconds
                cuda_times.append(elapsed)

        cuda_times = np.array(cuda_times)
        total_time_us = cuda_times.sum()
        avg_time_us = total_time_us / len(cameras_list) / ITER_MULT

        return avg_time_us


def run_full_benchmark():
    """Run the full benchmark experiment across all FMB sizes and methods."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    mesh_file = PROJECT_ROOT / "data/cow/cow.obj"
    
    # Fixed parameters
    width, height = 64, 64
    grid_size = (8, 8)
    block_size = (8, 8)
    warmup = 10
    
    # FMB sizes to test
    fmb_sizes = [40, 100, 200, 300, 400, 500, 600, 700, 800]
    
    # Methods to test
    methods = [
        ("FMB-JAX", None, None),
        ("CUDA Kernel 0", 0, None),
        ("CUDA Kernel 1 (chunk=4)", 1, 4),
        ("CUDA Kernel 1 (chunk=8)", 1, 8),
        ("CUDA Kernel 1 (chunk=16)", 1, 16),
        ("CUDA Kernel 1 (chunk=32)", 1, 32),
    ]
    
    # Results storage
    results = {
        "config": {
            "width": width,
            "height": height,
            "grid_size": grid_size,
            "block_size": block_size,
            "warmup": warmup,
            "fmb_sizes": fmb_sizes,
        },
        "data": {}
    }
    
    # Initialize data structure
    for method_name, _, _ in methods:
        results["data"][method_name] = []
    
    # Results file path
    results_file = PROJECT_ROOT / "scripts" / "data" / "benchmark_cache" / "full_benchmark_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FULL BENCHMARK EXPERIMENT")
    print("=" * 80)
    print(f"FMB sizes: {fmb_sizes}")
    print(f"Image size: {width}x{height}")
    print(f"Grid size: {grid_size}")
    print(f"Block size: {block_size}")
    print(f"Methods: {len(methods)}")
    print("=" * 80)
    
    for num_fmbs in fmb_sizes:
        print(f"\n{'='*60}")
        print(f"TESTING FMB SIZE: {num_fmbs}")
        print(f"{'='*60}")
        
        # Load or run optimization for this FMB size
        opt_results = load_or_run_optimization(
            mesh_file, num_fmbs, width, height, PROJECT_ROOT, force_rerun=False
        )
        
        # Test each method
        for method_name, kernel_id, num_fmb_chunks in methods:
            print(f"\nTesting {method_name}...")
            
            avg_time_us = benchmark_single_method(
                opt_results,
                method_name,
                kernel_id=kernel_id,
                num_fmb_chunks=num_fmb_chunks,
                warmup=warmup,
                grid_size=grid_size,
                block_size=block_size,
            )
            
            # Store result
            results["data"][method_name].append({
                "num_fmbs": num_fmbs,
                "avg_time_us": float(avg_time_us)
            })
            
            print(f"  Average time: {avg_time_us:.2f} μs")
        
        # Save results after each FMB size completion
        print(f"\nSaving results to {results_file}...")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print("✓ Results saved")
    
    # Create final plot
    create_performance_plot(results, PROJECT_ROOT)
    
    return results


def create_performance_plot(results, project_root):
    """Create a line plot showing performance across all methods and FMB sizes."""
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    fmb_sizes = results["config"]["fmb_sizes"]
    
    # Define colors and line styles for each method
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    for i, (method_name, method_data) in enumerate(results["data"].items()):
        # Extract times for this method
        times = [point["avg_time_us"] for point in method_data]
        
        plt.plot(
            fmb_sizes, 
            times, 
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            marker='o',
            linewidth=2,
            markersize=6,
            label=method_name
        )
    
    plt.xlabel('Number of Metaballs', fontsize=12)
    plt.ylabel('Runtime (microseconds)', fontsize=12)
    plt.title('Performance Comparison: FMB Rendering Methods', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    # Add configuration info as text
    config_text = f"Image: {results['config']['width']}x{results['config']['height']}, " \
                  f"Grid: {results['config']['grid_size']}, " \
                  f"Block: {results['config']['block_size']}"
    plt.figtext(0.02, 0.02, config_text, fontsize=8, style='italic')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = project_root / "scripts" / "data" / "benchmark_cache" / "performance_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Performance plot saved to {plot_file}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for method_name, method_data in results["data"].items():
        times = [point["avg_time_us"] for point in method_data]
        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)
        
        print(f"{method_name}:")
        print(f"  Min time: {min_time:.2f} μs")
        print(f"  Max time: {max_time:.2f} μs")
        print(f"  Avg time: {avg_time:.2f} μs")
        print()
    
    # Find fastest method for each FMB size
    print("Fastest method for each FMB size:")
    for i, num_fmbs in enumerate(fmb_sizes):
        method_times = {}
        for method_name, method_data in results["data"].items():
            method_times[method_name] = method_data[i]["avg_time_us"]
        
        fastest_method = min(method_times, key=method_times.get)
        fastest_time = method_times[fastest_method]
        
        print(f"  {num_fmbs} FMBs: {fastest_method} ({fastest_time:.2f} μs)")
    
    print("=" * 80)


if __name__ == "__main__":
    # Run the full benchmark experiment
    results = run_full_benchmark()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("Results saved to: scripts/data/benchmark_cache/full_benchmark_results.json")
    print("Plot saved to: scripts/data/benchmark_cache/performance_comparison.png")
    print("=" * 80)

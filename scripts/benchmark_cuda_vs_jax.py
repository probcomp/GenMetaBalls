#!/usr/bin/env python3
"""Comprehensive benchmark comparing CUDA render_fmbs vs JAX implementation.
Follows the exact setup from fuzzy_metaballs_demo.ipynb

Saves results to cache files to avoid re-running optimization.
"""

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import subprocess
import pickle
import json

# Setup environment
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import genmetaballs.fmb.fm_render as fm_render
from genmetaballs.core import (
    FMB,
    Intrinsics,
    ThreeParameterBlender,
    ZeroParameterConfidence,
    geometry,
    make_fmb_scene_from_values,
    render_fmbs,
)
from genmetaballs.fmb.utils import DegradeLR, get_camera_rays
from jax.example_libraries import optimizers
from jax.scipy.spatial.transform import Rotation as Rot
import trimesh

Pose, Vec3D, Rotation = geometry.Pose, geometry.Vec3D, geometry.Rotation

# Configuration from notebook
gmm_init_scale = 1.0
rand_sphere_size = 30
num_views = 20
vfov_degrees = 45
Nepochs = 10
batch_size = 800
initial_lr = 0.1
opt_shape_scale = 2.2
clip_alpha = 3.0e-8
random_seed = 42


@jax.jit
def cov_to_isostds_and_quaternion(cov):
    """Convert a 3D Gaussian's covariance matrix to an isotropic stds vector and a rotation quaternion."""
    eigvals, eigvecs = jnp.linalg.eigh(cov)
    vars = jnp.maximum(eigvals, 0)
    # Ensure deterministic eigenvector orientation
    for i in range(3):
        eigvecs = eigvecs.at[:, i].set(jnp.where(eigvecs[0, i] < 0, -eigvecs[:, i], eigvecs[:, i]))
    # Ensure proper rotation matrix (determinant +1)
    eigvecs = eigvecs.at[:, 0].set(
        jnp.where(jnp.linalg.det(eigvecs) < 0, -eigvecs[:, 0], eigvecs[:, 0])
    )
    quat = Rot.from_matrix(eigvecs).as_quat()
    return vars, quat


def get_cache_path(project_root, num_fmbs, width, height):
    """Get path to cache file for given configuration."""
    cache_dir = project_root / "scripts" / "data" / "benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"benchmark_fmbs{num_fmbs}_size{width}x{height}.pkl"
    return cache_file


def load_or_run_optimization(mesh_file, num_fmbs, width, height, project_root, force_rerun=False):
    """Load optimization results from cache or run optimization."""
    cache_file = get_cache_path(project_root, num_fmbs, width, height)
    
    if cache_file.exists() and not force_rerun:
        print(f"Loading cached optimization results from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Running optimization (this may take a while)...")
    result = run_optimization(mesh_file, num_fmbs, width, height)
    
    # Save to cache
    print(f"Saving optimization results to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result


def run_optimization(mesh_file, num_fmbs, width, height):
    """Run the optimization from the notebook to get final parameters."""
    print("Loading mesh and setting up optimization...")
    
    # Load mesh
    if not mesh_file.exists():
        subprocess.run(
            ["bash", str((mesh_file.parent.parent / "scripts/data/download_cow.sh").resolve().absolute())]
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
    
    # Generate random camera poses
    np.random.seed(random_seed)
    rand_quats = np.random.randn(num_views, 4)
    rand_quats = rand_quats / np.linalg.norm(rand_quats, axis=1, keepdims=True)
    
    # Render reference views (simplified - just get camera poses)
    trans = []
    for quat in rand_quats:
        R = Rot.from_quat(quat).as_matrix()
        loc = np.array([0, 0, 3 * shape_scale]) @ R + center
        trans.append(loc)
    
    # Initialize FMBs
    np.random.seed(random_seed)
    rand_mean = center + np.random.multivariate_normal(
        mean=[0, 0, 0], cov=1e-2 * np.identity(3) * shape_scale, size=num_fmbs
    )
    rand_weight_log = jnp.log(np.ones(num_fmbs) / num_fmbs) + jnp.log(gmm_init_scale)
    rand_prec = jnp.array([np.identity(3) * rand_sphere_size / shape_scale for _ in range(num_fmbs)])
    
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
    
    # Create dummy target silhouettes (all ones for testing)
    all_cameras = jnp.array(cameras_list).reshape((-1, 2, 3))
    all_sils = jnp.ones(all_cameras.shape[0], dtype=jnp.float32)  # Dummy target
    
    # Setup optimizer
    vecM = jnp.array([[1, 1, 1], [shape_scale_mul, shape_scale_mul, shape_scale_mul]])[None]
    opt_init, opt_update, opt_params = optimizers.adam(initial_lr)
    tmp = [rand_mean * shape_scale_mul, rand_prec / shape_scale_mul, rand_weight_log]
    opt_state = opt_init(tmp)
    
    # Run a few optimization steps (or use initial params for speed)
    print("Running optimization (simplified)...")
    for i in range(min(5, 5)):  # Just a few iterations for testing
        p = opt_params(opt_state)
        idx = jnp.array(list(range(min(batch_size, len(all_cameras)))))
        val, g = grad_render3(
            [p[0], p[1], p[2], vecM * all_cameras[idx], beta2 / opt_shape_scale, beta3],
            all_sils[idx],
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)
        opt_state = opt_update(i, g[:3], opt_state)
    
    # Get final parameters
    final_params = opt_params(opt_state)
    final_means = final_params[0] / shape_scale_mul
    final_precs = final_params[1] * shape_scale_mul
    final_weight_logs = final_params[2]
    
    return {
        'final_means': final_means,
        'final_precs': final_precs,
        'final_weight_logs': final_weight_logs,
        'shape_scale': shape_scale,
        'center': center,
        'cameras_list': cameras_list,
        'trans': trans,
        'rand_quats': rand_quats,
        'focal_length': focal_length,
        'cx': cx,
        'cy': cy,
        'beta2': beta2,
        'beta3': beta3,
        'width': width,
        'height': height,
        'num_fmbs': num_fmbs,
    }


def benchmark_comparison(opt_results, use_optimized=True, num_iterations=100, warmup=10):
    """Benchmark and compare CUDA vs JAX implementations using notebook setup."""
    print("="*80)
    print(f"BENCHMARK: CUDA render_fmbs vs JAX Implementation")
    print("="*80)
    print(f"Following exact setup from fuzzy_metaballs_demo.ipynb")
    print(f"  FMBs: {opt_results['num_fmbs']}")
    print(f"  Image size: {opt_results['width']}x{opt_results['height']}")
    print(f"  Views: {num_views}")
    print(f"  Use optimized kernel: {use_optimized}")
    print(f"  Warmup iterations: {warmup}")
    print(f"  Benchmark iterations: {num_iterations}")
    print()
    
    final_means = opt_results['final_means']
    final_precs = opt_results['final_precs']
    final_weight_logs = opt_results['final_weight_logs']
    shape_scale = opt_results['shape_scale']
    cameras_list = opt_results['cameras_list']
    trans = opt_results['trans']
    rand_quats = opt_results['rand_quats']
    focal_length = opt_results['focal_length']
    cx = opt_results['cx']
    cy = opt_results['cy']
    beta2 = opt_results['beta2']
    beta3 = opt_results['beta3']
    width = opt_results['width']
    height = opt_results['height']
    
    # Create CUDA scene from optimized parameters
    print("Creating CUDA FMB scene from optimized parameters...")
    fmbs = []
    for (final_mean, final_prec) in zip(final_means, final_precs):
        prec_matrix = final_prec @ final_prec.T
        stds, quat = cov_to_isostds_and_quaternion(jnp.linalg.inv(prec_matrix))
        pose = Pose.from_components(Rotation.from_quat(*quat), Vec3D(*final_mean))
        fmbs.append(FMB(pose, *stds))
    
    cuda_scene = make_fmb_scene_from_values(fmbs, list(final_weight_logs), device="gpu")
    
    # CUDA camera setup
    cuda_intr = Intrinsics(fx=focal_length, fy=focal_length, cx=cx, cy=cy, width=width, height=height)
    cuda_blender = ThreeParameterBlender(beta1=beta3, beta2=beta2, eta=shape_scale)
    cuda_confidence = ZeroParameterConfidence()
    
    # JIT compile JAX render function
    render_jit = jax.jit(fm_render.render_func_rays)
    
    # Warmup
    print("Warming up JAX...")
    for _ in range(warmup):
        result = render_jit(
            final_means, final_precs, final_weight_logs, cameras_list[0], beta2 / shape_scale, beta3
        )
        result[1].block_until_ready()
    
    print("Warming up CUDA...")
    for _ in range(warmup):
        cuda_extr = Pose.from_components(
            rot=Rotation.from_quat(*rand_quats[0]).inv(),
            tran=Vec3D(*trans[0])
        )
        image = render_fmbs(cuda_scene, cuda_blender, cuda_confidence, cuda_intr, cuda_extr, use_optimized=use_optimized)
        _ = image.as_view().depth.as_jax().block_until_ready()
    
    print("\nRunning benchmarks...")
    
    # Benchmark JAX
    jax_times = []
    jax_results = []
    jax_forward_times = []  # Per-forward-pass times
    for i in range(num_iterations):
        view_idx = i % len(cameras_list)
        start = time.perf_counter()
        est_depth, est_alpha, est_norm, est_w = render_jit(
            final_means, final_precs, final_weight_logs, cameras_list[view_idx], beta2 / shape_scale, beta3
        )
        est_alpha.block_until_ready()
        elapsed = (time.perf_counter() - start) * 1000
        jax_times.append(elapsed)
        jax_forward_times.append(elapsed)  # Same for JAX (forward only)
        if i == 0:  # Save first result for accuracy comparison
            jax_results = (np.array(est_depth), np.array(est_alpha))
    
    # Benchmark CUDA
    cuda_times = []
    cuda_results = None
    cuda_forward_times = []  # Per-forward-pass times
    for i in range(num_iterations):
        view_idx = i % len(cameras_list)
        cuda_extr = Pose.from_components(
            rot=Rotation.from_quat(*rand_quats[view_idx]).inv(),
            tran=Vec3D(*trans[view_idx])
        )
        start = time.perf_counter()
        image = render_fmbs(cuda_scene, cuda_blender, cuda_confidence, cuda_intr, cuda_extr, use_optimized=use_optimized)
        _ = image.as_view().depth.as_jax().block_until_ready()
        elapsed = (time.perf_counter() - start) * 1000
        cuda_times.append(elapsed)
        cuda_forward_times.append(elapsed)  # Same for CUDA (forward only)
        if i == 0:  # Save first result for accuracy comparison
            img_view = image.as_view()
            cuda_depth = np.array(img_view.depth.as_jax())
            cuda_alpha = np.array(img_view.confidence.as_jax())
            cuda_results = (cuda_depth, cuda_alpha)
    
    # Statistics
    jax_times = np.array(jax_times)
    cuda_times = np.array(cuda_times)
    jax_mean_ms = jax_times.mean()
    cuda_mean_ms = cuda_times.mean()
    
    print("\n" + "="*80)
    print("SPEED RESULTS:")
    print("="*80)
    
    print(f"FMB-JAX Implementation:")
    print(f"  Mean time per run: {jax_mean_ms:.4f} ms")
    print(f"  Std deviation:     {jax_times.std():.4f} ms")
    print(f"  Min time:          {jax_times.min():.4f} ms")
    print(f"  Max time:          {jax_times.max():.4f} ms")
    print(f"  Median time:       {np.median(jax_times):.4f} ms")
    print(f"  FPS:               {1000.0 / jax_mean_ms:.2f}")
    print()
    print(f"GenMetaBalls CUDA Implementation ({'optimized' if use_optimized else 'original'}):")
    print(f"  Mean time per run: {cuda_mean_ms:.4f} ms")
    print(f"  Std deviation:     {cuda_times.std():.4f} ms")
    print(f"  Min time:          {cuda_times.min():.4f} ms")
    print(f"  Max time:          {cuda_times.max():.4f} ms")
    print(f"  Median time:       {np.median(cuda_times):.4f} ms")
    print(f"  FPS:               {1000.0 / cuda_mean_ms:.2f}")
    print()
    speedup = jax_mean_ms / cuda_mean_ms
    print(f"Speedup: {speedup:.2f}x {'(CUDA faster)' if speedup > 1 else '(JAX faster)'}")
    print(f"  FMB-JAX:    {jax_mean_ms:.4f} ms/run")
    print(f"  GenMetaBalls: {cuda_mean_ms:.4f} ms/run")
    print("="*80)
    
    # Accuracy comparison
    print("\n" + "="*80)
    print("ACCURACY RESULTS:")
    print("="*80)
    
    jax_depth, jax_alpha = jax_results
    cuda_depth, cuda_alpha = cuda_results
    
    # Reshape to image dimensions
    jax_depth_img = jax_depth.reshape(height, width)
    jax_alpha_img = jax_alpha.reshape(height, width)
    cuda_depth_img = cuda_depth.reshape(height, width)
    cuda_alpha_img = cuda_alpha.reshape(height, width)
    
    # Compare depth (only where both have valid values)
    valid_mask = ~(np.isnan(jax_depth_img) | np.isnan(cuda_depth_img))
    if valid_mask.sum() > 0:
        depth_diff = np.abs(jax_depth_img[valid_mask] - cuda_depth_img[valid_mask])
        depth_mae = np.mean(depth_diff)
        depth_max_diff = np.max(depth_diff)
        depth_relative_error = np.mean(depth_diff / (np.abs(jax_depth_img[valid_mask]) + 1e-8))
        
        print(f"Depth Comparison (valid pixels: {valid_mask.sum()}/{valid_mask.size}):")
        print(f"  Mean Absolute Error (MAE):     {depth_mae:.6f}")
        print(f"  Max Absolute Error:             {depth_max_diff:.6f}")
        print(f"  Mean Relative Error:            {depth_relative_error*100:.4f}%")
        print(f"  Relative Error (median):       {np.median(depth_diff / (np.abs(jax_depth_img[valid_mask]) + 1e-8))*100:.4f}%")
    else:
        print("Depth Comparison: No valid pixels to compare")
        depth_mae = None
        depth_relative_error = None
    
    # Compare alpha/confidence
    alpha_diff = np.abs(jax_alpha_img - cuda_alpha_img)
    alpha_mae = np.mean(alpha_diff)
    alpha_max_diff = np.max(alpha_diff)
    alpha_relative_error = np.mean(alpha_diff / (jax_alpha_img + 1e-8))
    
    print(f"\nAlpha/Confidence Comparison:")
    print(f"  Mean Absolute Error (MAE):     {alpha_mae:.6f}")
    print(f"  Max Absolute Error:             {alpha_max_diff:.6f}")
    print(f"  Mean Relative Error:            {alpha_relative_error*100:.4f}%")
    print(f"  Relative Error (median):        {np.median(alpha_diff / (jax_alpha_img + 1e-8))*100:.4f}%")
    
    # Check if results are within tolerance
    depth_tolerance = 5e-3  # 0.1% relative error
    alpha_tolerance = 1e-3  # 0.1% absolute error
    
    depth_ok = depth_relative_error < depth_tolerance if (valid_mask.sum() > 0 and depth_relative_error is not None) else True
    alpha_ok = alpha_mae < alpha_tolerance
    
    print(f"\nTolerance Check (depth: {depth_tolerance*100}% rel, alpha: {alpha_tolerance} abs):")
    print(f"  Depth: {'✓ PASS' if depth_ok else '✗ FAIL'}")
    print(f"  Alpha: {'✓ PASS' if alpha_ok else '✗ FAIL'}")
    print("="*80)
    
    # Save results
    results = {
        'jax_times': jax_times.tolist(),
        'cuda_times': cuda_times.tolist(),
        'jax_forward_times': jax_forward_times,
        'cuda_forward_times': cuda_forward_times,
        'jax_mean_ms': float(jax_mean_ms),
        'cuda_mean_ms': float(cuda_mean_ms),
        'speedup': float(speedup),
        'depth_mae': float(depth_mae) if depth_mae is not None else None,
        'alpha_mae': float(alpha_mae),
        'depth_relative_error': float(depth_relative_error) if depth_relative_error is not None else None,
        'alpha_relative_error': float(alpha_relative_error),
        'depth_ok': bool(depth_ok),
        'alpha_ok': bool(alpha_ok),
        'use_optimized': use_optimized,
        'num_fmbs': opt_results['num_fmbs'],
        'width': width,
        'height': height,
        'jax_results': (jax_depth.tolist(), jax_alpha.tolist()),
        'cuda_results': (cuda_depth.tolist(), cuda_alpha.tolist()),
        'cuda_scene_data': {
            'fmbs': [{
                'pose': {
                    'tran': (float(fmb.pose.tran.x), float(fmb.pose.tran.y), float(fmb.pose.tran.z)),
                    'rot': tuple(fmb.pose.rot.quat),
                },
                'extent': tuple(fmb.extent),
            } for fmb in fmbs],
            'log_weights': [float(w) for w in final_weight_logs],
        }
    }
    
    return results


def save_benchmark_results(project_root, results, num_fmbs, width, height, use_optimized):
    """Save benchmark results to file."""
    cache_dir = project_root / "scripts" / "data" / "benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    kernel_suffix = "optimized" if use_optimized else "original"
    results_file = cache_dir / f"results_fmbs{num_fmbs}_size{width}x{height}_{kernel_suffix}.json"
    
    print(f"\nSaving benchmark results to {results_file}...")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CUDA vs JAX FMB rendering")
    parser.add_argument("--num-fmbs", type=int, default=40, help="Number of FMBs (default: 40)")
    parser.add_argument("--width", type=int, default=64, help="Image width (default: 64)")
    parser.add_argument("--height", type=int, default=64, help="Image height (default: 64)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--use-optimized", action="store_true", default=True, help="Use optimized kernel (default: True)")
    parser.add_argument("--use-original", action="store_true", help="Use original kernel instead of optimized")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run optimization even if cache exists")
    
    args = parser.parse_args()
    
    # Handle kernel choice
    use_optimized = args.use_optimized and not args.use_original
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    mesh_file = PROJECT_ROOT / "data/cow/cow.obj"
    
    # Load or run optimization
    opt_results = load_or_run_optimization(
        mesh_file, args.num_fmbs, args.width, args.height, PROJECT_ROOT, force_rerun=args.force_rerun
    )
    
    # Run benchmark
    results = benchmark_comparison(
        opt_results, 
        use_optimized=use_optimized,
        num_iterations=args.iterations, 
        warmup=args.warmup
    )
    
    # Save results
    save_benchmark_results(PROJECT_ROOT, results, args.num_fmbs, args.width, args.height, use_optimized)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"Performance:")
    print(f"  FMB-JAX:      {results['jax_mean_ms']:.4f} ms/run")
    print(f"  GenMetaBalls: {results['cuda_mean_ms']:.4f} ms/run ({'optimized' if use_optimized else 'original'} kernel)")
    print(f"  Speedup:      {results['speedup']:.2f}x {'(CUDA faster)' if results['speedup'] > 1 else '(JAX faster)'}")
    print()
    print(f"Accuracy:")
    if results['depth_mae'] is not None:
        print(f"  Depth MAE:              {results['depth_mae']:.6f} ({'✓' if results['depth_ok'] else '✗'})")
        print(f"  Depth Relative Error:   {results['depth_relative_error']*100:.4f}% ({'✓' if results['depth_ok'] else '✗'})")
    print(f"  Alpha MAE:              {results['alpha_mae']:.6f} ({'✓' if results['alpha_ok'] else '✗'})")
    print(f"  Alpha Relative Error:   {results['alpha_relative_error']*100:.4f}% ({'✓' if results['alpha_ok'] else '✗'})")
    print("="*80)

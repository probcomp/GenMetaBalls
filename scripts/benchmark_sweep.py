#!/usr/bin/env python3
"""Simple parameter sweep wrapper for benchmark_cuda_vs_jax.py"""

import itertools
import subprocess
from pathlib import Path

# Parameter ranges to sweep
NUM_FMBS = [100, 200, 400, 800]
IMAGE_SIZES = [32, 64, 128]  # Square images (width == height)
GRID_SIZES = [(4, 4), (8, 8), (16, 16)]
BLOCK_SIZES = [(8, 8), (16, 16), (32, 32)]
NUM_FMB_CHUNKS = [4, 8, 16, 32]

# Fixed parameters
KERNEL_ID = 1
WARMUP = 1
USE_WANDB = True  # Set to True to enable wandb logging

if __name__ == "__main__":
    script_path = Path(__file__).parent / "benchmark_cuda_vs_jax.py"

    # Generate all combinations
    combinations = list(
        itertools.product(NUM_FMBS, IMAGE_SIZES, GRID_SIZES, BLOCK_SIZES, NUM_FMB_CHUNKS)
    )

    print(f"Running {len(combinations)} benchmark combinations...")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    for idx, (num_fmbs, size, grid_size, block_size, num_fmb_chunks) in enumerate(combinations, 1):
        print(
            f"\n[{idx}/{len(combinations)}] Running: fmbs={num_fmbs}, size={size}x{size}, "
            f"grid={grid_size[0]}x{grid_size[1]}, block={block_size[0]}x{block_size[1]}, "
            f"chunks={num_fmb_chunks}"
        )

        # Generate unique run name for wandb
        run_name = (
            f"fmbs{num_fmbs}_size{size}x{size}_grid{grid_size[0]}x{grid_size[1]}"
            f"_block{block_size[0]}x{block_size[1]}_chunks{num_fmb_chunks}"
        )

        cmd = [
            "pixi",
            "run",
            "python",
            str(script_path),
            "--kernel-id",
            str(KERNEL_ID),
            "--num-fmbs",
            str(num_fmbs),
            "--width",
            str(size),
            "--height",
            str(size),
            "--warmup",
            str(WARMUP),
            "--grid-size",
            str(grid_size[0]),
            str(grid_size[1]),
            "--block-size",
            str(block_size[0]),
            str(block_size[1]),
            "--num-fmb-chunks",
            str(num_fmb_chunks),
            "--save-plot",
        ]

        if USE_WANDB:
            cmd.extend(["--use-wandb", "--run-name", run_name])

        print(f"Running command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            print("✓ Success")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed (exit code {e.returncode})")
            fail_count += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            fail_count += 1

    print("\n" + "=" * 80)
    print(
        f"Summary: {success_count} succeeded, {fail_count} failed out of {len(combinations)} total"
    )

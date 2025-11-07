# GenMetaBalls - Context and Setup Notes

## Project Overview

GenMetaBalls is a hybrid Python/CUDA project that demonstrates GPU-accelerated
computing from Python. The project uses:
- **Build System**: scikit-build-core with CMake
- **Python Bindings**: nanobind (modern, lightweight alternative to pybind11)
- **Package Manager**: pixi (conda-based dependency management)
- **Languages**: Python 3.13, C++20, CUDA

## Directory Structure

```
GenMetaBalls/
├── CMakeLists.txt                    # CMake build configuration
├── pyproject.toml                    # Python project + pixi configuration
├── genmetaballs/
│   └── src/
│       ├── genmetaballs/             # Python package
│       │   ├── __init__.py          # Package initialization
│       │   └── gpu_add.py           # Python wrapper for GPU functions
│       └── cuda/                     # CUDA/C++ source code
│           ├── bindings.cu          # Nanobind Python bindings
│           └── core/                # Core CUDA implementation
│               ├── add.cuh          # GPU addition kernel (header-only)
│               ├── utils.cu         # CUDA error checking implementation
│               └── utils.h          # CUDA utilities header
└── tests/
    ├── test_gpu_add.py              # Python test for GPU addition
    └── test_add.cu                  # Standalone C++ test
```

## Current Implementation

### GPU Addition Function

The project implements a simple GPU vector addition as a demonstration:

**CUDA Kernel** (core/add.cuh):
- `add_kernel`: Element-wise addition on GPU
- `gpu_add<grid_dim, block_dim>`: Template function that:
  1. Allocates device memory
  2. Copies input vectors to GPU
  3. Launches kernel with specified grid/block dimensions
  4. Copies results back to CPU
  5. Cleans up device memory

**Python Bindings** (bindings.cu):
- Uses nanobind to expose GPU functions to Python
- Module name: `_genmetaballs_bindings`
- Constants: GRID_DIM=4096, BLOCK_DIM=1024
- Exposes: `gpu_add` function with fixed grid/block dimensions

**Python Interface** (gpu_add.py):
- Wrapper function: `gpu_add(a: list[float], b: list[float]) -> list[float]`
- Delegates to C++ binding

### Test Suite

**Python Test** (test_gpu_add.py):
- Tests 8196 random float32 values
- Verifies results within 1e-6 tolerance
- Currently has import issues that need fixing

**C++ Test** (test_add.cu):
- Tests 4096 elements with deterministic values
- Verifies exact equality
- Standalone test without Python dependencies

## Expected Workflow

Once fixed, the workflow should be:

1. **Install dependencies**: `pixi install`
   - Should install Python, CUDA toolkit, compilers, build tools

2. **Run tests**: `pixi run python tests/test_gpu_add.py`
   - Loads the compiled `_genmetaballs_bindings` module
   - Calls GPU addition function
   - Verifies results

3. **Reinstalling and rebuilding**: `pixi reinstall genmetaballs`
   - After editing the source codes rebuild using this comment

## Build Process (scikit-build-core)

When `pixi install` runs, scikit-build-core will:
1. Configure CMake with specified options
2. Build the C++/CUDA code
3. Create the nanobind module `_genmetaballs_bindings`
4. Install the module into the Python package directory
5. Make `genmetaballs` importable in Python

## Notes

- Platform: linux-64 only (specified in pixi config)
- License: MIT
- Authors: Arijit Dasgupta, Matin Ghavami, Xiaoyan Wang
- This is a template project for future metaballs generation implementation
- Current gpu_add is demonstration code to verify the build system works

# GenMetaBalls - Context and Setup Notes

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenMetaBalls is a hybrid Python/CUDA project that demonstrates GPU-accelerated
computing from Python. The project uses:
- **Build System**: scikit-build-core with CMake
- **Python Bindings**: nanobind (modern, lightweight alternative to pybind11)
- **Package Manager**: pixi (conda-based dependency management)
- **Languages**: Python 3.13, C++20, CUDA

**Critical Architecture Pattern:**
The project uses a three-layer architecture:
1. **CUDA Layer** (`genmetaballs/src/cuda/core/`): Header-only template kernels (e.g., `add.cuh`)
2. **Bindings Layer** (`genmetaballs/src/cuda/bindings.cu`): nanobind module `_genmetaballs_bindings` that instantiates templates with fixed parameters
3. **Python Layer** (`genmetaballs/src/genmetaballs/`): User-facing Python API that wraps the bindings

This layering allows compile-time optimization via templates while exposing a simple Python interface.

## Common Commands

### Initial Setup
```bash
pixi install
```

### After Modifying C++/CUDA Code
```bash
pixi reinstall genmetaballs
```
This triggers scikit-build-core to rebuild the C++/CUDA code and reinstall the Python module.

### Testing
```bash
# C++/CUDA tests (Google Test via ctest)
pixi run ctest

# Python tests (pytest)
pixi run pytest

# All tests
pixi run test
```

## Build Process Details

When `pixi install` or `pixi reinstall genmetaballs` runs, scikit-build-core:
1. Configures CMake with settings from `pyproject.toml`
2. Builds three CMake targets:
   - `genmetaballs_core`: Static library with CUDA kernels and utilities
   - `_genmetaballs_bindings`: nanobind extension module (installed to Python package)
   - `test_add`: C++ test executable
3. Installs `_genmetaballs_bindings.abi3.so` into the `genmetaballs` package directory
4. Makes `genmetaballs` importable in Python

**Important:** The build uses a persistent build directory (`build/`) specified in `pyproject.toml` to speed up rebuilds.

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

## Adding New CUDA Functionality

To add a new GPU function following the existing pattern:

1. **Create kernel in `cuda/core/`**: Header-only template (`.cuh` file)
2. **Expose in `bindings.cu`**: Add `m.def()` call to instantiate template with fixed parameters
3. **Wrap in Python**: Create wrapper in `genmetaballs/` and export in `__init__.py`
4. **Rebuild**: Run `pixi reinstall genmetaballs`
5. **Test**: Add C++ test in `tests/test_*.cu` and Python test in `tests/test_*.py`

## Testing

The project has two test suites:

1. **C++/CUDA Tests**: Configured using Google Test (powered by ctest)
   ```bash
   pixi run ctest
   ```

2. **Python Tests**: Configured using pytest
   ```bash
   pixi run pytest
   ```

3. **Run All Tests**: Run both test suites together
   ```bash
   pixi run test
   ```


## Build Process (scikit-build-core)

When `pixi install` runs, scikit-build-core will:
1. Configure CMake with specified options
2. Build the C++/CUDA code
3. Create the nanobind module `_genmetaballs_bindings`
4. Install the module into the Python package directory
5. Make `genmetaballs` importable in Python

## Platform and Dependencies

- Platform: linux-64 only
- Python: 3.13+ required (uses stable ABI via nanobind)
- CUDA: 12.8.x
- CMake: 4.1+ (note: higher than typical minimum due to CUDA requirements)

## Notes

- Platform: linux-64 only (specified in pixi config)
- License: MIT
- Authors: Arijit Dasgupta, Matin Ghavami, Xiaoyan Wang
- This is a template project for future metaballs generation implementation
- Current gpu_add is demonstration code to verify the build system works

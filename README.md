# GenMetaBalls

Let's get the ball rolling with blazing fast CUDA kernels!

## Installation

### Usage Setup

To simply run `genmetaballs`:

```bash
pixi install
```

### Development Setup

For development:

```bash
pixi install
pixi run dev-setup
```

The `dev-setup` task sets up [pre-commit](https://pre-commit.com/) git hooks:
- **Pre-commit**: Formats and lints code before each commit
- **Pre-push**: Runs all tests before pushes

The `dev-setup` task also generates `compile_commands.json` (needed for accurate C++/CUDA linting).


## Testing

### C++/CUDA Tests

C++/CUDA tests are configured using [Google Test](https://github.com/google/googletest) (powered by [ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html)). Run them with:

```bash
pixi run ctest
```

### Python Tests

Python tests are configured using [pytest](https://docs.pytest.org/en/stable/). Run them with:

```bash
pixi run pytest
```

### Run All Tests

To run both C++/CUDA and Python tests together:

```bash
pixi run test
```

## Formatting & Linting

All commands work on both C++/CUDA and Python files automatically.

Format all files:
```bash
pixi run format
```

Lint all files:
```bash
pixi run lint
```

Auto-fix linting issues and format files:
```bash
pixi run fix
```

These commands are automatically run by the pre-commit hooks when you commit code.

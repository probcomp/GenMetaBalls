# GenMetaBalls

Let's get the ball rolling with blazing fast CUDA kernels!

## Installation

For initial installation, run:

```bash
pixi install
```

### Development Setup

To install and use GenMetaBalls:

```bash
pixi install
```

For development (formatting, linting, git hooks, testing):

```bash
pixi install
pixi run dev-setup
```

This sets up [pre-commit](https://pre-commit.com/) git hooks:
- **Pre-commit**: Formats and lints code before each commit
- **Pre-push**: Runs all tests before pushes

Run the hooks manually if needed:
```bash
pixi run pre-commit-run        # Run formatting/linting checks
pixi run pre-commit-run-push   # Run all tests
```
```

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

### Formatting

Format all files (C++/CUDA + Python):
```bash
pixi run format
```

Check formatting without modifying files:
```bash
pixi run format-check
```

### Linting

Lint all files:
```bash
pixi run lint
```

Auto-fix linting issues:
```bash
pixi run lint-fix
```

**Language-specific commands:**
- `format-cpp`, `format-python` - Format specific language
- `format-check-cpp`, `format-check-python` - Check formatting for specific language
- `lint-cpp`, `lint-python` - Lint specific language
- `lint-cpp-fix`, `lint-python-fix` - Auto-fix linting issues for specific language
- `lint-full` - C++/CUDA linting with compile_commands.json (more accurate)
- `lint-full-fix` - Auto-fix C++/CUDA issues using compile_commands.json

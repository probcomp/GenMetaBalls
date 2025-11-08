# GenMetaBalls

Let's get the ball rolling with blazing fast CUDA kernels!

## Installation

For initial installation, run:

```bash
pixi install
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

from . import _genmetaballs_bindings as _gmbb


def gpu_add(a: list[float], b: list[float]) -> list[float]:
    return _gmbb.gpu_add(a, b)

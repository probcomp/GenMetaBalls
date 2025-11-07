from genmetaballs import gpu_add
import numpy as np


def test_gpu_add(seed):
    N = 8196
    rng = np.random.default_rng(seed)
    a = rng.normal(size=N).astype(np.float32).tolist()
    b = rng.normal(size=N).astype(np.float32).tolist()
    c = gpu_add(a, b)
    assert all(abs(x + y - z) < 1e-6 for (x, y, z) in zip(a, b, c))

if __name__ == '__main__':
    test_gpu_add(0)
    print('All tests passed.')


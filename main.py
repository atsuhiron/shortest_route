import numpy as np

from const import SearchMode
from algorithm.all_search_optimizer import AllSearchOptimizer
import plot


def gen_data(num: int, dim: int, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.random((num, dim)).astype(np.float32)


if __name__ == "__main__":
    data = gen_data(10, 2, 1111)
    optim = AllSearchOptimizer(data, SearchMode.FIX_START_GOAL)
    ret = optim.optimize()
    print(ret)
    plot.plot(ret.get_route())

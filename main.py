import numpy as np

from const import SearchMode
from algorithm.all_search_optimizer import AllSearchOptimizer
import plot


def gen_data(num: int, dim: int) -> np.ndarray:
    return np.random.random((num, dim)).astype(np.float32)


if __name__ == "__main__":
    data = gen_data(9, 2)
    optim = AllSearchOptimizer(data, SearchMode.FREE)
    ret = optim.optimize()
    print(ret)
    plot.plot(ret.get_route())
import numpy as np

from const import SearchMode
from algorithm.all_search_optimizer import AllSearchOptimizer
from algorithm.all_search_optimizer_low_mem import AllSearchOptimizerLowMem
from algorithm.all_search_nogil_mt_chunk_optimizer import AllSearchNGMTChunkOptimizer
from algorithm.two_opt_optimizer import TwoOptOptimizer
from algorithm.two_opt_mp_optimizer import TwoOptMPOptimizer
import plot


def gen_data(num: int, dim: int, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.random((num, dim)).astype(np.float32)


if __name__ == "__main__":
    __spec__ = None
    data = gen_data(13, 2, 53)
    # optim = AllSearchOptimizer(data, SearchMode.FIX_START_GOAL)
    optim = AllSearchOptimizerLowMem(data, SearchMode.FIX_START_GOAL)
    # optim = AllSearchNGMTChunkOptimizer(data, SearchMode.FIX_START_GOAL, proc_num=10)
    # optim = TwoOptOptimizer(data, SearchMode.FIX_START_GOAL, 5000)
    # optim = TwoOptMPOptimizer(data, SearchMode.FIX_START_GOAL, 5000, proc_num=10)
    ret = optim.optimize()
    print(ret)
    plot.plot(ret.get_route())

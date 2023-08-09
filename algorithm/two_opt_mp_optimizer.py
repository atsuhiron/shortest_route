from typing import Generator
import time
from multiprocessing.pool import Pool

import numpy as np
import numba

from algorithm.two_opt_optimizer import TwoOptOptimizer
from algorithm.two_opt_optimizer import optimize_core
from algorithm.two_opt_optimizer import optimize_ann_core
import route_result
from const import SearchMode


@numba.jit("u1[:](u1[:], u1, u1)", nopython=True)
def _two_opt(arr: np.ndarray, idx1: int, idx2: int):
    copied = arr.copy()
    copied[idx1], copied[idx2] = arr[idx2], arr[idx1]
    return copied


@numba.jit("Tuple((u1[:], f8, i8))(Tuple((f4[:, :], u1[:], u1[:, :], i8)))", nopython=True)
def _optimize(args: tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> tuple[np.ndarray, float, int]:
    arr, init_order, opt_patterns, index = args
    min_order, min_length = optimize_core(arr, init_order, opt_patterns)
    return min_order, min_length, index


@numba.jit("Tuple((u1[:], f8, i8))(Tuple((f4[:, :], u1[:], u1[:, :], i8, f8)))", nopython=True)
def _optimize_ann(args: tuple[np.ndarray, np.ndarray, np.ndarray, int, float]) -> tuple[np.ndarray, float, int]:
    arr, init_order, opt_patterns, index, k = args
    min_order, min_length = optimize_ann_core(arr, init_order, opt_patterns, k)
    return min_order, min_length, index


class TwoOptMPOptimizer(TwoOptOptimizer):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode, init_num: int, proc_num: int, pool: Pool = None,
                 k: float = 0, verbose: bool = True):
        super().__init__(arr, search_mode, init_num, k)
        self.proc_num = proc_num
        self.verbose = verbose
        self.pool = pool

    def optimize(self) -> route_result.RouteResult:
        init_orders = self.gen_init_orders()
        opt_patterns = self.gen_all_opt_pattern()

        start = time.time()
        cs = self.calc_chunk_size(self.init_num, self.proc_num)
        if self.verbose:
            print(f"chunksize={cs}")

        if self.k <= 0:
            optimize_func = _optimize
        else:
            optimize_func = _optimize_ann

        min_orders = np.empty((self.init_num, len(self.arr)), dtype=np.uint8)
        min_lengths = np.empty(self.init_num, dtype=np.float32)
        if self.pool is None:
            with Pool(self.proc_num) as pool:
                for ret in pool.imap_unordered(optimize_func, self.gen_args(init_orders, opt_patterns), cs):
                    _min_order, _min_length, index = ret
                    min_orders[index] = _min_order
                    min_lengths[index] = _min_length
        else:
            for ret in self.pool.imap_unordered(optimize_func, self.gen_args(init_orders, opt_patterns), cs):
                _min_order, _min_length, index = ret
                min_orders[index] = _min_order
                min_lengths[index] = _min_length

        min_init_idx = np.argmin(min_lengths)
        elapsed = time.time() - start

        return route_result.RouteResult(self.arr, min_orders[min_init_idx], min_lengths[min_init_idx], elapsed)

    def gen_args(self, init_orders: np.ndarray, opt_patterns: np.ndarray) \
            -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, int], None, None] | \
            Generator[tuple[np.ndarray, np.ndarray, np.ndarray, int, float], None, None]:
        count = 0
        if self.k <= 0:
            for init_order in init_orders:
                yield self.arr, init_order, opt_patterns, count
                count += 1
        else:
            for init_order in init_orders:
                yield self.arr, init_order, opt_patterns, count, self.k
                count += 1

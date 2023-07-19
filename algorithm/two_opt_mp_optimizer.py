from typing import Generator
import time
from multiprocessing.pool import Pool

import numpy as np
import numba

from algorithm.base_route_optimizer import calc_route_length_f4
from algorithm.two_opt_optimizer import TwoOptOptimizer
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
    min_order = init_order
    min_length = calc_route_length_f4(arr, min_order)

    while True:
        check_index = np.arange(len(opt_patterns)).astype(np.uint8)
        np.random.shuffle(check_index)

        for idx in check_index:
            swapped_order = _two_opt(min_order, opt_patterns[idx, 0], opt_patterns[idx, 1])
            swapped_length = calc_route_length_f4(arr, swapped_order)
            if swapped_length < min_length:
                min_length = swapped_length
                min_order = swapped_order
                break
        else:
            break
    return min_order, min_length, index


class TwoOptMPOptimizer(TwoOptOptimizer):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode, init_num: int, proc_num: int, pool: Pool = None,
                 verbose: bool = True):
        super().__init__(arr, search_mode, init_num)
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

        min_orders = np.empty((self.init_num, len(self.arr)), dtype=np.uint8)
        min_lengths = np.empty(self.init_num, dtype=np.float32)
        if self.pool is None:
            with Pool(self.proc_num) as pool:
                for ret in pool.imap_unordered(_optimize, self.gen_args(init_orders, opt_patterns), cs):
                    _min_order, _min_length, index = ret
                    min_orders[index] = _min_order
                    min_lengths[index] = _min_length
        else:
            for ret in self.pool.imap_unordered(_optimize, self.gen_args(init_orders, opt_patterns), cs):
                _min_order, _min_length, index = ret
                min_orders[index] = _min_order
                min_lengths[index] = _min_length

        min_init_idx = np.argmin(min_lengths)
        elapsed = time.time() - start

        return route_result.RouteResult(self.arr, min_orders[min_init_idx], min_lengths[min_init_idx], elapsed)

    def gen_args(self, init_orders: np.ndarray, opt_patterns: np.ndarray) \
            -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, int], None, None]:
        count = 0
        for init_order in init_orders:
            yield self.arr, init_order, opt_patterns, count
            count += 1

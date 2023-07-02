import time
import itertools as it

import numpy as np
import numba
import tqdm

from algorithm.base_route_optimizer import calc_route_length_f4
from algorithm.base_route_optimizer import reconst_full_order
from algorithm.base_route_optimizer import BaseRouteOptimizer
import route_result
from const import SearchMode


@numba.jit("u1[:](u1[:], u1, u1)", cache=True, nopython=True)
def two_opt(arr: np.ndarray, idx1: int, idx2: int):
    copied = arr.copy()
    copied[idx1], copied[idx2] = arr[idx2], arr[idx1]
    return copied


@numba.jit("Tuple((u1[:], f8))(f4[:, :], u1[:], u1[:, :])", cache=True, nopython=True)
def _optimize(arr: np.ndarray, init_order: np.ndarray, opt_patterns: np.ndarray) -> tuple[np.ndarray, float]:
    min_order = init_order
    min_length = calc_route_length_f4(arr, min_order)

    while True:
        check_index = np.arange(len(opt_patterns)).astype(np.uint8)
        np.random.shuffle(check_index)

        for idx in check_index:
            swapped_order = two_opt(min_order, opt_patterns[idx, 0], opt_patterns[idx, 1])
            swapped_length = calc_route_length_f4(arr, swapped_order)
            if swapped_length < min_length:
                min_length = swapped_length
                min_order = swapped_order
                break
        else:
            break
    return min_order, min_length


class TwoOptOptimizer(BaseRouteOptimizer):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode, init_num: int):
        super().__init__(arr, search_mode)
        self.init_num = init_num

    def optimize(self) -> route_result.RouteResult:
        init_orders = self.gen_init_orders()
        opt_patterns = self.gen_all_opt_pattern()

        start = time.time()
        min_orders = []
        min_lengths = []
        for ii in tqdm.tqdm(range(self.init_num)):
            _min_order, _min_length = _optimize(self.arr, init_orders[ii], opt_patterns)
            min_orders.append(_min_order)
            min_lengths.append(_min_length)
        elapsed = time.time() - start

        min_init_idx = np.argmin(min_lengths)
        return route_result.RouteResult(self.arr, min_orders[min_init_idx], min_lengths[min_init_idx], elapsed)

    def gen_init_orders(self) -> np.ndarray:
        init_pattern = np.vstack([np.arange(len(self.arr)).astype(np.uint8) for _ in range(self.init_num)])

        if self.search_mode == SearchMode.FREE:
            for ii in range(self.init_num):
                np.random.shuffle(init_pattern[ii])
            return init_pattern
        if self.search_mode == SearchMode.FIX_START:
            for ii in range(self.init_num):
                np.random.shuffle(init_pattern[ii, 1:])
            return init_pattern
        if self.search_mode == SearchMode.FIX_GOAL:
            for ii in range(self.init_num):
                np.random.shuffle(init_pattern[ii, :-1])
            return init_pattern
        for ii in range(self.init_num):
            np.random.shuffle(init_pattern[ii, 1:-1])
        return init_pattern

    def gen_all_opt_pattern(self) -> np.ndarray:
        if self.search_mode == SearchMode.FREE:
            return np.array(list(it.combinations(range(len(self.arr)), 2)), dtype=np.uint8)
        if self.search_mode == SearchMode.FIX_START:
            return np.array(list(it.combinations(range(1, len(self.arr)), 2)), dtype=np.uint8)
        if self.search_mode == SearchMode.FIX_GOAL:
            return np.array(list(it.combinations(range(len(self.arr) - 1), 2)), dtype=np.uint8)
        return np.array(list(it.combinations(range(1, len(self.arr) - 1), 2)), dtype=np.uint8)

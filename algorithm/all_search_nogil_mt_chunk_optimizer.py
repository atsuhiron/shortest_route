from typing import Generator
import time
import threading

import numpy as np
import numba

from algorithm.base_route_optimizer import calc_route_length_f4
from algorithm.base_route_optimizer import reconst_full_order
from algorithm.base_route_optimizer import BaseRouteOptimizer
import route_result
from const import SearchMode


@numba.jit("void(f4[:, :], u1[:, :], i8, i8, f4[:])", nopython=True, nogil=True)
def _optimize_chunk(arr: np.ndarray,
                    orders: np.ndarray,
                    search_mode: int,
                    start_index: int,
                    out_len_arr: np.ndarray) -> None:
    diff = 0
    for order in orders:
        order = reconst_full_order(order, search_mode)
        out_len_arr[start_index + diff] = calc_route_length_f4(arr, order)
        diff += 1


class AllSearchNGMTChunkOptimizer(BaseRouteOptimizer):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode, thread_num: int):
        super().__init__(arr, search_mode)
        self.length_arr = None
        self.thread_num = thread_num

    def optimize(self) -> route_result.RouteResult:
        perm_obj, total_num = self.get_sliced_perm_by_search_mode()
        perm_arr = self.perm_to_arr(perm_obj)
        self.length_arr = np.empty(total_num, dtype=np.float32)

        start = time.time()
        threads = [threading.Thread(target=_optimize_chunk, args=arg) for arg in self.gen_args(perm_arr)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        min_index = np.argmin(self.length_arr)
        min_order = reconst_full_order(perm_arr[min_index], self.search_mode.value)
        elapsed = time.time() - start

        return route_result.RouteResult(self.arr, min_order, self.length_arr[min_index], elapsed)

    @staticmethod
    def perm_to_arr(perm_obj) -> np.ndarray:
        return np.array(list(perm_obj), dtype=np.uint8)

    def gen_args(self, perm_arr: np.ndarray) -> Generator[tuple[np.ndarray, np.ndarray, int, int], None, None]:
        cs = len(perm_arr) // self.thread_num
        for ci in range(self.thread_num):
            yield self.arr, perm_arr[ci * cs: (ci + 1) * cs], self.search_mode.value, ci * cs, self.length_arr


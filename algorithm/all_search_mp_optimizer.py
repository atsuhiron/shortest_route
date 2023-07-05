from typing import Generator

import time
from multiprocessing.pool import Pool

import numpy as np
import numba

from algorithm.base_route_optimizer import calc_route_length_f4
from algorithm.base_route_optimizer import reconst_full_order
from algorithm.base_route_optimizer import BaseRouteOptimizer
import route_result
from const import SearchMode

"""
Too late :(
"""


@numba.jit("Tuple((f8, i8))(Tuple((f4[:, :], u1[:], i8, i8)))", nopython=True, nogil=True)
def _optimize_single(arg: tuple[np.ndarray, np.ndarray, int, int]) -> tuple[float, int]:
    arr, order, search_mode, index = arg
    order = reconst_full_order(order, search_mode)
    return calc_route_length_f4(arr, order), index


class AllSearchMPOptimizer(BaseRouteOptimizer):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode, proc_num: int):
        super().__init__(arr, search_mode)
        self.proc_num = proc_num

    def optimize(self) -> route_result.RouteResult:
        perm_obj, total_num = self.get_sliced_perm_by_search_mode()
        perm_arr = self.perm_to_arr(perm_obj)
        length_arr = np.empty(total_num, dtype=np.float32)

        start = time.time()
        cs = min(max(int(total_num / self.proc_num / 100), 1), 1000)
        print(f"chunksize={cs}")

        with Pool(self.proc_num) as pool:
            for ret in pool.imap_unordered(_optimize_single, self.gen_args(perm_arr), chunksize=cs):
                length, index = ret
                length_arr[index] = length

        min_index = np.argmin(length_arr)
        min_order = reconst_full_order(perm_arr[min_index], self.search_mode.value)
        elapsed = time.time() - start

        return route_result.RouteResult(self.arr, min_order, length_arr[min_index], elapsed)

    @staticmethod
    def perm_to_arr(perm_obj) -> np.ndarray:
        return np.array(list(perm_obj), dtype=np.uint8)

    def gen_args(self, perm_arr: np.ndarray) -> Generator[tuple[np.ndarray, np.ndarray, int, int], None, None]:
        count = 0
        for perm in perm_arr:
            yield self.arr, perm, self.search_mode.value, count
            count += 1

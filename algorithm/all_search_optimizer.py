import time

import numpy as np
import numba
from tqdm import tqdm

from algorithm.base_route_optimizer import calc_route_length_f4
from algorithm.base_route_optimizer import BaseRouteOptimizer
import route_result
from const import SearchMode


class AllSearchOptimizer(BaseRouteOptimizer):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode):
        super().__init__(arr, search_mode)

    def optimize(self) -> route_result.RouteResult:
        sliced_arr = self.slice_by_search_mode()
        perm_obj, total_num = self.get_sliced_perm_by_search_mode()
        perm_arr = self.perm_to_arr(perm_obj, len(sliced_arr), total_num)

        start = time.time()
        min_order, min_length = self._optimize(sliced_arr, perm_arr)
        elapsed = time.time() - start
        min_order = self.reconst_full_order(min_order)
        return route_result.RouteResult(self.arr, min_order, min_length, elapsed)

    @staticmethod
    def perm_to_arr(perm_obj, data_length: int, perm_length: int) -> np.ndarray:
        return np.array(list(perm_obj), dtype=np.int8)

    @staticmethod
    @numba.jit("Tuple((i1[:], f8))(f4[:, :], i1[:, :])", cache=True, nopython=True)
    def _optimize(sliced_arr: np.ndarray, orders: np.ndarray) -> tuple[np.ndarray, float]:
        min_order = None
        min_length = 1e100

        for order in orders:
            length = calc_route_length_f4(sliced_arr, order)

            if length < min_length:
                min_length = length
                min_order = order
        return min_order, min_length

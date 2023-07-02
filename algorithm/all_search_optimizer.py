import time

import numpy as np
import numba

from algorithm.base_route_optimizer import calc_route_length_f4
from algorithm.base_route_optimizer import reconst_full_order
from algorithm.base_route_optimizer import BaseRouteOptimizer
import route_result
from const import SearchMode


@numba.jit("Tuple((i1[:], f8))(f4[:, :], i1[:, :], i8)", nopython=True)
def _optimize(arr: np.ndarray, orders: np.ndarray, search_mode: int) -> tuple[np.ndarray, float]:
    min_order = None
    min_length = 1e100

    for order in orders:
        order = reconst_full_order(order, search_mode)
        length = calc_route_length_f4(arr, order)

        if length < min_length:
            min_length = length
            min_order = order
    return min_order, min_length


class AllSearchOptimizer(BaseRouteOptimizer):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode):
        super().__init__(arr, search_mode)

    def optimize(self) -> route_result.RouteResult:
        perm_obj, total_num = self.get_sliced_perm_by_search_mode()
        perm_arr = self.perm_to_arr(perm_obj)

        start = time.time()
        min_order, min_length = _optimize(self.arr, perm_arr, self.search_mode.value)
        elapsed = time.time() - start

        return route_result.RouteResult(self.arr, min_order, min_length, elapsed)

    @staticmethod
    def perm_to_arr(perm_obj) -> np.ndarray:
        return np.array(list(perm_obj), dtype=np.int8)

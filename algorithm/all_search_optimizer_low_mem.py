import time
import itertools as it

import numpy as np
import tqdm

from algorithm.base_route_optimizer import calc_route_length_f4
from algorithm.base_route_optimizer import reconst_full_order
from algorithm.base_route_optimizer import BaseRouteOptimizer
import route_result
from const import SearchMode


def _optimize(arr: np.ndarray, perm_obj: it.permutations, search_mode: int, total_num: int) -> tuple[np.ndarray, float]:
    min_order = None
    min_length = 1e100

    for order in tqdm.tqdm(perm_obj, total=total_num, miniters=0.25):
        order = reconst_full_order(np.array(order, dtype=np.uint8), search_mode)
        length = calc_route_length_f4(arr, order)

        if length < min_length:
            min_length = length
            min_order = order
    return min_order, min_length


class AllSearchOptimizerLowMem(BaseRouteOptimizer):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode):
        super().__init__(arr, search_mode)

    def optimize(self) -> route_result.RouteResult:
        perm_obj, total_num = self.get_sliced_perm_by_search_mode()

        start = time.time()
        min_order, min_length = _optimize(self.arr, perm_obj, self.search_mode.value, total_num)
        elapsed = time.time() - start

        return route_result.RouteResult(self.arr, min_order, min_length, elapsed)

import abc
import itertools as it
import math

import numpy as np
import numba

import route_result
from const import SearchMode


@numba.jit("f8(f4[:,:], i1[:])", cache=True, nopython=True)
def calc_route_length_f4(arr: np.ndarray, order: np.ndarray) -> float:
    arr = arr[order]
    edge_vecs = arr[1:] - arr[:-1]
    edge_norms = np.empty(len(edge_vecs), dtype=np.float32)

    for ii in range(len(edge_vecs)):
        edge_norms[ii] = np.sqrt(np.sum(np.square(edge_vecs[ii])))
    return float(sum(edge_norms))


class BaseRouteOptimizer(metaclass=abc.ABCMeta):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode):
        self.arr = arr
        self.search_mode = search_mode

    @abc.abstractmethod
    def optimize(self) -> route_result.RouteResult:
        pass

    def get_sliced_perm_by_search_mode(self) -> tuple[it.permutations, int]:
        actual_size = len(self.arr) - 2
        if self.search_mode == SearchMode.FREE:
            actual_size = len(self.arr)
        if self.search_mode == SearchMode.FIX_START:
            actual_size = len(self.arr) - 1
        if self.search_mode == SearchMode.FIX_GOAL:
            actual_size = len(self.arr) - 1
        return it.permutations(range(actual_size)), math.factorial(actual_size)

    @staticmethod
    def reconst_full_orders(orders: np.ndarray, search_mode_int: int) -> np.ndarray:
        """
        Reconstruct all orders.
        """
        if search_mode_int == 0:
            # FREE
            return orders
        if search_mode_int == 1:
            # FIX_STAR
            orders += 1
            return np.insert(orders, 0, 0, axis=1)
        if search_mode_int == 2:
            # FIX_GOAL
            return np.insert(orders, len(orders[0]), len(orders[0]), axis=1)
        # FIX_START_GOAL
        orders += 1
        orders = np.insert(orders, 0, 0, axis=1)
        return np.insert(orders, len(orders[0]), len(orders[0]), axis=1)

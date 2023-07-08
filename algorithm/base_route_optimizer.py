import abc
import itertools as it
import math

import numpy as np
import numba
from numba.typed import List

import route_result
from const import SearchMode


@numba.jit("f8(f4[:,:], u1[:])", nopython=True)
def calc_route_length_f4(arr: np.ndarray, order: np.ndarray) -> float:
    arr = arr[order]
    edge_vecs = arr[1:] - arr[:-1]
    edge_norms = np.empty(len(edge_vecs), dtype=np.float32)

    for ii in range(len(edge_vecs)):
        edge_norms[ii] = np.sqrt(np.sum(np.square(edge_vecs[ii])))
    return float(sum(edge_norms))


@numba.jit("u1[:](u1[:], i8)", nopython=True)
def reconst_full_order(order: np.ndarray, search_mode_int: int) -> np.ndarray:
    """
    Reconstruct an order.
    """
    if search_mode_int == 0:
        # FREE
        return order
    if search_mode_int == 1:
        # FIX_STAR
        order_list = List(order)
        order_list.insert(0, np.uint8(0))
        return np.array(list(order_list), dtype=np.uint8)
    if search_mode_int == 2:
        # FIX_GOAL
        order_list = List(order)
        order_list.append(np.uint8(len(order_list)))
        return np.array(list(order_list), dtype=np.uint8)
    # FIX_START_GOAL
    order_list = List(order)
    order_list.insert(0, np.uint8(0))
    order_list.append(np.uint8(len(order_list)))
    return np.array(list(order_list), dtype=np.uint8)


class BaseRouteOptimizer(metaclass=abc.ABCMeta):
    def __init__(self, arr: np.ndarray, search_mode: SearchMode):
        self.arr = arr
        self.search_mode = search_mode

    @abc.abstractmethod
    def optimize(self) -> route_result.RouteResult:
        pass

    def get_sliced_perm_by_search_mode(self) -> tuple[it.permutations, int]:
        size = len(self.arr)
        if self.search_mode == SearchMode.FREE:
            return it.permutations(range(size)), math.factorial(size)
        if self.search_mode == SearchMode.FIX_START:
            return it.permutations(range(1, size)), math.factorial(size - 1)
        if self.search_mode == SearchMode.FIX_GOAL:
            return it.permutations(range(size - 1)), math.factorial(size - 1)
        return it.permutations(range(1, size - 1)), math.factorial(size - 2)

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

    @staticmethod
    def calc_chunk_size(total_num: int, proc_num: int):
        return min(max(int(total_num / proc_num / 100), 1), 1000)

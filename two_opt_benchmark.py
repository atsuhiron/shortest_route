import os
import shutil
from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from algorithm.base_route_optimizer import SearchMode
from algorithm.two_opt_mp_optimizer import TwoOptMPOptimizer

_SAMPLE_DIR = "sample_points"
_POINT_NUMS = [20, 30, 40, 50, 60, 70, 80, 90, 100]


def _get_sample_name(num: int) -> str:
    return f"{_SAMPLE_DIR}/sample_{num}.npy"


def _create_sample_arrays(dim: int = 2, override: bool = False):
    assert dim >= 2, "Dimension must be equal or larger than 2"

    already_exists = os.path.exists(_SAMPLE_DIR) and os.path.isdir(_SAMPLE_DIR)
    if override and already_exists:
        print(f"{_SAMPLE_DIR} is exists. Overriding...")
        shutil.rmtree(_SAMPLE_DIR)
        os.makedirs(_SAMPLE_DIR)
    elif already_exists:
        print(f"{_SAMPLE_DIR} is exists. Nothing is done.")
        return
    else:
        os.makedirs(_SAMPLE_DIR)

    for point_num in _POINT_NUMS:
        ra = np.random.random((point_num, dim)).astype(np.float32)
        np.save(_get_sample_name(point_num), ra)


def _plot_time_benchmark(result_dict: dict[str, list]):
    plt.errorbar(result_dict["points"], result_dict["time_avg"], result_dict["time_std"], fmt="o")
    plt.xlabel("Number of point")
    plt.ylabel("time [s]")
    plt.show()


def run_time_benchmark(trial_num: int, init_num: int | list[int]):
    sample_size = len(_POINT_NUMS)
    if isinstance(trial_num, list):
        assert len(trial_num) == sample_size, f"The length of list of trial number must be {sample_size}"
    else:
        trial_num = [trial_num] * sample_size
    if isinstance(init_num, list):
        assert len(init_num) == sample_size, f"The length of list of initial number must be {sample_size}"
    else:
        init_num = [init_num] * sample_size

    proc_num = os.cpu_count() // 2
    time_record = [np.zeros(trial_num[si]) for si in range(sample_size)]
    length_record = [np.zeros(trial_num[si]) for si in range(sample_size)]

    with Pool(proc_num) as pool:
        for si in tqdm(range(sample_size), desc="All"):
            arr = np.load(_get_sample_name(_POINT_NUMS[si]))
            for ti in tqdm(range(trial_num[si]), desc=f"sample {_POINT_NUMS[si]}", leave=False):
                optim = TwoOptMPOptimizer(arr, SearchMode.FREE, init_num[si], proc_num, pool, False)
                result = optim.optimize()
                time_record[si][ti] = result.search_time_sec
                length_record[si][ti] = result.length

    benchmark_res = {"points": _POINT_NUMS,
                     "time_avg": [float(trials[1:].mean()) for trials in time_record],
                     "time_std": [float(trials[1:].std()) for trials in time_record],
                     "length_mean": [float(trials.mean()) for trials in length_record],
                     "length_std": [float(trials.std()) for trials in length_record]}
    _plot_time_benchmark(benchmark_res)
    return benchmark_res

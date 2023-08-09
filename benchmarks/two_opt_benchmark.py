import os
from multiprocessing.pool import Pool

from tqdm import tqdm
import numpy as np

from algorithm.base_route_optimizer import SearchMode
from algorithm.two_opt_mp_optimizer import TwoOptMPOptimizer
import benchmarks.benchmark_util as bmu


def run_time_benchmark(trial_num: int, init_num: int | list[int]) -> dict[str, list]:
    sample_size = len(bmu.POINT_NUMS)
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
            arr = np.load(bmu.get_sample_name(bmu.POINT_NUMS[si]))
            for ti in tqdm(range(trial_num[si]), desc=f"sample {bmu.POINT_NUMS[si]}", leave=False):
                optim = TwoOptMPOptimizer(arr, SearchMode.FREE, init_num[si], proc_num, pool, 0, False)
                result = optim.optimize()
                time_record[si][ti] = result.search_time_sec
                length_record[si][ti] = result.length

    benchmark_res = {"points": bmu.POINT_NUMS,
                     "time_avg": [float(trials[1:].mean()) for trials in time_record],
                     "time_std": [float(trials[1:].std()) for trials in time_record],
                     "length_avg": [float(trials.mean()) for trials in length_record],
                     "length_std": [float(trials.std()) for trials in length_record]}
    bmu.plot_time_benchmark(benchmark_res)
    return benchmark_res


def run_length_benchmark(point_num: int, trial_num: int) -> dict[str, list]:
    sample_size = len(bmu.INIT_NUMS)

    proc_num = os.cpu_count() // 2
    length_record = np.zeros((sample_size, trial_num))

    try:
        arr = np.load(bmu.get_sample_name(point_num))
    except FileNotFoundError:
        arr = np.random.random((point_num, 2)).astype(np.float32)

    with Pool(proc_num) as pool:
        for si in tqdm(range(sample_size), desc="All"):
            for ti in tqdm(range(trial_num), desc=f"init_num {bmu.INIT_NUMS[si]}", leave=False):
                optim = TwoOptMPOptimizer(arr, SearchMode.FREE, bmu.INIT_NUMS[si], proc_num, pool, 0, False)
                result = optim.optimize()
                length_record[si, ti] = result.length

    benchmark_res = {"init_nums": bmu.INIT_NUMS,
                     "length_avg": [float(trials.mean()) for trials in length_record],
                     "length_std": [float(trials.std()) for trials in length_record]}
    bmu.plot_length_benchmark(benchmark_res)
    return benchmark_res

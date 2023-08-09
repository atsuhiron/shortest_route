from multiprocessing.pool import Pool

from tqdm import tqdm
import numpy as np

from algorithm.base_route_optimizer import SearchMode
from algorithm.two_opt_mp_optimizer import TwoOptMPOptimizer
import benchmarks.benchmark_util as bmu


def run_annealing_benchmark(k: float) -> dict[str, dict[str, list]]:
    point_num = 100
    trial_num = 40
    trunc = 5
    sample_size = len(bmu.INIT_NUMS) - trunc
    record_ann = np.zeros((sample_size, trial_num), dtype=np.float64)
    record_nor = np.zeros((sample_size, trial_num), dtype=np.float64)

    with Pool(10) as pool:
        arr = np.load(bmu.get_sample_name(point_num))
        for si in tqdm(range(sample_size)):
            init_num = bmu.INIT_NUMS[si]
            for ti in tqdm(range(trial_num), desc=f"init_num {bmu.INIT_NUMS[si]}", leave=False):
                optim = TwoOptMPOptimizer(arr, SearchMode.FREE, init_num, 10, pool, k, False)
                result = optim.optimize()
                record_ann[si][ti] = result.length

                optim = TwoOptMPOptimizer(arr, SearchMode.FREE, init_num, 10, pool, 0, False)
                result = optim.optimize()
                record_nor[si][ti] = result.length

    benchmark_res_nor = {"init_nums": bmu.INIT_NUMS[: -trunc],
                         "length_avg": [float(trials.mean()) for trials in record_nor],
                         "length_std": [float(trials.std()) for trials in record_nor]}
    benchmark_res_ann = {"init_nums": bmu.INIT_NUMS[: -trunc],
                         "length_avg": [float(trials.mean()) for trials in record_ann],
                         "length_std": [float(trials.std()) for trials in record_ann]}
    bmu.plot_length_benchmark(benchmark_res_nor, "normal", False)
    bmu.plot_length_benchmark(benchmark_res_ann, "annealing")
    return {"normal": benchmark_res_nor, "annealing": benchmark_res_ann}


if __name__ == "__main__":
    __spec__ = None
    run_annealing_benchmark(1.0)

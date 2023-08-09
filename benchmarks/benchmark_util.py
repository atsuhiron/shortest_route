import os
import shutil

import numpy as np
import matplotlib.pyplot as plt


SAMPLE_DIR = "sample_points"
POINT_NUMS = [20, 30, 40, 50, 60, 70, 80, 90, 100]
INIT_NUMS = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]


def get_sample_name(num: int) -> str:
    return f"{SAMPLE_DIR}/sample_{num}.npy"


def create_sample_arrays(dim: int = 2, override: bool = False):
    assert dim >= 2, "Dimension must be equal or larger than 2"

    already_exists = os.path.exists(SAMPLE_DIR) and os.path.isdir(SAMPLE_DIR)
    if override and already_exists:
        print(f"{SAMPLE_DIR} is exists. Overriding...")
        shutil.rmtree(SAMPLE_DIR)
        os.makedirs(SAMPLE_DIR)
    elif already_exists:
        print(f"{SAMPLE_DIR} is exists. Nothing is done.")
        return
    else:
        os.makedirs(SAMPLE_DIR)

    for point_num in POINT_NUMS:
        ra = np.random.random((point_num, dim)).astype(np.float32)
        np.save(get_sample_name(point_num), ra)


def plot_time_benchmark(result_dict: dict[str, list], label: str = None, show: bool = True):
    plt.errorbar(result_dict["points"], result_dict["time_avg"], result_dict["time_std"],
                 fmt="o", label=label, capsize=2)
    plt.xlabel("Number of point")
    plt.ylabel("time [s]")
    if show:
        plt.show()


def plot_length_benchmark(result_dict: dict[str, list], label: str = None, show: bool = True):
    plt.errorbar(result_dict["init_nums"], result_dict["length_avg"], result_dict["length_std"],
                 fmt="o", label=label, capsize=2)
    plt.xlabel("Number of initial num")
    plt.ylabel("length")
    plt.xscale("log")
    if show:
        if label is not None:
            plt.legend()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot(arr: np.ndarray):
    plt.plot([arr[0, 0]], [arr[0, 1]], "o", markersize=10, label="start")
    plt.plot([arr[-1, 0]], [arr[-1, 1]], "o", markersize=10, label="end")
    plt.plot(arr[:, 0], arr[:, 1], "o", ls="-")
    plt.legend()
    plt.show()
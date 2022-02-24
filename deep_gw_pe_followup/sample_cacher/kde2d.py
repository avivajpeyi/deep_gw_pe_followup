import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from deep_gw_pe_followup.plotting.hist2d import plot_probs

GRID_POINTS = 20j


def get_kde(x: np.ndarray, y: np.ndarray) -> stats.gaussian_kde:
    print("builing KDE")
    return stats.gaussian_kde(np.vstack([x, y]))


def evaluate_kde_on_grid(kde, x, y, num_gridpoints=GRID_POINTS):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    x_grid, y_grid = np.mgrid[xmin:xmax:num_gridpoints, ymin:ymax:num_gridpoints]
    xy_array = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(xy_array).T
    # z = z / np.sum(z)
    return xy_array[0], xy_array[1], z


def plot_kde(kde, xrange, yrange, xlabel, ylabel, fname):
    x, y, prob = evaluate_kde_on_grid(kde, xrange, yrange, num_gridpoints=100j)
    norm_factor = np.sum(prob)
    print(f"norm_factor: {norm_factor}")
    plot_probs(x, y, prob, xlabel=xlabel, ylabel=ylabel, fname=fname)
    return norm_factor


from typing import Dict


class KDE2D:
    def __init__(self, data: Dict):
        self.x_label = data.keys()[0]
        self.y_label = data.keys()[1]
        self.x = data[self.x_label]
        self.y = data[self.y_label]

    @property
    def normalize_factor(self):
        pass

    def plot(self):
        pass

    def eval(self, x, y):
        pass

    @classmethod
    def from_pickle(cls):
        pass

    def save_pickle(self, fname):
        pass

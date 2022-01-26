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
    return xy_array[0], xy_array[1], z



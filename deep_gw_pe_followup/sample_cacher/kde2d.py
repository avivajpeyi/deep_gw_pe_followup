import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from deep_gw_pe_followup.plotting.hist2d import plot_probs
import os

GRID_POINTS = 500j


def get_kde(x: np.ndarray, y: np.ndarray) -> stats.gaussian_kde:
    return stats.gaussian_kde(np.vstack([x, y]))


def evaluate_kde_on_grid(kde, x, y, num_gridpoints=GRID_POINTS):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    x_grid, y_grid = np.mgrid[xmin:xmax:num_gridpoints, ymin:ymax:num_gridpoints]
    xy_array = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(xy_array).T
    return x_grid, y_grid, z.reshape(x_grid.shape)


def plot_kde(kde, xrange, yrange, xlabel, ylabel, fname):
    x, y, prob = evaluate_kde_on_grid(kde, xrange, yrange, num_gridpoints=GRID_POINTS)
    norm_factor = np.sum(prob)
    print(f"norm_factor: {norm_factor}")
    plot_probs(x, y, prob, xlabel=xlabel, ylabel=ylabel, fname=fname)
    return norm_factor


def calc_and_save_kde_grid(fname, samples):
    kde = get_kde(samples.q, samples.xeff)
    qrange = [min(samples.q),max(samples.q)]
    xeffrange = [min(samples.xeff),max(samples.xeff)]
    x, y, z = evaluate_kde_on_grid(kde, x=qrange, y=xeffrange, num_gridpoints=GRID_POINTS)
    np.savez_compressed(fname, x=x, y=y, z=z)


def load_kde_gridpts(fname, clean=False, samples=None):
    if clean and os.path.isfile(fname):
        os.remove(fname)
    if not os.path.isfile(fname):
        calc_and_save_kde_grid(fname, samples)
    return np.load(fname)

import matplotlib.pyplot as plt
from matplotlib.colors import (to_rgba, ListedColormap)
import numpy as np
from scipy import stats
import os
import pandas as pd
from scipy.ndimage import gaussian_filter
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from scipy.interpolate import interp1d
import seaborn as sns
import numpy as np
from deep_gw_pe_followup.sample_cacher import evaluate_kde_on_grid, get_kde


OUT = "outkde"
N = 10000


def gauss_smooth(args, smooth):
    x, y, z = args
    z = gaussian_filter(z, smooth)
    return x, y, z


def add_contours(ax, cargs, levels, color, label_col=None, smooth=None):
    if smooth:
        cargs = gauss_smooth(cargs, smooth)
    CS = ax.contour(*cargs, levels=levels, linewidths=0.5, colors=color, alpha=0.7, zorder=100, )
    if label_col:
        ax.clabel(CS, CS.levels, inline=True, fmt=fmt_contour, fontsize=10, colors=label_col)


def make_colormap_to_white(color="tab:orange", alpha=1):
    color_rgb = np.array(to_rgba(color))
    # set lower part: 1 * 256/4 entries
    # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
    lower = np.ones((int(256 / 4), 4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the first color of the upper colormap
    for i in range(3):
        lower[:, i] = np.linspace(1, color_rgb[i], lower.shape[0])
    lower[:,-1] = alpha
    cmap = np.vstack(lower)
    return ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])



def fmt_contour(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"




# -

def plot_heatmap(x, y, z, ax, cmap, add_cbar=False, smooth=None):
    x = np.unique(x)
    y = np.unique(y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = z.reshape(len(x), len(y))
    args = (X, Y, Z)
    if smooth:
        args = gauss_smooth(args, smooth)
    cbar = ax.pcolor(*args, cmap=cmap, vmin=np.nanmin(z), vmax=np.nanmax(z), zorder=-100)
    if add_cbar:
        fig = ax.get_figure()
        cbar = fig.colorbar(cbar, ax=ax)
        return cbar
    return None


def find_closest_idx(arr, val):
    idx = np.abs(arr - val).argmin()
    return idx


def kde_predict(kde, query_x, query_y):
    return kde([query_x, query_y])[0]

def add_letter_marker(ax, x, y, color, letter):
    kwgs = dict(color=color, zorder=100)
    ax.scatter(x, y, marker=".", s=10, **kwgs)
    ax.annotate(letter, xy=(x, y), xycoords='data',
                xytext=(5, 0), textcoords='offset points', **kwgs)

def lab_to_rgb(*args):
    """Convert Lab color to sRGB, with components clipped to (0, 1)."""
    Lab = LabColor(*args)
    sRGB = convert_color(Lab, sRGBColor)
    return np.clip(sRGB.get_value_tuple(), 0, 1)


def get_cylon():
    L_samples = np.linspace(100, 0, 5)

    a_samples = (
        33.34664938,
        98.09940562,
        84.48361516,
        76.62970841,
        21.43276891)

    b_samples = (
        62.73345997,
        2.09003022,
        37.28252236,
        76.22507582,
        16.24862535)

    L = np.linspace(100, 0, 255)
    a = interp1d(L_samples, a_samples[::-1], 'cubic')(L)
    b = interp1d(L_samples, b_samples[::-1], 'cubic')(L)

    colors = [lab_to_rgb(Li, ai, bi) for Li, ai, bi in zip(L, a, b)]
    cmap = np.vstack(colors)
    return ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

# +


def format_cntr_data(x,y, levels, smooth=1.2, bins=75, range=[[0,1],[0,0.6]]):
    """
    STOLEN FROM CORNER
    """
    H, X, Y = np.histogram2d(
        x.flatten(),
        y.flatten(),
        bins=bins,
        range=range,
    )

    if H.sum() == 0:
        raise ValueError(
            "It looks like the provided 'range' is not valid "
            "or the sample is empty."
        )

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)


    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        raise ValueError("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ]
    )
    return X2, Y2, H2, V, H


# +

def add_cntr(ax, x,y, color, label="",  add_heat=200j, plot_median=False, label_contour_cols=[]):
    levels = [
        1 - np.exp(-0.5),     # 3-sig
    #    1 - np.exp(-2),      # 2-sig
        1 - np.exp(-9 / 2.0)  # 1-sig
    ]

    X2, Y2, H2, V, H = format_cntr_data(x,y, levels=levels, range=[[0,1],[-1,1]])
    cmap = sns.color_palette(color, as_cmap=True)
    c = cmap(levels[-1])

    CS = ax.contour(X2, Y2, H2.T, V, colors=[c] * len(levels), alpha=0.5, antialiased=True)

    if len(label_contour_cols):
        fmt = {}
        strs = [r'3-$\sigma$', r'1-$\sigma$']
        for l, s in zip(CS.levels, strs):
            fmt[l] = s
        ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=14, colors=label_contour_cols)



    if np.abs(add_heat)>0:
        cmap = make_colormap_to_white(c,alpha=1)
        plot_heatmap(*grid_data_with_kde(x,y, add_heat), ax, cmap, add_cbar=False)
    else:
        ax.contourf(
            X2,
            Y2,
            H2.T,
            [V.min(), H.max()],
            cmap=cmap,
            antialiased=False,
            alpha=0.1,
        )
        ax.contourf(
                X2,
                Y2,
                H2.T,
                [V.max(), H.max()],
                cmap=cmap,
                antialiased=False,
            alpha=0.5
            )

    if plot_median:
        ax.plot(np.median(x), np.median(y), color=c, zorder=100, label=label)
        ax.scatter(np.median(x), np.median(y), color=c, zorder=100)

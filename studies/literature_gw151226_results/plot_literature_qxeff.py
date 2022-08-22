# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: tess
#     language: python
#     name: tess
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import stats
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import seaborn as sns

SIGMA_LEVELS = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)]
SIGMA_LEVELS = [SIGMA_LEVELS[0], SIGMA_LEVELS[-1]] # 3 and 1-sigma levels


def load_qxeff_data(fn):
    data = np.load(fn)
    return pd.DataFrame(dict(q=data['q'],xeff=data['xeff']))



def format_cntr_data(x,y, levels, smooth=1.2, bins=75, range=[[0,1],[0,0.6]]):
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


def add_cntr(ax, x,y, color, label, levels=SIGMA_LEVELS):
    X2, Y2, H2, V, H = format_cntr_data(x,y, levels=levels)
    cmap = sns.color_palette(color, as_cmap=True)
    c = cmap(levels[-1])
    ax.contour(X2, Y2, H2.T, V, colors=[c] * len(levels), alpha=0.5, antialiased=True)
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
    ax.plot(np.median(x), np.median(y), color=c, zorder=100, label=label)
    ax.scatter(np.median(x), np.median(y), color=c, zorder=100)


def main():
    NITZ_QXEFF = load_qxeff_data("nitz_qxeff.npz")
    MATEU_QXEFF = load_qxeff_data("mateu_qxeff.npz")
    
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    add_cntr(ax, NITZ_QXEFF['q'].values, NITZ_QXEFF['xeff'].values, "Oranges", "Nitz+")
    add_cntr(ax, MATEU_QXEFF['q'].values, MATEU_QXEFF['xeff'].values, "Greens", "Mateu+")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\chi_{\rm eff}$")
    l = ax.legend()
    fig.savefig("literature_qxeff.png", dpi=500)
    
if __name__ == "__main__":
    main()



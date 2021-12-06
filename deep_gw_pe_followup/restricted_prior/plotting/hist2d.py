import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CMAP = "hot"


def plot_probs(x, y, p, xlabel, ylabel, plabel=None, fname=None):
    plt.close('all')




    fig, axes = plt.subplots(2, 1, figsize=(4, 8))
    ax = axes[0]

    try:
        if isinstance(p, pd.Series):
            z = p.values
        else:
            z = p.copy()
        z[z == -np.inf] = np.nan

        ax.tricontour(x, y, z, 15, linewidths=0.5, colors='k')
        cmap = ax.tricontourf(
            x, y, z, 15,
            vmin=np.nanmin(z), vmax=np.nanmax(z),
            # norm=plt.Normalize(vmax=abs(p).max(), vmin=-abs(p).max()),
            cmap=CMAP
        )
    except Exception:
        cmap = ax.scatter(x, y, c=p, cmap=CMAP)

    if plabel:
        ax.colorbar(cmap, label=plabel)

    plot_heatmap(axes[1])

    for ax in axes:
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    if fname:
        fig.tight_layout()
        fig.savefig(fname)
    else:
        return fig, axes


def plot_heatmap(x, y, p, ax):
    if isinstance(p, pd.Series):
        z = p.values
    else:
        z = p.copy()
    z[z == -np.inf] = np.nan

    x = np.unique(x.values)
    y = np.unique(y.values)
    X, Y = np.meshgrid(x, y, indexing='ij')

    Z = z.reshape(len(x), len(y))

    ax.pcolor(X, Y, Z, cmap=CMAP, vmin=np.nanmin(z), vmax=np.nanmax(z))

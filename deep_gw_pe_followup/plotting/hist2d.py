import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deep_gw_pe_followup import get_mpl_style

from matplotlib.colors import to_hex, to_rgba

sns.set_theme(style="ticks")
plt.style.use(get_mpl_style())

GREEN = "#70B375"
ORANGE = "#B37570"
PURPLE = "#7570B3"

SIGMA_LEVELS = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)]

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

    plot_heatmap(x, y, z, axes[1])

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
        x = x.values
        y = y.values
    else:
        z = p.copy()
    z[z == -np.inf] = np.nan

    x = np.unique(x)
    y = np.unique(y)
    X, Y = np.meshgrid(x, y, indexing='ij')

    Z = z.reshape(len(x), len(y))

    ax.pcolor(X, Y, Z, cmap=CMAP, vmin=np.nanmin(z), vmax=np.nanmax(z))


def get_alpha_colormap(hex_color, level=SIGMA_LEVELS):
    rbga = to_rgba(hex_color)
    return (to_hex((rbga[0], rbga[1], rbga[2], l), True) for l in level)


def seaborn_plot_hist(
        samps_list,
        params_list,
        cmaps_list,
        labels,
        markers,
        zorders,
        linestyles,
        params=dict(x="mass_ratio", y="chi_eff"),
):
    fig, ax = plt.subplots(figsize=(4, 4))
    kwargs = dict(
        **params,
        antialiased=True,
        linewidths=0.75,
        legend=False,
        levels=SIGMA_LEVELS,
        extend="max",
    )

    low_alpha = 0.7

    for cmap_str, samps, param, label, marker, z, linestyle in zip(cmaps_list, samps_list, params_list, labels, markers,
                                                                   zorders, linestyles):
        cmap = sns.color_palette(cmap_str, as_cmap=True)
        c = cmap(SIGMA_LEVELS[2])
        if len(samps) > 0:
            samps = samps.sample(10000, random_state=0)
            ax = sns.kdeplot(
                data=samps,
                cmap=cmap,
                **kwargs,
                ax=ax,
                zorder=z,
                fill=True,
                alpha=low_alpha,
            )
            ax = sns.kdeplot(
                data=samps,
                color=c,
                **kwargs,
                ax=ax,
                zorder=z,
                fill=False,
                linestyles=linestyle,
                alpha=low_alpha,
            )
        if len(param) > 0:
            ax.scatter(
                x=[param["mass_ratio"]],
                y=[param["chi_eff"]],
                zorder=1,
                color=c,
                marker=marker,
                label=label,
            )

    ax.set(xlabel=r"$q$", ylabel=r"$\chi_{\rm eff}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)
    ax.set_xticks([-0, 0.5, 1])
    ax.set_yticks([0.1, 0.3, 0.6])

    ax.legend(frameon=False, markerscale=2, fontsize=15)
    plt.minorticks_off()
    plt.tight_layout()

    return ax

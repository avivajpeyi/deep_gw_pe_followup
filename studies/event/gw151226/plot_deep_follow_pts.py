import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
from deep_gw_pe_followup import get_mpl_style
from deep_gw_pe_followup.plotting.hist2d import make_colormap_to_white
from GW151226_reader.data_reader import load_res
from matplotlib.colors import to_hex, to_rgba
from matplotlib.colors import (to_rgba, ListedColormap)
from bilby.gw.prior import CBCPriorDict
from deep_gw_pe_followup.restricted_prior.conversions import calc_xeff
from bilby.gw.conversion import generate_all_bbh_parameters

plt.style.use(get_mpl_style())

SIGMA_LEVELS = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)]
SIGMA_LEVELS = [SIGMA_LEVELS[0], SIGMA_LEVELS[1]]
ORANGE = "#eaa800"

N = 10000



def get_high_low_points():
    high = dict(mass_ratio=0.68, chi_eff=0.15)
    low = dict(mass_ratio=0.15, chi_eff=0.5)
    return high, low


def get_data():
    r = load_res()
    keys = ["chi_eff", "mass_ratio"]
    return (
        r["lvk_data"].result.posterior[keys],
        r["ias_data"].result.posterior[keys]
    )


def add_letter_marker(ax, x, y, color,  label='', letter=None):
    kwgs = dict(color=color, zorder=100)

    if letter not in "-":
        ax.scatter(x, y, marker='.', s=10, **kwgs)
        ax.scatter(100,100, marker=letter, s=10, **kwgs, label=label)
        ax.annotate(letter, xy=(x, y), xycoords='data',
                    xytext=(5, 0), textcoords='offset points', **kwgs)
    else:
        ax.scatter(x, y, marker=letter, s=10, **kwgs, label=label)


def get_bogus_samples():
    return pd.DataFrame(dict(
        mass_ratio=np.random.normal(loc=-100, size=N),
        chi_eff=np.random.normal(loc=-100, size=N),
    ))


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
        extend="max",
    )

    low_alpha = 0.7

    for cmap_str, samps, param, label, marker, z, linestyle in zip(cmaps_list, samps_list, params_list, labels, markers,
                                                                   zorders, linestyles):

        cmap = make_colormap_to_white(cmap_str)
        c = cmap(SIGMA_LEVELS[-1])
        if len(samps) > 0:
            samps = samps.sample(N, random_state=0)
            ax = sns.kdeplot(
                data=samps,
                cmap=cmap,
                levels=SIGMA_LEVELS,
                **kwargs,
                ax=ax,
                zorder=z,
                fill=True,
                alpha=low_alpha,
            )
            ax = sns.kdeplot(
                data=samps,
                color=c,
                levels=[SIGMA_LEVELS[0]],
                **kwargs,
                ax=ax,
                zorder=z,
                fill=False,
                linestyles=linestyle,
                alpha=low_alpha,
            )
        if len(param) > 0:
            add_letter_marker(ax, param["mass_ratio"], param["chi_eff"], c, label, marker)

    ax.set_xlabel(r"$q$", labelpad=-15, fontsize=25)
    ax.set_ylabel(r"$\chi_{\rm eff}$", labelpad=-15, fontsize=25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)
    ax.set_xticks([0.15, 0.68])
    ax.set_yticks([0.15,  0.5])
    ax.grid(False)

    leg = ax.legend(frameon=False, markerscale=3.5, fontsize=15)
    for i, line in enumerate(leg.get_lines()):
            line.set_linewidth(10)

    plt.minorticks_off()
    plt.tight_layout()

    return ax


def plot_comparison():
    out = "PLOTS"
    os.makedirs(out, exist_ok=True)

    high, low = get_high_low_points()
    lvk_samp, ias_samp = get_data()
    bogus_samples = get_bogus_samples()
    bogus_sample = dict(mass_ratio=-100, chi_eff=50)

    seaborn_plot_hist(
        samps_list=[ias_samp, lvk_samp, bogus_samples, bogus_samples],
        params_list=[bogus_sample, bogus_sample, high, low],
        cmaps_list=["tab:blue", ORANGE, "black", "black"],
        labels=["IAS", "LVK", "High-$q$", "Low-$q$", ],
        markers=["_", "_", "$\\rm{A}$", "$\\rm{B}$"],
        zorders=[-100, -100, -10, -10],
        linestyles=["solid", "solid", "dashed", "dashed"]
    )
    plt.savefig(f"{out}/high_low_on_lvk.png", transparent=True)

    print("DONE")


def main():
    plot_comparison()
    print("Complete!")


if __name__ == "__main__":
    main()

import pandas as pd
import pickle
from deep_gw_pe_followup.sample_cacher.kde2d import (
    get_kde,
    evaluate_kde_on_grid,
    plot_kde,
)
from deep_gw_pe_followup.plotting.hist2d import plot_probs, plot_heatmap

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
from deep_gw_pe_followup import get_mpl_style
from GW151226_reader.data_reader import load_res
from matplotlib.colors import to_hex, to_rgba

from bilby.gw.prior import CBCPriorDict
from deep_gw_pe_followup.restricted_prior.conversions import calc_xeff
from bilby.gw.conversion import generate_all_bbh_parameters
from deep_gw_pe_followup.plotting.hist2d import seaborn_plot_hist

TEST_VALS = dict(
    LOW_Q={"mass_ratio": 0.15, "chi_eff": 0.5},
    HIGH_Q={"mass_ratio": 0.68, "chi_eff": 0.15},
)


def load_lvk_q_xeff_samples() -> pd.DataFrame:
    res = load_res()
    return res["lvk_data"].result.posterior[["chi_eff", "mass_ratio"]]




def plot_kde_and_points(kde):
    x, y, prob = evaluate_kde_on_grid(kde, x=[0, 1], y=[0, 0.75], num_gridpoints=100j)
    prob = prob / np.sum(prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_heatmap(x, y, prob, ax)
    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

    ax.tricontour(x, y, prob, linewidths=0.5, colors='white', levels=[0.1, 0.39, 0.86, 0.98], inline=True, fmt=fmt, fontsize=10)
    # ax.scatter()
    plt.savefig("qxeff.png")


def main():
    s = load_lvk_q_xeff_samples()
    kde = get_kde(s.mass_ratio, s.chi_eff)
    norm_factor = plot_kde(
        kde,
        xrange=[0, 1],
        yrange=[0, 0.75],
        xlabel=r"$q$",
        ylabel=r"$\chi_{\rm eff}$",
        fname="lvk_kde.png",
    )
    for name, val in TEST_VALS.items():
        print(
            f"posterior prob for {name}: {kde([val['mass_ratio'], val['chi_eff']]) / norm_factor}"
        )
    low_q = kde([TEST_VALS['LOW_Q']['mass_ratio'], TEST_VALS['LOW_Q']['chi_eff']])
    high_q = kde([TEST_VALS['HIGH_Q']['mass_ratio'], TEST_VALS['HIGH_Q']['chi_eff']])
    print(high_q/low_q)

if __name__ == "__main__":
    # s = load_lvk_q_xeff_samples()
    # kde = get_kde(s.mass_ratio, s.chi_eff)
    # plot_kde_and_points(kde)
    main()

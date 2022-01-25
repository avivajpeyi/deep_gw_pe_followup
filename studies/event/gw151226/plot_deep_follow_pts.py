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

sns.set_theme(style="ticks")
plt.style.use(get_mpl_style())

GREEN = "#70B375"
ORANGE = "#B37570"
PURPLE = "#7570B3"

SIGMA_LEVELS = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)]


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

def get_max_l_data():
    res = load_res()
    for l, r in res.items():
        max_param = r.get_max_log_likelihood_param()
        print(l, dict(q=max_param["mass_ratio"], xeff=max_param["chi_eff"]))

    keys = ["chi_eff", "mass_ratio"]
    lvk_samp = res["lvk_data"].result.posterior[keys]
    lvk_max_param = {
        k: v for k, v in res["lvk_data"].get_max_log_likelihood_param().items() if k in keys
    }
    ias_samp = res["ias_data"].result.posterior[keys]
    ias_max_param = {
        k: v for k, v in res["ias_data"].get_max_log_likelihood_param().items() if k in keys
    }
    return lvk_samp, lvk_max_param, ias_samp, ias_max_param


def plot_comparison():
    lvk_samp, lvk_max_param, ias_samp, ias_max_param = get_max_l_data()
    seaborn_plot_hist(
        samps_list=[ias_samp, lvk_samp],
        params_list=[ias_max_param, lvk_max_param],
        cmaps_list=["Greens", "Oranges"],
        labels=["IAS", "LVK"],
        markers=["+", "*"],
        zorders=[-100, -10],
        linestyles=["solid", "dashed"]
    )
    plt.savefig("maxL_pts_for_IAS_and_LVK.png")

    lvk_prior = CBCPriorDict(filename="priors/GW151226.prior")
    s = pd.DataFrame(lvk_prior.sample(100000))
    s['cos_tilt_1'] = np.cos(s['tilt_1'])
    s['cos_tilt_2'] = np.cos(s['tilt_2'])
    s['chi_eff'] = calc_xeff(a1=s.a_1, a2=s.a_2, cos1=s.cos_tilt_1, cos2=s.cos_tilt_2, q=s.mass_ratio)
    prior_samples = s[keys]
    seaborn_plot_hist(
        samps_list=[prior_samples, ias_samp, lvk_samp],
        params_list=[dict(mass_ratio=-100, chi_eff=50), ias_max_param, lvk_max_param],
        cmaps_list=["Blues", "Greens", "Oranges"],
        labels=["Prior", "IAS", "LVK"],
        markers=["_", "+", "*"],
        zorders=[-200, -100, -10],
        linestyles=["solid", "solid", "dashed"]
    )
    plt.savefig("maxL_pts_onn_prior.png")

    for label, r in res.items():
        param = generate_all_bbh_parameters(r.get_max_log_likelihood_param())
        param = {k: v for k, v in param.items() if k in lvk_prior}
        ln_prior = lvk_prior.ln_prob(param)
        print(label)
        print(f"Ln prior prob: {ln_prior} (num params: {len(param)}, {sorted(param)})")
        print(f"Ln log likelihood: {r.get_max_log_likelihood()}")

def main():
    plot_comparison()
    print("Complete!")


if __name__ == "__main__":
    main()

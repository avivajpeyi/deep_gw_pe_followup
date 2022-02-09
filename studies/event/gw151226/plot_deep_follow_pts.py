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


def get_high_low_points():
    high = dict(mass_ratio=0.68, chi_eff=0.15)
    low = dict(mass_ratio=0.15, chi_eff=0.5)
    return high, low

def get_max_l_data():
    res = load_res()
    for l, r in res.items():
        max_param = r.get_max_log_likelihood_param()
        lnl = r.get_max_log_likelihood()
        print(l, dict(q=max_param["mass_ratio"], xeff=max_param["chi_eff"], lnl=lnl))

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


def get_bogus_samples():
    return pd.DataFrame(dict(
        mass_ratio = np.random.normal(loc=-100, size=100000),
        chi_eff=np.random.normal(loc=-100, size=100000),
    ))

def plot_comparison():
    lvk_samp, lvk_max_param, ias_samp, ias_max_param = get_max_l_data()

    high, low = get_high_low_points()
    lvk_samp, lvk_max_param, ias_samp, ias_max_param = get_max_l_data()
    seaborn_plot_hist(
        samps_list=[ias_samp, get_bogus_samples() ],
        params_list=[high, low],
        cmaps_list=["Greens", "Greens"],
        labels=["High-q", "Low-q"],
        markers=["X", "s"],
        zorders=[-100, -10],
        linestyles=["solid", "dashed"]
    )
     plt.savefig("mode_pts_for_IAS.png")




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
    prior_samples = s[['mass_ratio', 'chi_eff']]
    seaborn_plot_hist(
        samps_list=[prior_samples, ias_samp, lvk_samp],
        params_list=[dict(mass_ratio=-100, chi_eff=50), ias_max_param, lvk_max_param],
        cmaps_list=["Blues", "Greens", "Oranges"],
        labels=["Prior", "IAS", "LVK"],
        markers=["_", "+", "*"],
        zorders=[-200, -100, -10],
        linestyles=["solid", "solid", "dashed"]
    )
    plt.savefig("maxL_pts_on_prior.png")

    

def main():
    plot_comparison()
    print("Complete!")


if __name__ == "__main__":
    main()

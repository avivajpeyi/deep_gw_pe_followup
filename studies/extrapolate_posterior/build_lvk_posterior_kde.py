import pandas as pd
import pickle
from deep_gw_pe_followup.sample_cacher.kde2d import (
    get_kde,
    evaluate_kde_on_grid,
    plot_kde,
)
from deep_gw_pe_followup.plotting.hist2d import plot_probs

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
    IAS={"mass_ratio": 0.10537432884262372, "chi_eff": 0.5899239152977372},
    LVK={"mass_ratio": 0.29103432408863716, "chi_eff": 0.2959265313232419},
    LOW_Q={"mass_ratio": 0.15, "chi_eff": 0.5},
    HIGH_Q={"mass_ratio": 0.68, "chi_eff": 0.15},
)


def load_lvk_q_xeff_samples() -> pd.DataFrame:
    res = load_res()
    return res["lvk_data"].result.posterior[["chi_eff", "mass_ratio"]]


def pickle_and_save_kde(kde_obj, pickle_fname):
    pass


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
            f"posterior prob for {name}: {kde(val['mass_ratio'], val['chi_eff'])/norm_factor}"
        )


if __name__ == "__main__":
    main()

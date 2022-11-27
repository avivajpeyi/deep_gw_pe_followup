from scipy import stats
import numpy as np
from tqdm.auto import tqdm

from uncertainties import ufloat


def standardise_params(pt):
    """Converts a point to the standardised parameter names"""
    if "mass_ratio" not in pt:
        pt["mass_ratio"] = pt["q"]
    if "chi_eff" not in pt:
        pt["chi_eff"] = pt["xeff"]
    if "q" not in pt:
        pt['q'] = pt["mass_ratio"]
    if 'xeff' not in pt:
        pt['xeff'] = pt["chi_eff"]
    return pt


def calc_kde_odds(q_xeff_posterior, pt1, pt2, samp_frac=0.7, repetitions=100):
    """Calculate the odds of pt1 vs pt2 given the posterior samples"""
    pt1 = standardise_params(pt1)
    pt2 = standardise_params(pt2)
    q_xeff_posterior = standardise_params(q_xeff_posterior)

    kde_odds = np.zeros(repetitions)

    for i in range(repetitions):
        # Sample from the posterior
        samp = q_xeff_posterior.sample(int(samp_frac * len(q_xeff_posterior)))
        kde = stats.gaussian_kde(np.vstack([samp.q, samp.xeff]))
        # calculate the odds
        kde_odds[i] = kde([pt1["q"], pt1["xeff"]]) / kde([pt2["q"], pt2["xeff"]])

    return ufloat(np.mean(kde_odds), np.std(kde_odds))

from scipy import stats
import numpy as np


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


def calc_kde_odds(q_xeff_posterior, pt1, pt2):
    """Calculate the odds of pt1 vs pt2 given the posterior samples"""
    pt1 = standardise_params(pt1)
    pt2 = standardise_params(pt2)
    q_xeff_posterior = standardise_params(q_xeff_posterior)

    kde = stats.gaussian_kde(np.vstack([q_xeff_posterior['q'], q_xeff_posterior['xeff']]))

    pt1_post = kde.evaluate([pt1["mass_ratio"], pt1["chi_eff"]])

    pt2_post = kde.evaluate([pt2["mass_ratio"], pt2["chi_eff"]])

    return pt1_post[0] / pt2_post[0]

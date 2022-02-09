from deep_gw_pe_followup.plotting import plot_probs, plot_corner
from deep_gw_pe_followup.sample_cacher import cacher

from bilby.core.prior import PriorDict, Uniform, Sine, Beta, TruncatedGaussian, PowerLaw, FromFile
from deep_gw_pe_followup.restricted_prior.conversions import calc_xeff
import pandas as pd
import numpy as np
from deep_gw_pe_followup.sample_cacher.kde2d import get_kde, evaluate_kde_on_grid

from effective_spins.computers import compute_and_store_prior_p_and_xeff

import os

SAMPLES_FILE = "qxeff_samples.h5"
ASTRO_SAMPLES_FILE = "astro_qxeff_samples.h5"
KDE_GRID_FILE = "kde_qxeff_grid.h5"
ASTRO_KDE_GRID_FILE = "kde_astro_qxeff_grid.h5"
NUMERICAL_FILE = f"{compute_and_store_prior_p_and_xeff.OUTDIR}/p_q_and_xeff.h5"


def get_dynamical_astro_prior():
    alpha = 2.58
    mu = 0.49
    var = 0.04
    zmin = 0.18
    chi_alpha = ((1 - mu) / var - 1 / mu) * (mu**2)
    chi_beta = chi_alpha * (1 / mu - 1)
    return PriorDict(dict(  # parameters  from shanika's results
        mass_ratio=PowerLaw(alpha=alpha, minimum=0, maximum=1),
        a_1=Beta(alpha=chi_alpha, beta=chi_beta),
        a_2=Beta(alpha=chi_alpha, beta=chi_beta),
        cos_tilt_1=Uniform(zmin, 1),
        cos_tilt_2=Uniform(zmin, 1),
    ))


def get_prior():
    return PriorDict(dict(
        mass_ratio=Uniform(name='mass_ratio', minimum=0.0556, maximum=1, latex_label='$q$', unit=None),
        a_1=Uniform(minimum=0, maximum=0.99, name='a_1', latex_label='$a_1$', unit=None),
        a_2=Uniform(minimum=0, maximum=0.99, name='a_2', latex_label='$a_2$', unit=None),
        tilt_1=Sine(name='tilt_1', latex_label='$\\theta_1$', unit=None, minimum=0, maximum=3.141592653589793),
        tilt_2=Sine(name='tilt_2', latex_label='$\\theta_2$', unit=None, minimum=0, maximum=3.141592653589793),
    ))


def draw_samples(prior, fname, n=int(1e7)):
    s = pd.DataFrame(prior.sample(n))
    if 'cos_tilt_1' not in s.columns.values:
        ct1, ct2 = np.cos(s.tilt_1), np.cos(s.tilt_2)
    else:
        ct1, ct2 = s.cos_tilt_1, s.cos_tilt_2
    plot_corner(s, fname=fname.replace(".h5", "_all.png"))
    q = s.mass_ratio
    xeff = calc_xeff(a1=s.a_1, a2=s.a_2, cos1=ct1, cos2=ct2, q=q)
    df = pd.DataFrame(dict(q=q, xeff=xeff))
    cacher.store_probabilities(df, fname, append=True, no_duplicates=False)
    plot_corner(df, fname=fname.replace(".h5", ".png"))
    return df


def build_kde_grid(df, fname):
    kde = get_kde(df.q, df.xeff)
    x, y, z = evaluate_kde_on_grid(kde, df.q, df.xeff, num_gridpoints=100j)
    kde_df = pd.DataFrame(dict(q=x, xeff=y, p=z))

    cacher.store_probabilities(kde_df, fname)
    plot_probs(kde_df.q.values, kde_df.xeff.values, p=kde_df.p, xlabel=r"$q$", ylabel=r"$\chi_{\rm eff}$",
               fname=fname.replace(".h5", ".png"))
    return kde_df


def main():
    if os.path.isfile(SAMPLES_FILE):
        samp_df = cacher.load_probabilities(SAMPLES_FILE)
    else:
        samp_df = draw_samples(get_prior(), SAMPLES_FILE, int(1e7))

    if os.path.isfile(KDE_GRID_FILE):
        kde_df = cacher.load_probabilities(KDE_GRID_FILE)
    else:
        kde_df = build_kde_grid(samp_df)
    norm_factor = np.sum(kde_df.p)
    print(f"norm factor: {norm_factor}")

    # if os.path.isfile(ASTRO_SAMPLES_FILE):
    #     samp_df = cacher.load_probabilities(ASTRO_SAMPLES_FILE)
    # else:
    #     samp_df = draw_samples(get_dynamical_astro_prior(), ASTRO_SAMPLES_FILE, int(1e7))

    # if os.path.isfile(ASTRO_KDE_GRID_FILE):
    #     kde_df = cacher.load_probabilities(ASTRO_KDE_GRID_FILE)
    # else:
    #     kde_df = build_kde_grid(samp_df, ASTRO_KDE_GRID_FILE)

    if os.path.isfile(NUMERICAL_FILE):
        numerical_df = cacher.load_probabilities(NUMERICAL_FILE)
    else:
        compute_and_store_prior_p_and_xeff.generate_dataset('q')
        numerical_df = cacher.load_probabilities(NUMERICAL_FILE)
    norm_factor = np.sum(numerical_df.p)
    print(f"norm factor: {norm_factor}")


if __name__ == '__main__':
    main()

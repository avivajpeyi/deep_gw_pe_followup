"""
to get pi(m1|old)
generate U[Mc], approx Delta[q] --> calculate m1 --> m1prior -> KDE(m1)
"""

from bilby.core.prior import Uniform, Interped, PriorDict
from bilby.gw.conversion import (
    chirp_mass_and_mass_ratio_to_total_mass,
    total_mass_and_mass_ratio_to_component_masses
)

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

N = 1000000


def generate_mc_q_samples(mc_bounds, q_bounds, n_samples=N) -> pd.DataFrame:
    print(f"q prior width: {q_bounds[1] - q_bounds[0]}")
    prior = PriorDict(dict(
        chirp_mass=Uniform(minimum=mc_bounds[0], maximum=mc_bounds[1]),
        mass_ratio=Uniform(minimum=q_bounds[0], maximum=q_bounds[1]),
    ))
    return pd.DataFrame(prior.sample(n_samples))


def convert_mc_q_samples_to_m1m2(samples_df: pd.DataFrame) -> pd.DataFrame:
    mc, q = samples_df.chirp_mass.values, samples_df.mass_ratio.values
    mtot = chirp_mass_and_mass_ratio_to_total_mass(chirp_mass=mc, mass_ratio=q)
    m1, m2 = total_mass_and_mass_ratio_to_component_masses(mass_ratio=q, total_mass=mtot)
    return pd.DataFrame(dict(mass_1=m1, mass_2=m2))


def build_interped_prior(s, name):
    s_space = np.linspace(min(s), max(s), num=1000)
    density, bins = np.histogram(s, s_space, density=True)
    y = savgol_filter(density,  window_length=51, polyorder=2)
    x = (bins[1:] + bins[:-1]) / 2
    assert len(x) == len(y)
    return Interped(xx=x, yy=y, name=name)


def get_m1m2_prior_from_mcq_prior(mc_bounds, q_bounds):
    mcq_df = generate_mc_q_samples(mc_bounds, q_bounds)
    m1m2_df = convert_mc_q_samples_to_m1m2(mcq_df)
    m1_prior = build_interped_prior(s=m1m2_df.mass_1, name='mass_1')
    m2_prior = build_interped_prior(s=m1m2_df.mass_2, name='mass_2')
    return m1_prior, m2_prior

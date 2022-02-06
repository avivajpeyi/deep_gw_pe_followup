"""
to get pi(m1|new)
generate U[m1, m2] --> cut q=q0 -->  KDE(m1)
"""

from bilby.core.prior import Uniform, Interped, PriorDict
from bilby.gw.conversion import (
    component_masses_to_mass_ratio
)

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

N = 1000000


def generate_m1m2_samples(m1_bounds, m2_bounds, n_samples=N) -> pd.DataFrame:
    prior = PriorDict(dict(
        m1=Uniform(minimum=m1_bounds[0], maximum=m1_bounds[1]),
        m2=Uniform(minimum=m2_bounds[0], maximum=m2_bounds[1]),
    ))
    return pd.DataFrame(prior.sample(n_samples))


def add_q(samples_df: pd.DataFrame) -> pd.DataFrame:
    samples_df['q'] = component_masses_to_mass_ratio(
        mass_1=samples_df.m1.values,
        mass_2=samples_df.m2.values,
    )
    return samples_df


def cut_q_samples(s, q_bounds):
    orig_len = len(s)
    s = s[s['q'] >= q_bounds[0]]
    s = s[s['q'] <= q_bounds[1]]
    final_len = len(s)
    percent_dropped = 100 * (orig_len-final_len)/orig_len
    print(f"Samples cut {orig_len}-->{final_len} ({percent_dropped:.1f}% dropped)")
    return s, percent_dropped


def build_interped_prior(s, name):
    s_space = np.linspace(min(s), max(s), num=1000)
    density, bins = np.histogram(s, s_space, density=True)
    y = savgol_filter(density, window_length=51, polyorder=2)
    x = (bins[1:] + bins[:-1]) / 2
    assert len(x) == len(y)
    return Interped(xx=x, yy=y, name=name)


def get_m1m2_prior_after_q_cut(m1_bounds, m2_bounds, q_bounds):
    m1m2_df = generate_m1m2_samples(m1_bounds, m2_bounds)
    df = add_q(m1m2_df)
    df, percent_dropped = cut_q_samples(df, q_bounds)
    m1_prior = build_interped_prior(s=m1m2_df.m1, name='mass_1')
    m2_prior = build_interped_prior(s=m1m2_df.m2, name='mass_2')
    return m1_prior, m2_prior, percent_dropped

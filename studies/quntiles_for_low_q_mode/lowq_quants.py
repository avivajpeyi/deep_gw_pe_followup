import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from bilby.gw.result import CBCResult
from GW151226_reader.data_reader import load_res

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
    p = r["ias_data"].result.posterior
    p = p[p['mass_ratio'] < 0.3]
    p = p[p['chi_eff'] > 0.45]
    p['tilt_1'] = np.rad2deg(np.arccos(p['cos_tilt_1']))
    return p


d = get_data()
r = CBCResult()
r.posterior = d

r.outdir = "."
r.parameter_labels = list(d.columns.values)
r.search_parameter_keys = list(d.columns.values)
r.parameter_labels_with_unit = list(d.columns.values)
r.plot_corner(parameters=['chi_eff', 'mass_ratio', 'a_1', 'tilt_1', 'chi_p'])
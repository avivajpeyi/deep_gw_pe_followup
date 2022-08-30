import numpy as np


def posterior_odds(prior_odds_z0_by_z1, ln_z0, ln_z0):
    ln_bf = ln_z0 - ln_z1

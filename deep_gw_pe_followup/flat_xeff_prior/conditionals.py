import numpy as np
from .conversions import calc_chi1z, calc_chi2z
from .conversions import get_s1z_from_limits, get_s2z, s1z_lim
from typing import List

from scipy.optimize import minimize



def condition_func_xdiff(reference_params, xeff, q):
    xdiff = np.linspace(-1, 1, 1000000)
    chi1z = calc_chi1z(q=q, xeff=xeff, xdiff=xdiff)
    chi2z = calc_chi2z(q=q, xeff=xeff, xdiff=xdiff)
    # chi1z(xdiff) == 1
    # xdiff == f
    valid = (chi1z ** 2 <= 1.) * (chi2z ** 2 <= 1.)  # z component cant be more than 1
    xdiff = xdiff[valid]
    if len(xdiff) < 10:
        print(f"Very narrow! {xeff, q} --> {xdiff}")
        return dict(minimum=np.nan, maximum=np.nan)
    return dict(minimum=min(xdiff), maximum=max(xdiff))


def condition_func_chi1pmagSqr(reference_params, xeff, q, xdiff):
    chi1z = calc_chi1z(q=q, xeff=xeff, xdiff=xdiff)
    return dict(alpha=1, minimum=0, maximum=1 - chi1z ** 2)


def condition_func_chi2pmagSqr(reference_params, xeff, q, xdiff):
    chi2z = calc_chi2z(q=q, xeff=xeff, xdiff=xdiff)
    return dict(alpha=1, minimum=0, maximum=1 - chi2z ** 2)


def condition_func_r1(reference_params, xeff, q, newxdiff):
    s1z_min, s1z_max = s1z_lim(xeff, q)
    s1z = get_s1z_from_limits(s1z_min, s1z_max, newxdiff)
    return dict(alpha=1, minimum=0, maximum=1 - s1z ** 2)


def condition_func_r2(reference_params, xeff, q, newxdiff):
    s1z_min, s1z_max = s1z_lim(xeff, q)
    s1z = get_s1z_from_limits(s1z_min, s1z_max, newxdiff)
    s2z = get_s2z(xeff, q, s1z)
    return dict(alpha=1, minimum=0, maximum=1 - s2z ** 2)



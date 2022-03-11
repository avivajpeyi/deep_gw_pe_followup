import numpy as np
from .conversions import calc_chi1z, calc_chi2z

@np.vectorize
def condition_func_xdiff(reference_params, xeff, q):
    xdiff = np.linspace(-1, 1, 10000)
    chi1z = calc_chi1z(q=q, xeff=xeff, xdiff=xdiff)
    chi2z = calc_chi2z(q=q, xeff=xeff, xdiff=xdiff)
    valid = (chi1z ** 2 <= 1.) * (chi2z ** 2 <= 1.)  # z component cant be more than 1
    xdiff = xdiff[valid == True]
    if len(xdiff) < 10:
        print("Very narrow!")
        return dict(minimum=np.nan, maximum=np.nan)
    return dict(minimum=min(xdiff), maximum=max(xdiff))


def condition_func_chi1pmagSqr(reference_params, xeff, q, xdiff):
    chi1z = calc_chi1z(q=q, xeff=xeff, xdiff=xdiff)
    return dict(minimum=0, maximum=1 - chi1z ** 2)


def condition_func_chi2pmagSqr(reference_params, xeff, q, xdiff):
    chi2z = calc_chi2z(q=q, xeff=xeff, xdiff=xdiff)
    return dict(minimum=0, maximum=1 - chi2z ** 2)


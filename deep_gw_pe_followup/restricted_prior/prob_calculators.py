import numpy as np
from numpy.random import uniform as unif

from .conversions import calc_a2


def zero_nonvalid_a2(p, a2):
    return np.where(((0 < a2) & (a2 < 1)), p, 0)


def jacobian(q, cos2):
    return np.abs((1 + q) / (q * cos2))  # analytically derived


def get_p_a1_given_xeff_q(a1, xeff, q, n=int(1e4)):
    cos1, cos2 = unif(-1, 1, n), unif(-1, 1, n)
    a2 = calc_a2(xeff=xeff, q=q, cos1=cos1, cos2=cos2, a1=a1)
    integrand = zero_nonvalid_a2(jacobian(q, cos2), a2)
    return np.mean(integrand)


def get_p_cos1_given_xeff_q_a1(cos1, a1, xeff, q, n=int(1e4)):
    cos2 = unif(-1, 1, n)
    a2 = calc_a2(xeff=xeff, q=q, cos1=cos1, cos2=cos2, a1=a1)
    integrand = zero_nonvalid_a2(jacobian(q, cos2), a2)
    return np.mean(integrand)


def get_p_cos2_given_xeff_q_a1_cos1(cos2, a1, xeff, q, cos1):
    a2 = calc_a2(xeff=xeff, q=q, cos1=cos1, cos2=cos2, a1=a1)
    integrand = zero_nonvalid_a2(jacobian(q, cos2), a2)
    return np.mean(integrand)

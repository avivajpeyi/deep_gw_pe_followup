import numpy as np


@np.vectorize
def calc_a2(xeff, q, cos1, cos2, a1):
    return (xeff * (1 + q) - a1 * cos1) / (q * cos2)


def q_factor(q):
    return ((3.0 + 4.0 * q) / (4.0 + 3.0 * q)) * q


def calc_xp_given_xeff(xeff, a1, a2, q, cos1, cos2, tan1, tan2):
    case1 = (xeff * (1 + q) - a2 * q * cos2) * tan1
    case2 = (xeff + q * xeff - a1 * cos1) * tan2 * q_factor(q)
    return np.maximum(case1, case2)


def calc_xp(a1, a2, q, sin1, sin2):
    case1 = a1 * sin1
    case2 = a2 * sin2 * q_factor(q)
    return np.maximum(case1, case2)

@np.vectorize
def calc_xeff(a1, a2, cos1, cos2, q):
    return ((a1 * cos1) + (q * a2 * cos2)) / (1.0 + q)

import numpy as np


@np.vectorize
def calc_xeff(chi1z, chi2z, q):
    xeff = (chi1z + (q * chi2z)) / (1.0 + q)
    if -1.0 <= xeff <= 1.0:
        return xeff
    else:
        return np.nan


@np.vectorize
def calc_xdiff(chi1z, chi2z, q):
    xdiff = (q * chi1z - chi2z) / (1 + q)
    if -1.0 <= xdiff <= 1.0:
        return xdiff
    else:
        return np.nan


@np.vectorize
def calc_chi1z(xeff, xdiff, q):
    chi1z = ((1 + q) * (xeff + xdiff * q)) / (1 + q ** 2)
    if chi1z ** 2 <= 1:
        return chi1z
    else:
        return chi1z  # np.nan


@np.vectorize
def calc_chi2z(xeff, xdiff, q):
    chi2z = ((1 + q) * (xeff * q - xdiff)) / (1 + q ** 2)
    if chi2z ** 2 <= 1:
        return chi2z
    else:
        return chi2z  # np.nan


@np.vectorize
def calc_chix(phi, xy_mag):
    chix = xy_mag * np.cos(phi)
    if chix ** 2 <= 1:
        return chix
    else:
        return np.nan


@np.vectorize
def calc_chiy(phi, xy_mag):
    chiy = xy_mag * np.sin(phi)
    if chiy ** 2 <= 1:
        return chiy
    else:
        return np.nan


@np.vectorize
def spin_cartesian_to_sphereical(sx, sy, sz):
    return (
        np.sqrt(sx ** 2 + sy ** 2 + sz ** 2),  # r
        np.arctan2(np.sqrt(sx ** 2 + sy ** 2) / sz),  # theta
        np.arctan2(sy / sx)  # phi
    )


@np.vectorize
def spin_sphereical_to_cartesian(a, theta, phi):
    return (
        a * np.sin(theta) * np.cos(phi),  # x
        a * np.sin(theta) * np.sin(phi),  # y
        a * np.cos(theta)  # z
    )


@np.vectorize
def sqr_sum(x, y, z):
    return x ** 2 + y ** 2 + z ** 2


def convert_xeff_xdiff_to_spins(parameters):
    p = parameters.copy()

    p['chi1z'] = calc_chi1z(q=p['q'], xeff=p['xeff'], xdiff=p['xdiff'])
    p['chi2z'] = calc_chi2z(q=p['q'], xeff=p['xeff'], xdiff=p['xdiff'])

    chi1p_mag = np.sqrt(p['chi1pmagSqr'])
    p['chi1x'] = calc_chix(phi=p['phi1'], xy_mag=chi1p_mag)
    p['chi1y'] = calc_chiy(phi=p['phi1'], xy_mag=chi1p_mag)

    chi2p_mag = np.sqrt(p['chi2pmagSqr'])
    p['chi2x'] = calc_chix(phi=p['phi2'], xy_mag=chi2p_mag)
    p['chi2y'] = calc_chiy(phi=p['phi2'], xy_mag=chi2p_mag)

    p['chi1mag'] = np.sqrt(sqr_sum(p['chi1x'], p['chi1y'], p['chi1z']))  # r1 <= 1
    p['chi2mag'] = np.sqrt(sqr_sum(p['chi2x'], p['chi2y'], p['chi2z']))  # r1 <= 1

    p['cos1'] = p['chi1z'] / p['chi1mag']
    p['cos2'] = p['chi2z'] / p['chi2mag']
    return p


def get_s1z(xeff, q, s2z):
    return (1 + q) * xeff - q * s2z


def get_s2z(xeff, q, s1z):
    return ((1 + q) * xeff - s1z) / q


def get_s1z_from_limits(s1z_min, s1z_max, newxdiff):
    return s1z_min + newxdiff * (s1z_max - s1z_min)


def s1z_lim(xeff, q):
    s1z_min = np.maximum(get_s1z(xeff, q, s2z=1), -1)
    s1z_max = np.minimum(get_s1z(xeff, q, s2z=-1), 1)
    return s1z_min, s1z_max



def transform_to_spins(parameters):
    xeff, q, newxdiff = parameters['xeff'], parameters['q'], parameters['newxdiff']
    p = parameters.copy()
    s1z_min, s1z_max = s1z_lim(xeff, q)
    p['s1z'] = get_s1z_from_limits(s1z_min, s1z_max, newxdiff)
    p['s2z'] = get_s2z(xeff, q, p['s1z'])

    p['R1'] = 1 - p['s1z'] ** 2
    p['R2'] = 1 - p['s2z'] ** 2

    p['s1x'] = calc_chix(phi=p['phi1'], xy_mag=p['r1'])
    p['s1y'] = calc_chiy(phi=p['phi1'], xy_mag=p['r1'])

    p['s2x'] = calc_chix(phi=p['phi2'], xy_mag=p['r2'])
    p['s2y'] = calc_chiy(phi=p['phi2'], xy_mag=p['r2'])

    p['a1'] = np.sqrt(sqr_sum(p['s1x'], p['s1y'], p['s1z']))  # r1 <= 1
    p['a2'] = np.sqrt(sqr_sum(p['s2x'], p['s2y'], p['s2z']))  # r1 <= 1

    p['cos1'] = p['s1z'] / p['a1']
    p['cos2'] = p['s2z'] / p['a2']
    return p

import os
import numpy as np
import unittest

from bilby.core.prior import PriorDict
from bilby.core.prior import Uniform, Constraint, ConditionalUniform, ConditionalPriorDict

from deep_gw_pe_followup.flat_xeff_prior.conversions import *
from deep_gw_pe_followup.flat_xeff_prior.conditionals import(
    condition_func_xdiff,
    condition_func_chi1pmagSqr,
    condition_func_chi2pmagSqr
)

import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(4)

CLEAN_AFTER = False

import corner
from deep_gw_pe_followup.plotting.corner import CORNER_KWARGS


class TestPrior(unittest.TestCase):

    def setUp(self) -> None:
        self.outdir = "./out_flat_xeff_prior"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.outdir) and CLEAN_AFTER:
            shutil.rmtree(self.outdir)

    def test_conversions(self):
        q = 0.5
        a1, theta1, phi1 = 0.8, 0, 0
        a2, theta2, phi2 = 0.8, np.pi / 2, 0

        chi1x, chi1y, chi1z = spin_sphereical_to_cartesian(a1, theta1, phi1)
        chi2x, chi2y, chi2z = spin_sphereical_to_cartesian(a2, theta2, phi2)
        xy1_mag = np.sqrt(chi1x ** 2 + chi1y ** 2)
        xy2_mag = np.sqrt(chi2x ** 2 + chi2y ** 2)

        xeff = calc_xeff(q=q, chi1z=chi1z, chi2z=chi2z)
        xdiff = calc_xdiff(q=q, chi1z=chi1z, chi2z=chi2z)

        chi1z_new = calc_chi1z(q=q, xeff=xeff, xdiff=xdiff)
        chi2z_new = calc_chi2z(q=q, xeff=xeff, xdiff=xdiff)
        assert chi1z_new - chi1z < 0.001
        assert chi2z_new - chi2z < 0.001

        chi1x_new = calc_chix(phi=phi1, xy_mag=xy1_mag)
        chi2x_new = calc_chix(phi=phi2, xy_mag=xy2_mag)
        assert chi1x_new - chi1x < 0.001
        assert chi2x_new - chi2x < 0.001

        chi1y_new = calc_chiy(phi=phi1, xy_mag=xy1_mag)
        chi2y_new = calc_chiy(phi=phi2, xy_mag=xy2_mag)
        assert chi1y_new - chi1y < 0.001
        assert chi2y_new - chi2y < 0.001

    def test_prior(self):
        d = get_flat_in_xeff_prior()
        s = pd.DataFrame(d.sample(1000))
        s = convert_xeff_xdiff_to_spins(s)[['xeff', 'xdiff', 'chi1mag', 'cos1']]
        s = s.dropna()
        labels = [l for l in s.columns.values]
        fig = corner.corner(s.values, labels=labels, **CORNER_KWARGS)
        fig.savefig(f"{self.outdir}/samples.png")



def get_flat_in_xeff_prior():
    return ConditionalPriorDict(
        dictionary=dict(
            xeff=Uniform(-1, 1),
            q=Uniform(0, 1),
            phi1=Uniform(0, 2 * np.pi),
            phi2=Uniform(0, 2 * np.pi),
            xdiff=ConditionalUniform(condition_func=condition_func_xdiff, minimum=-1, maximum=1),
            chi1pmagSqr=ConditionalUniform(condition_func=condition_func_chi1pmagSqr, minimum=0, maximum=1),
            chi2pmagSqr=ConditionalUniform(condition_func=condition_func_chi2pmagSqr, minimum=0, maximum=1),
        ),
        conversion_function=convert_xeff_xdiff_to_spins
    )



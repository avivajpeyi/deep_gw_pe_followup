import os
import time
import unittest

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from bilby.core.result import Result
import pandas as pd

from deep_gw_pe_followup.uniform_component_mass_prior_reweighter import (
    m1m2_prior_from_mcq_prior, m1m2_prior_with_q_cut, get_m1_weight
)
from deep_gw_pe_followup import get_mpl_style

CLEAN_AFTER = False


class TestPrior(unittest.TestCase):

    def setUp(self) -> None:
        self.q_tolerance = 0.0005
        self.outdir = f"./out_m1m2_prior_tol{self.q_tolerance}"
        os.makedirs(self.outdir, exist_ok=True)
        true_q = 0.10537432884262372
        # true_q = 0.29103432408863716
        self.qpri = Uniform(minimum=true_q-self.q_tolerance, maximum=true_q+self.q_tolerance)
        self.mcpri = Uniform(name='chirp_mass', minimum=6.0, maximum=15.0)
        self.m1pri = Uniform(minimum=3.022, maximum=54.398, latex_label=r"$m_1^{\rm orig}$", name='m1_orig')
        self.m2pri = Uniform(minimum=3.022, maximum=54.398, latex_label=r"$m_2^{\rm orig}$", name='m2_orig')

    def tearDown(self):
        import shutil
        if os.path.exists(self.outdir) and CLEAN_AFTER:
            shutil.rmtree(self.outdir)

    def test_m1_prior_from_mcq_prior(self):
        mc_prior = self.mcpri
        q_prior = self.qpri
        m1_prior, m2_prior = m1m2_prior_from_mcq_prior.get_m1m2_prior_from_mcq_prior(
            mc_bounds=[mc_prior.minimum, mc_prior.maximum],
            q_bounds=[q_prior.minimum, q_prior.maximum],
        )
        pdict = PriorDict(dict(mc=mc_prior, q=q_prior, m1=m1_prior, m2=m2_prior))
        samples = pd.DataFrame(pdict.sample(10000))
        r = Result(
            label='m1_from_mcq',
            outdir=self.outdir,
            search_parameter_keys=['mc', 'q', 'm1', 'm2'],
            priors=pdict,
            posterior=samples
        )
        fig = r.plot_corner(priors=True, save=False)
        fig.suptitle(f"mc-q (tol {self.q_tolerance}) prior --> m1 m2")
        fig.savefig(f"{self.outdir}/m1m2_from_mcq.png")


    def test_m1_prior_from_q_cut(self):
        m1_prior, m2_prior = self.m1pri, self.m2pri
        q_prior = self.qpri
        m1_prior_new, m2_prior_new, percent_dropped = m1m2_prior_with_q_cut.get_m1m2_prior_after_q_cut(
            m1_bounds=[m1_prior.minimum, m1_prior.maximum],
            m2_bounds=[m2_prior.minimum, m2_prior.maximum],
            q_bounds=[q_prior.minimum, q_prior.maximum],
        )
        pdict = PriorDict(dict(
            m1_orig=m1_prior, m2_orig=m2_prior, q=q_prior, m1=m1_prior_new, m2=m2_prior_new
        ))
        samples = pd.DataFrame(pdict.sample(10000))
        r = Result(
            label='m1_from_q_cut',
            outdir=self.outdir,
            search_parameter_keys=['m1_orig', 'm2_orig', 'q', 'm1', 'm2'],
            priors=pdict,
            posterior=samples
        )
        fig = r.plot_corner(priors=True, save=False)
        fig.suptitle(f"unif m1m2 --> q cut (tol {self.q_tolerance}) --> {int(percent_dropped)}% dropped --> m1m2 ")
        fig.savefig(f"{self.outdir}/m1m2_from_qcut.png")

    def test_plot_weights(self):

        mc_prior = self.mcpri
        q_prior = self.qpri
        m1_prior_old, _ = m1m2_prior_from_mcq_prior.get_m1m2_prior_from_mcq_prior(
            mc_bounds=[mc_prior.minimum, mc_prior.maximum],
            q_bounds=[q_prior.minimum, q_prior.maximum],
        )

        m1_prior, m2_prior = self.m1pri, self.m2pri
        q_prior = self.qpri
        m1_prior_new, _, _ = m1m2_prior_with_q_cut.get_m1m2_prior_after_q_cut(
            m1_bounds=[m1_prior.minimum, m1_prior.maximum],
            m2_bounds=[m2_prior.minimum, m2_prior.maximum],
            q_bounds=[q_prior.minimum, q_prior.maximum],
        )

        m1 = np.linspace(self.m1pri.minimum, self.m1pri.maximum, 1000)
        ln_weights = get_m1_weight.get_m1_ln_weights(m1, orig_pri=m1_prior_old, new_pri=m1_prior_new)

        print(ln_weights)

        get_mpl_style()
        plt.close('all')
        plt.plot(m1, np.exp(ln_weights), color="tab:blue")
        plt.fill_between(m1, ln_weights, min(ln_weights), alpha=0.4, color="tab:blue")
        plt.xlim(min(m1), max(m1))
        plt.xlabel(r"$m_1$")
        plt.yscale('log')
        # plt.ylabel(r"$\ln \pi(m_1| {\rm new}) - \ln \pi(m_1|{\rm old})$")
        plt.ylabel(r"Weights")
        plt.tight_layout()
        plt.savefig(f"{self.outdir}/weights.png")



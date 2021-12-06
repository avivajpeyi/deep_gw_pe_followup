import os
import time
import unittest

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import DeltaFunction, PriorDict, Uniform

from deep_gw_pe_followup.restricted_prior import prior
from deep_gw_pe_followup.restricted_prior.conversions import calc_a2, calc_xeff

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color="tab:orange",
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9.0 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)

CLEAN_AFTER = False


class TestPrior(unittest.TestCase):

    def setUp(self) -> None:
        self.outdir = "./out_prior"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.outdir) and CLEAN_AFTER:
            shutil.rmtree(self.outdir)

    def test_sample_from_prior(self):
        t0 = time.time()
        prior.RestrictedPrior(dictionary=dict(mass_ratio=0.2, chi_eff=0.1)).sample(100)
        duration = time.time() - t0
        self.assertLess(duration, 5)

    def test_bbh_prior_dict_conversion(self):
        p = PriorDict(filename="../studies/fast_injection/restricted.prior")
        r = prior.RestrictedPrior.from_bbh_priordict(p)
        self.assertIsInstance(r, prior.RestrictedPrior)

    def test_prior_prob_plot(self):
        p = prior.RestrictedPrior(filename="../studies/fast_injection/restricted.prior", clean=False)
        N = 5000

        t0 = time.time()
        samples = p.sample(N)
        print(f"Time: {(time.time() - t0) / N}")
        samples = pd.DataFrame(samples)
        kwargs = dict(q=samples.mass_ratio.values, xeff=samples.chi_eff, a1=samples.a_1, cos1=samples.cos_tilt_1, cos2=samples.cos_tilt_2)
        samples['a_2'] = calc_a2(**kwargs)

        print(samples.describe().T)
        df = samples.copy()
        df = get_valid_samples(df)
        invalid_df = df[df['a2_valid'] == False]

        plot_1d = False

        if plot_1d:
            if len(invalid_df) > 0:
                plot_samps = invalid_df.to_dict('records')
            else:
                plot_samps = df.sample(10).to_dict('records')

            for i, samp in enumerate(plot_samps):
                fig, axes = plt.subplots(3, 1, figsize=(3, 7))
                a1_pri = p.get_a1_prior()
                cos1_pri = p.get_cos1_prior(given_a1=samp['a_1'])
                cos2_pri = p.get_cos2_prior(given_cos1=samp['cos_tilt_1'], given_a1=samp['a_1'])
                axes[0].plot(a1_pri.xx, a1_pri.yy)
                axes[0].axvline(samp['a_1'], color="C1")
                axes[0].set_xlabel("a1")
                axes[0].set_xlim(0, 1)
                axes[1].plot(cos1_pri.xx, cos1_pri.yy)
                axes[1].axvline(samp['cos_tilt_1'], color="C1")
                axes[1].set_xlabel("cos1")
                axes[1].set_xlim(-1, 1)
                axes[2].plot(cos2_pri.xx, cos2_pri.yy)
                axes[2].axvline(samp['cos_tilt_2'], color="C1")
                axes[2].set_xlabel("cos2")
                axes[2].set_xlim(-1, 1)
                axes[0].set_title(f"prob={p.prob(samp):0.2f}")
                plt.tight_layout()
                fig.savefig(f'{self.outdir}/test{i:002}.png')

        t0 = time.time()
        rejection_samples = self.sample_uniform_dist(n=1e6, q=0.2, xeff=0.1).sample(N)
        print(f"Time: {(time.time() - t0) / 1e6}")

        labels = ['a1', 'cos1', 'cos2', 'a2', 'q', 'xeff']
        samples = samples[['a_1', 'cos_tilt_1', 'cos_tilt_2', 'a_2', 'mass_ratio', 'chi_eff']]
        range = [(0, 1), (-1, 1), (-1, 1), (0, 1), (0, 1), (-1, 1)]
        rejection_samples = rejection_samples[samples.columns.values]
        fig = corner.corner(samples, labels=labels, color="tab:blue", **CORNER_KWARGS, range=range)
        fig = corner.corner(rejection_samples, color="tab:orange", fig=fig, **CORNER_KWARGS, range=range)
        fig.savefig(os.path.join(self.outdir, "samples.png"))

    def sample_uniform_dist(self, n, q, xeff, xeff_tol=0.001):
        s = pd.DataFrame(PriorDict(dict(
            a_1=Uniform(0, 1),
            a_2=Uniform(0, 1),
            cos_tilt_1=Uniform(-1, 1),
            cos_tilt_2=Uniform(-1, 1),
            mass_ratio=DeltaFunction(q),
        )).sample(int(n)))
        s['chi_eff'] = calc_xeff(a1=s.a_1, a2=s.a_2, cos1=s.cos_tilt_1, cos2=s.cos_tilt_2, q=s.mass_ratio)
        s = s[np.abs(s['chi_eff'] - xeff) <= xeff_tol]
        return s

    def test_prior_from_file(self):
        r = prior.RestrictedPrior(filename="../studies/fast_injection/restricted.prior")
        s = pd.DataFrame(r.sample(1000))
        range = [RANGES.get(k, (min(s[k]), max(s[k]))) for k in s.keys()]
        labels = [k.replace("_", " ") for k in s.keys()]
        fig = corner.corner(s, color="tab:blue", **CORNER_KWARGS, range=range, labels=labels)
        fig.savefig(f"{self.outdir}/all_samp.png")



    def test_sample_from_unit_cube(self):
        """from bilby.core.base_sampler"""
        r = prior.RestrictedPrior(filename="../studies/fast_injection/restricted.prior")
        for i in range(100):
            unit = np.random.rand(r.ndim)
            theta = r.rescale(r.search_params, unit)
            params = {
                key: t for key, t in zip(r.search_params, theta)}
            ln_prob = r.ln_prob(params)

            if not np.isfinite(ln_prob):
                r.debug_sample(params, fname=f"{self.outdir}/unit_pri.png")
                self.fail(f"Unit: {unit},\n"
                          f"Theta: {theta},\n"
                          f"lnProb: {ln_prob}")


    def test_plot(self):
        param = {'a_1': 0.11555590351574924, 'cos_tilt_1': -0.6656216267247913, 'phi_12': 0.0380814382, 'phi_jl': 0.65953863,
         'chirp_mass': 0.349375531, 'luminosity_distance': 0.927999428, 'dec': 0.470610876, 'ra': 0.0280144439,
         'theta_jn': 0.952277741, 'psi': 0.578920808, 'phase': 0.860491949}
        r = prior.RestrictedPrior(filename="../studies/fast_injection/restricted.prior")
        r.debug_sample(param, f'{self.outdir}/param_test.png')

    @staticmethod
    def plot_cdf(cos2_pri):
        plt.close()
        plt.plot(cos2_pri.cumulative_distribution.x, cos2_pri.cumulative_distribution.y)
        vals = cos2_pri.sample(50)
        for v in vals:
            plt.axvline(v, color="C1", alpha=0.2)
        plt.ylabel("CDF")
        plt.xlabel("cos2")
        plt.savefig("cdf_and_samps.png")


def get_valid_samples(df):
    df['a2_valid'] = (df['a_2'] >= 0) & (df['a_2'] <= 1)
    return df


RANGES = dict(
    a_1=(0, 1),
    cos_tilt_1=(-1, 1),
    cos_tilt_2=(-1, 1),
    a_2=(0, 1),
    q=(0, 1),
    xeff=(-11, 1),
    phi_12=(0, 2 * np.pi),
    phi_jl=(0, 2 * np.pi),
    chirp_mass=(25, 100),
    luminosity_distance=(1e2, 5e3),
    mass_ratio=(0,1),
    chi_eff=(-1,1),
)

if __name__ == '__main__':
    unittest.main()

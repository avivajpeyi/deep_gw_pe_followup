import unittest
import os
from deep_gw_pe_followup.restricted_prior import prior
import time
import corner
import pandas as pd
from bilby.core.prior import PriorDict, DeltaFunction, Uniform
from deep_gw_pe_followup.restricted_prior.conversions import calc_xeff
import numpy as np

import matplotlib.pyplot as plt


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
        prior.RestrictedPrior(q=0.2, xeff=0.1).sample(100)
        duration = time.time() - t0
        self.assertLess(duration, 5)

    def test_prior_prob_plot(self):
        p = prior.RestrictedPrior(q=0.2, xeff=0.1, clean=False)

        t0 = time.time()
        samples = p.sample(1000)
        print(f"Time: {(time.time()-t0)/1000}")
        samples = pd.DataFrame(samples)

        print(samples.describe().T)
        df = samples.copy()
        df = get_valid_samples(df)
        invalid_df = df[df['a2_valid'] == False]

        if len(invalid_df)>0:
            plot_samps = invalid_df.to_dict('records')
        else:
            plot_samps = df.sample(10).to_dict('records')

        for i, samp in enumerate(plot_samps):
            fig, axes = plt.subplots(3,1, figsize=(3,7))
            a1_pri = p.get_a1_prior()
            cos1_pri = p.get_cos1_prior(given_a1=samp['a1'])
            cos2_pri = p.get_cos2_prior(given_cos1=samp['cos1'], given_a1=samp['a1'])
            axes[0].plot(a1_pri.xx, a1_pri.yy)
            axes[0].axvline(samp['a1'],color="C1")
            axes[0].set_xlabel("a1")
            axes[0].set_xlim(0,1)
            axes[1].plot(cos1_pri.xx, cos1_pri.yy)
            axes[1].axvline(samp['cos1'],color="C1")
            axes[1].set_xlabel("cos1")
            axes[1].set_xlim(-1, 1)
            axes[2].plot(cos2_pri.xx, cos2_pri.yy)
            axes[2].axvline(samp['cos2'],color="C1")
            axes[2].set_xlabel("cos2")
            axes[2].set_xlim(-1, 1)
            axes[0].set_title(f"prob={p.prob(samp):0.2f}")
            plt.tight_layout()
            fig.savefig(f'{self.outdir}/test{i:002}.png')


        t0 = time.time()
        rejection_samples = self.sample_uniform_dist(n=1e6, q=0.2, xeff=0.1).sample(1000)
        print(f"Time: {(time.time() - t0)/1e6}")

        rejection_samples = rejection_samples[samples.columns.values]

        print(rejection_samples.describe().T)

        fig = corner.corner(samples, labels=samples.columns.values, color="tab:blue", **CORNER_KWARGS)
        fig = corner.corner(rejection_samples, color="tab:orange", fig=fig, **CORNER_KWARGS)
        fig.savefig(os.path.join(self.outdir, "samples.png"))

    def sample_uniform_dist(self, n, q, xeff, xeff_tol=0.01):
        s = pd.DataFrame(PriorDict(dict(
            a1=Uniform(0, 1),
            a2=Uniform(0, 1),
            cos1=Uniform(-1, 1),
            cos2=Uniform(-1, 1),
            q=DeltaFunction(q),
        )).sample(int(n)))
        s['xeff'] = calc_xeff(**s.to_dict('list'))
        s = s[np.abs(s['xeff'] - xeff) <= xeff_tol]
        return s

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
    df['a2_valid'] = (df['a2'] >= 0) & (df['a2'] <= 1)
    return df


if __name__ == '__main__':
    unittest.main()

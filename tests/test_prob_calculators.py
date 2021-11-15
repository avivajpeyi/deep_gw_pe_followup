import multiprocessing
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import PriorDict, Uniform, DeltaFunction
from joblib import Parallel, delayed
from tqdm import tqdm



from deep_gw_pe_followup.restricted_prior.cacher import store_probabilities, plot_probs, load_probabilities
from deep_gw_pe_followup.restricted_prior.conversions import calc_xeff
from deep_gw_pe_followup.restricted_prior.prob_calculators import (get_p_a1_given_xeff_q, get_p_cos1_given_xeff_q_a1,
                                                                   get_p_cos2_given_xeff_q_a1_cos1)

from deep_gw_pe_followup.restricted_prior.prior import RestrictedPrior

num_cores = multiprocessing.cpu_count()


plt.style.use(
    'https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/55fab35b1c811dc3c0fdc6d169d67607e2c3edfc/publication.mplstyle')

CLEAN_AFTER = False


class TestProbCalculators(unittest.TestCase):

    def setUp(self):
        self.outdir = "./out_restricted_prior"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.outdir) and CLEAN_AFTER:
            shutil.rmtree(self.outdir)

    def test_pa1(self):
        self.p_a1_computer(xeff=0.3, q=0.9, fname=f'{self.outdir}/pa1_test1.png')
        self.p_a1_computer(xeff=-0.4, q=0.8, fname=f'{self.outdir}/pa1_test2.png')


    def test_pcos1_given_a1_xeff_q(self):
        self.p_cos1_a1_computer(xeff=0.3, q=0.9, fname=f'{self.outdir}/pcos1a1_test1.png')


    def test_pcos2_given_cos1_a1_xeff_q(self):
        self.p_cos2_computer(xeff=0.3,q=0.9, cos1=0.4, a1=0.2,fname=f'{self.outdir}/pcos2_test1.png')


        
    def p_cos2_computer(self, xeff, q, cos1, a1, fname='pcos2_test.png'):
        dc2 = 0.01
        cos2s= np.arange(-1,1,dc2)
        p_c2 = np.array([get_p_cos2_given_xeff_q_a1_cos1(xeff=xeff,q=q, a1=a1,cos1=cos1,cos2=cos2) for cos2 in cos2s])
        p_c2 = p_c2 / np.sum(p_c2) / dc2
        plt.plot(cos2s, p_c2)
        plt.xlabel('cos2')
        plt.ylabel('p(cos2)')
        plt.title(f"xeff={xeff},q={q}, a1={a1},cos1={cos1}")
        plt.savefig(fname)



    def p_cos1_a1_computer(self,xeff=0.3, q=0.9, fname='pa1_test.png'):
        mc_integral_n =int(5e4)
        dcos1, da1 = 0.01, 0.005
        a1s = np.arange(0, 1, da1)
        cos1s = np.arange(-1, 1, dcos1)

        samples_fname =  f"{self.outdir}/cached_pcos1a1.h5"
        if os.path.isfile(samples_fname):
            data = load_probabilities(samples_fname)
        else:
            data = dict(a1=np.array([]), cos1=np.array([]), p=np.array([]))
            for a1 in tqdm(a1s):
                p_cos1_for_a1 = Parallel(n_jobs=num_cores)(delayed(get_p_cos1_given_xeff_q_a1)(cos1, a1, xeff, q, mc_integral_n) for cos1 in cos1s)
                data['a1'] = np.append(data['a1'], np.array([a1 for _ in cos1s]))
                data['cos1'] = np.append(data['cos1'], cos1s)
                data['p'] = np.append(data['p'], p_cos1_for_a1)
            data = pd.DataFrame(data)
            store_probabilities(data, f"{self.outdir}/cached_pcos1a1.h5")

        s = self.sample_uniform_dist(int(1e6), q, xeff)
        fig, axes = plot_probs(data.a1, data.cos1, data.p, plabel='p', xlabel='a1', ylabel='cos1', fname=fname)
        for ax in axes:
            ax.plot(s.a1, s.cos1, ',w')
            ax.set_xlim(0,1)
            ax.set_ylim(-1, 1)
        fig.tight_layout()
        fig.savefig(fname)
        fig, axes = plot_probs(data.a1, data.cos1, np.log(data.p), plabel='lnp', xlabel='a1', ylabel='cos1', fname=fname.replace(".png", "_lnp.png"))
        for ax in axes:
            ax.plot(s.a1, s.cos1, ',w')
            ax.set_xlim(0,1)
            ax.set_ylim(-1, 1)
        fig.tight_layout()
        fig.savefig(fname.replace(".png", "_lnp.png"))



    def p_a1_computer(self, xeff=0.3, q=0.9, fname='pa1_test.png', n=int(1e5)):

        mc_integral_n =int(5e4)
        dcos1, da1 = 0.01, 0.005
        a1s = np.arange(0, 1, da1)
        cos1s = np.arange(-1, 1, dcos1)

        p = RestrictedPrior(q=q, xeff=xeff, build_cache=False).get_a1_prior()
        p_a1 = p.prob(a1s)



        # rejection-sampled distribution
        data = dict(x=np.array([]), y=np.array([]))
        for i in range(10):
            tolerance = 0.01
            s = self.sample_uniform_dist(n,q)
            s = s[np.abs(s['xeff'] - xeff) <= tolerance]
            p_a1_rej, bins = np.histogram(s.a1, bins=a1s, density=True)
            p_a1_rej = p_a1_rej / np.sum(p_a1_rej) / da1
            a1_cents = 0.5 * (bins[1:] + bins[:-1])
            data['x'] = np.append(data['x'],a1_cents)
            data['y'] = np.append(data['y'], p_a1_rej)

        fig, ax = plt.subplots(1, 1)

        plot_ci(data['x'], data['y'], ax, label="Rejection Sampling", alpha=0.3)
        ax.plot(a1s, p_a1, label="Analytical", color='tab:blue')
        ax.set_xlabel('$a_1$')
        ax.set_ylabel(r'$p(a_1|\chi_{\rm eff},q)$')
        ax.set_title(r"$\chi_{\rm eff} = " + str(xeff) + ", q=" + str(q) + "$")
        ax.set_xlim(0, 1)
        ax.legend()
        plt.tight_layout()
        fig.savefig(fname)


    def sample_uniform_dist(self, n, q, xeff, xeff_tol=0.01):
        s = pd.DataFrame(PriorDict(dict(
            a1=Uniform(0, 1),
            a2=Uniform(0, 1),
            cos1=Uniform(-1, 1),
            cos2=Uniform(-1, 1),
            q=DeltaFunction(q),
        )).sample(n))
        s['xeff'] = calc_xeff(**s.to_dict('list'))
        s = s[np.abs(s['xeff'] - xeff) <= xeff_tol]
        return s



def plot_ci(x, y, ax, label='', alpha=0.7, zorder=-10):
    cols = ['#EE7550', '#F19463', '#F6B176']
    ci = bin_by(x, y)
    # plot the 3rd stdv
    ax.fill_between(ci.x, ci['5th'], ci['95th'], alpha=alpha, color=cols[2] ,zorder=zorder-3)
    ax.fill_between(ci.x, ci['10th'], ci['90th'], alpha=alpha, color=cols[1] ,zorder=zorder-2)
    ax.fill_between(ci.x, ci['25th'], ci['75th'], alpha=alpha, color=cols[0] ,zorder=zorder-1)
    # plt the line
    ax.plot(ci.x, ci['median'], color=cols[0], label=label, zorder=zorder)


def bin_by(x, y, nbins=30, bins=None):
    """
    Divide the x axis into sections and return groups of y based on its x value
    """
    if bins is None:
        bins = np.linspace(x.min(), x.max(), nbins)

    bin_space = (bins[-1] - bins[0]) / (len(bins) - 1) / 2

    indicies = np.digitize(x, bins + bin_space)

    output = []
    for i in range(0, len(bins)):
        output.append(y[indicies == i])
    #
    # prepare a dataframe with cols: median; mean; 1up, 1dn, 2up, 2dn, 3up, 3dn
    df_names = ['mean', 'median', '5th', '95th', '10th', '90th', '25th', '75th']
    df = pd.DataFrame(columns=df_names)
    to_delete = []
    # for each bin, determine the std ranges
    for y_set in output:
        if y_set.size > 0:
            av = y_set.mean()
            intervals = np.percentile(y_set, q=[50, 5, 95, 10, 90, 25, 75])
            res = [av] + list(intervals)
            df = df.append(pd.DataFrame([res], columns=df_names))
        else:
            # just in case there are no elements in the bin
            to_delete.append(len(df) + 1 + len(to_delete))

    # add x values
    bins = np.delete(bins, to_delete)
    df['x'] = bins

    return df


if __name__ == '__main__':
    unittest.main()

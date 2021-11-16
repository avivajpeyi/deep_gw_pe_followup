import bilby
from bilby.core.prior import Prior, PriorDict, Interped
import shutil
from bilby.core.prior import Uniform
from bilby.gw.prior import BBHPriorDict
from .prob_calculators import get_p_a1_given_xeff_q, get_p_cos2_given_xeff_q_a1_cos1, get_p_cos1_given_xeff_q_a1
from .conversions import calc_a2
from .cacher import load_probabilities, store_probabilities

import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import os

import functools

from tqdm.auto import tqdm

import numpy as np

num_cores = multiprocessing.cpu_count()

D = dict(a1=0.01, cos1=0.01, cos2=0.01)
X = dict(
    a1=np.linspace(0, 1, int(1. / D['a1'])),
    cos1=np.linspace(-1, 1, int(1. / D['cos1'])),
    cos2=np.linspace(-1, 1, int(1. / D['cos2']))
)
MCMC_N = int(5e4)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_boundary_idx(x):
    """finds idx where data is non zero (assumes that there wont be gaps)"""
    non_z = np.nonzero(x)[0]
    return non_z[0], non_z[-1]

def find_boundary(x, y):
    b1, b2 = find_boundary_idx(y)
    vals = [x[b1], x[b2]]
    return min(vals), max(vals)

class RestrictedPrior:
    def __init__(self, q=1, xeff=0, clean=False, build_cache=True, mcmc_n=MCMC_N):
        self.q = q
        self.xeff = xeff
        self.outdir = f"cache_q{q}-xeff{xeff}".replace(".", "_")
        self.mcmc_n = mcmc_n
        if clean:
            shutil.rmtree(self.outdir)
        os.makedirs(self.outdir, exist_ok=True)

        # build cache
        self.a1_prior = self.get_a1_prior()
        if build_cache:
            self.cos1_prior = self.get_cos1_prior(given_a1=1)

    def get_a1_prior(self):
        fname = os.path.join(self.outdir, "a1_given_qxeff.h5")
        if os.path.exists(fname):
            data = load_probabilities(fname)
        else:
            a1s, da1 = X['a1'], D['a1']
            # p_a1 = np.array([get_p_a1_given_xeff_q(a1=a1, xeff=self.xeff, q=self.q, n=self.mcmc_n*10) for a1 in tqdm(a1s, desc="Building a1 cache")])

            p_a1 = Parallel(n_jobs=num_cores)(
                delayed(get_p_a1_given_xeff_q)(a1, self.xeff, self.q, self.mcmc_n*100)
                for a1 in tqdm(a1s, desc="Building a1 cache"))

            p_a1 = p_a1 / np.sum(p_a1) / da1
            data = pd.DataFrame(dict(a1=a1s, p_a1=p_a1))
            store_probabilities(data, fname)

        a1 = data.a1.values
        p_a1 = data.p_a1.values

        min_b, max_b = find_boundary(a1, p_a1)

        return Interped(xx=a1, yy=p_a1, minimum=min_b, maximum=max_b, name="a_1", latex_label=r"$a_1$")

    @functools.cached_property
    def cached_cos1_data(self):
        fname = os.path.join(self.outdir, "cos1_given_qxeffa1.h5")
        if os.path.isfile(fname):
            data = load_probabilities(fname)
        else:
            a1s, cos1s = X['a1'], X['cos1']

            data = dict(a1=np.array([]), cos1=np.array([]), p_cos1=np.array([]))
            for a1 in tqdm(a1s, desc="Building p_cos1 cache"):
                p_cos1_for_a1 = Parallel(n_jobs=num_cores)(
                    delayed(get_p_cos1_given_xeff_q_a1)(cos1, a1, self.xeff, self.q, self.mcmc_n) for cos1 in cos1s)
                data['a1'] = np.append(data['a1'], np.array([a1 for _ in cos1s]))
                data['cos1'] = np.append(data['cos1'], cos1s)
                data['p_cos1'] = np.append(data['p_cos1'], p_cos1_for_a1)
            data = pd.DataFrame(data)
            store_probabilities(data, fname)
        return data

    def get_cos1_prior(self, given_a1, ):
        data = self.cached_cos1_data
        closest_a1 = find_nearest(data.a1, given_a1)
        data = data[data.a1 == closest_a1]
        cos1 = data.cos1.values
        p_cos1 = data.p_cos1.values

        min_b, max_b = find_boundary(cos1, p_cos1)

        return Interped(
            xx=cos1, yy=p_cos1,
            minimum=min_b, maximum=max_b, name="cos_tilt_1",
            latex_label=r"$\cos \theta_1$"
        )

    def get_cos2_prior(self, given_cos1, given_a1):
        cos2s, dc2 = X['cos2'], D['cos2']

        args = (given_a1, self.xeff, self.q, given_cos1)
        p_c2 = np.array([get_p_cos2_given_xeff_q_a1_cos1(cos2, *args) for cos2 in cos2s])
        p_c2 = p_c2 / np.sum(p_c2) / dc2

        min_b, max_b = find_boundary(cos2s, p_c2)

        return Interped(
            xx=cos2s, yy=p_c2,
            minimum=min_b, maximum=max_b, name="cos_tilt_2",
            latex_label=r"$\cos \theta_2$"
        )

    def sample(self, size=1):
        kwgs = dict(q=self.q, xeff=self.xeff)
        a1 = self.a1_prior.sample(size)
        cos1 = np.hstack([self.get_cos1_prior(given_a1=i).sample(1) for i in tqdm(a1, desc="cos1 samples")])
        cos2 = np.hstack(
            [self.get_cos2_prior(given_a1=i, given_cos1=j).sample(1) for i, j in
             tqdm(zip(a1, cos1), desc="cos2 samples", total=len(a1))]
        )
        a2 = calc_a2(**kwgs, cos1=cos1, cos2=cos2, a1=a1)
        return dict(
            a1=a1,
            cos1=cos1,
            cos2=cos2,
            a2=a2,
        )

    def prob(self, sample):
        sample = sample.copy()
        if 'a1' in sample:
            sample['a_1'] = sample['a1']
        if 'cos1' in sample:
            sample['cos_tilt_1'] = sample['cos1']
        if 'cos2' in sample:
            sample['cos_tilt_2'] = sample['cos2']
        a1, c1, c2 = sample['a_1'], sample['cos_tilt_1'], sample['cos_tilt_2']
        p_a1 = self.a1_prior.prob(a1)
        p_c1 = self.get_cos1_prior(given_a1=a1).prob(c1)
        p_c2 = self.get_cos2_prior(given_a1=a1, given_cos1=c1).prob(c2)
        p_a2 = 1
        return p_a1 * p_c1 * p_c2 * p_a1

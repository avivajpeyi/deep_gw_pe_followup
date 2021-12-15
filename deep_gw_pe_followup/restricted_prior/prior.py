import multiprocessing
import os
import shutil

from joblib import Parallel, delayed
import multiprocessing

from cached_property import cached_property


import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import (Constraint, DeltaFunction,
                              Interped, Prior)
from bilby.gw.prior import (CBCPriorDict,
                            convert_to_lal_binary_black_hole_parameters,
                            fill_from_fixed_priors, generate_mass_parameters)
from tqdm.auto import tqdm

from .cacher import load_probabilities, store_probabilities
from .conversions import calc_a2
from .placeholder_prior import PlaceholderDelta
from .prob_calculators import (get_p_cos2_given_xeff_q_a1_cos1, get_p_a1_given_xeff_q, get_p_cos1_given_xeff_q_a1)
from .plotting.hist2d import plot_heatmap

import logging
from bilby.core.utils import logger

logger.setLevel(logging.INFO)

num_cores = multiprocessing.cpu_count()

D = dict(a1=0.01, cos1=0.01, cos2=0.01)
X = dict(
    a1=np.linspace(0, 1, int(1. / D['a1'])),
    cos1=np.linspace(-1, 1, int(2. / D['cos1'])),
    cos2=np.linspace(-1, 1, int(2. / D['cos2']))
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
    start, end = min(vals), max(vals)
    return start, end


class RestrictedPrior(CBCPriorDict):
    def __init__(self, dictionary=None, filename=None, clean=False, build_cache=True, mcmc_n=MCMC_N, cache=None):
        super().__init__(dictionary=dictionary, filename=filename, conversion_function=None)
        self.q = (self['q'] if 'q' in self else self['mass_ratio']).peak
        self.xeff = (self['xeff'] if 'xeff' in self else self['chi_eff']).peak
        self.mcmc_n = mcmc_n

        if clean:
            shutil.rmtree(self.cache)
        self.cache = cache

        self.search_params = self.get_search_params()
        self.ndim = len(self.search_params)

        # build cache
        if build_cache:
            self['a_1'] = self.get_a1_prior()
            self['cos_tilt_1'] = self.get_cos1_prior(given_a1=0.5)
            self['cos_tilt_2'] = self.get_cos2_prior(given_a1=0.5, given_cos1=0.5)

        self.assert_no_constraint_priors()  # we have forced the normalisation_constraint_factor = 1

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, c):
        if c is None:
            c = f"cache_q{self.q}-xeff{self.xeff}".replace(".", "_")
        if os.path.isdir(c):
            logger.debug(f"Loading RestricedPrior from cache {c}")
            self._cache = c
        else:
            logger.debug(f"Building RestricedPrior cache {c}")
            self._cache = c
            os.makedirs(self._cache, exist_ok=True)
        self._cache = os.path.abspath(self._cache)

    @property
    def restricted_params(self):
        return ['a_1', 'cos_tilt_1', 'cos_tilt_2', 'a_2']

    def assert_no_constraint_priors(self):
        for p in self:
            assert not isinstance(p, Constraint)

    def get_a1_prior(self):
        fname = os.path.join(self.cache, "a1_given_qxeff.h5")
        if os.path.exists(fname):
            data = load_probabilities(fname)
            logger.debug(f"Loaded {fname}")
        else:
            logger.debug(f"Creating {fname}")
            a1s = X['a1']
            da1 = a1s[1] - a1s[0]
            p_a1 = Parallel(n_jobs=num_cores, verbose=1)(
                delayed(get_p_a1_given_xeff_q)(a1, self.xeff, self.q, self.mcmc_n * 100)
                for a1 in tqdm(a1s, desc="Building a1 cache"))

            p_a1 = p_a1 / np.sum(p_a1) / da1
            data = pd.DataFrame(dict(a1=a1s, p_a1=p_a1))
            store_probabilities(data, fname)

        a1 = data.a1.values
        p_a1 = data.p_a1.values

        min_b, max_b = find_boundary(a1, p_a1)

        return Interped(xx=a1, yy=p_a1, minimum=min_b, maximum=max_b, name="a_1", latex_label=r"$a_1$")

    @cached_property
    def cached_cos1_data(self):
        fname = os.path.join(self.cache, "cos1_given_qxeffa1.h5")
        if os.path.isfile(fname):
            data = load_probabilities(fname)
            logger.debug(f"Loaded {fname}")
        else:
            logger.debug(f"Creating {fname}")
            a1s, cos1s = X['a1'], X['cos1']
            data = dict(a1=np.array([]), cos1=np.array([]), p_cos1=np.array([]))
            for a1 in tqdm(a1s, desc="Building p_cos1 cache"):
                p_cos1_for_a1 = Parallel(n_jobs=num_cores, verbose=1)(
                    delayed(get_p_cos1_given_xeff_q_a1)(cos1, a1, self.xeff, self.q, self.mcmc_n) for cos1 in cos1s)
                data['a1'] = np.append(data['a1'], np.array([a1 for _ in cos1s]))
                data['cos1'] = np.append(data['cos1'], cos1s)
                data['p_cos1'] = np.append(data['p_cos1'], p_cos1_for_a1)
            data = pd.DataFrame(data)
            store_probabilities(data, fname)
        return data

    @classmethod
    def from_bbh_priordict(cls, dict):
        dict = {k: v for k, v in dict.items()}
        return cls(dictionary=dict)

    def get_cos1_prior(self, given_a1, ):
        data = self.cached_cos1_data
        closest_a1 = find_nearest(data.a1, given_a1)
        data = data[data.a1 == closest_a1]
        cos1 = data.cos1.values
        p_cos1 = data.p_cos1.values

        try:
            min_b, max_b = find_boundary(cos1, p_cos1)
        except Exception:
            min_b = min(cos1)
            max_b = max(cos1)

        return Interped(
            xx=cos1, yy=p_cos1,
            minimum=min_b, maximum=max_b, name="cos_tilt_1",
            latex_label=r"$\cos \theta_1$"
        )

    def get_cos2_prior(self, given_a1, given_cos1):
        cos2s = X['cos2']
        dc2 = cos2s[1] - cos2s[0]

        args = (given_a1, self.xeff, self.q, given_cos1)
        p_cos2 = np.array([get_p_cos2_given_xeff_q_a1_cos1(cos2, *args) for cos2 in cos2s])
        p_cos2 = p_cos2 / np.sum(p_cos2) / dc2

        try:
            min_b, max_b = find_boundary(cos2s, p_cos2)
        except Exception:
            min_b = min(cos2s)
            max_b = max(cos2s)

        if min_b == max_b:
            return PlaceholderDelta(peak=min_b, name="cos_tilt_2", latex_label=r"$\cos \theta_2$")

        return Interped(
            xx=cos2s, yy=p_cos2,
            minimum=min_b, maximum=max_b, name="cos_tilt_2",
            latex_label=r"$\cos \theta_2$"
        )

    def sample_restricted(self, size=1):
        a1 = np.atleast_1d(self['a_1'].sample(size))
        if isinstance(size, int) and size > 10:
            cos1 = np.hstack([
                self.get_cos1_prior(i).sample(1)
                for i in tqdm(a1, desc="Getting cos1 samples", total=size)
            ])
            cos2 = np.hstack([
                self.get_cos2_prior(i, j).sample(1)
                for i, j in tqdm(zip(a1, cos1), desc="Getting cos2 samples", total=size)
            ])

        else:
            cos1 = np.hstack([self.get_cos1_prior(given_a1=i).sample(1) for i in a1])
            cos2 = np.hstack([
                self.get_cos2_prior(given_a1=i, given_cos1=j).sample(1)
                for i, j in zip(a1, cos1)
            ])
        return dict(
            a_1=a1,
            cos_tilt_1=cos1,
            cos_tilt_2=cos2,
        )

    def update_restricted_priors(self, sample):
        a1, c1 = sample['a_1'], sample['cos_tilt_1']
        self['cos_tilt_1'] = self.get_cos1_prior(given_a1=a1)
        self['cos_tilt_2'] = self.get_cos2_prior(given_a1=a1, given_cos1=c1)

    def get_search_params(self):
        """
        Go through the list of priors and add keys to the fixed and search
        parameter key list depending on whether
        the respective parameter is fixed.
        """
        search_parameter = list()
        for key in self:
            if isinstance(self[key], Prior) \
                    and self[key].is_fixed is False:
                search_parameter.append(key)
        return search_parameter

    # Overloaded functions

    def default_conversion_function(self, sample):
        if 'cos_tilt_1' in sample:
            sample['tilt_1'] = np.arccos(sample['cos_tilt_1'])
            sample['tilt_2'] = np.arccos(sample['cos_tilt_2'])
        sample['a_2'] = calc_a2(xeff=self.xeff, q=self.q, cos1=sample['cos_tilt_1'], cos2=sample['cos_tilt_2'],
                                a1=sample['a_1'])
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)
        return out_sample

    def sample_subset(self, keys=iter([]), size=None):
        samples = super().sample_subset(keys=keys, size=size)
        restricted_samples = self.sample_restricted(size)
        return {**samples, **restricted_samples}

    def prob(self, sample, **kwargs):
        self.update_restricted_priors(sample)
        return super().prob(sample)

    def ln_prob(self, sample, axis=None):
        self.update_restricted_priors(sample)
        self._prepare_evaluation(*zip(*sample.items()))
        res = {key: self[key].ln_prob(sample[key], **self.get_required_variables(key)) for key in sample}
        reslist = [v for v in res.values()]
        ln_prob = np.sum(reslist, axis=axis)
        return self.check_ln_prob(sample, ln_prob)

    def cdf(self, sample):
        self.update_restricted_priors(sample)
        return super().cdf(sample)

    def rescale_restricted(self, keys, theta):
        """theta:drawn from unit-cube"""
        unit = {k: t for k, t in zip(keys, theta)}
        scaled = {}
        scaled['a_1'] = self['a_1'].rescale(unit['a_1'])
        self['cos_tilt_1'] = self.get_cos1_prior(given_a1=scaled['a_1'])
        scaled['cos_tilt_1'] = self['cos_tilt_1'].rescale(unit['cos_tilt_1'])
        self['cos_tilt_2'] = self.get_cos2_prior(given_a1=scaled['a_1'], given_cos1=scaled['cos_tilt_1'])
        scaled['cos_tilt_2'] = self['cos_tilt_2'].rescale(unit['cos_tilt_2'])
        return scaled

    def rescale(self, keys, theta):
        """theta:drawn from unit-cube"""
        scaled = self.rescale_restricted(keys, theta)
        self.update_restricted_priors(scaled)
        return super().rescale(keys=keys, theta=theta)

    def normalize_constraint_factor(self, keys, min_accept=10000, sampling_chunk=50000, nrepeats=10):
        return 1.0

    def debug_sample(self, sample, fname='debug_prior.png'):
        self.update_restricted_priors(sample)
        nparam = len(self)
        fig, axes = plt.subplots(nparam, 1, figsize=(5, 2 * nparam))
        for i, label in enumerate(self):
            if not isinstance(self[label], DeltaFunction):
                xx = np.linspace(start=self[label].minimum, stop=self[label].maximum, num=100)
                yy = self[label].prob(xx)
                axes[i].plot(xx, yy, "C1")
                if label in sample:
                    axes[i].axvline(sample[label], c="C2", linestyle="--")
            else:
                axes[i].axvline(self[label].peak, c="C1")
                axes[i].axvline(self[label].peak, c="C2", linestyle="--")
            axes[i].set_xlabel(label)
        plt.tight_layout()
        plt.savefig(fname)

    def time_prior(self, n_evaluations=100):
        """ Times the prior evaluation and print an info message

        Parameters
        ==========
        n_evaluations: int
            The number of evaluations to estimate the evaluation time from

        """

        t1 = datetime.datetime.now()
        for _ in range(n_evaluations):
            theta = self.sample()
        total_time = (datetime.datetime.now() - t1).total_seconds()
        self._eval_time = total_time / n_evaluations

        if self._eval_time == 0:
            self._eval_time = np.nan
            logger.info("Unable to measure single prior sample time")
        else:
            logger.info("Single prior evaluation took {:.3e} s".format(self._eval_time))

    def plot_cache(self):
        """ Plot of a1 and 2D plot of a1-cos1 """
        fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
        axes[0].set_ylabel('p(a1)')
        axes[1].set_ylabel('cos1')
        axes[1].set_xlabel('a1')
        axes[0].plot(self['a_1'].xx, self['a_1'].yy)
        data = self.cached_cos1_data
        plot_heatmap(x=data['a1'], y=data['cos1'], p=data['p_cos1'], ax=axes[1])
        plt.tight_layout()
        plt.savefig(f"{self.cache}/plot.png")

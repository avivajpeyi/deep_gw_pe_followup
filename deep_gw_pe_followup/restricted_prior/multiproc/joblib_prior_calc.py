from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import numpy as np

num_cores = multiprocessing.cpu_count()

from ..prob_calculators import get_p_a1_given_xeff_q, get_p_cos1_given_xeff_q_a1


def joblib_get_p_a1(a1s, xeff, q, mcmc_n):
    p_a1 = Parallel(n_jobs=num_cores, verbose=1)(
        delayed(get_p_a1_given_xeff_q)(a1, xeff, q, mcmc_n * 100)
        for a1 in tqdm(a1s, desc="Building a1 cache")
    )
    return p_a1


def joblib_p_cos1_given_a1_calc(cos1s, a1s, xeff, q, mcmc_n):
    data = dict(a1=np.array([]), cos1=np.array([]), p_cos1=np.array([]))
    for a1 in tqdm(a1s, desc=f"Building p_cos1 cache ({num_cores} cores)"):
        p_cos1_for_a1 = joblib_get_p_cos1(cos1s, a1, xeff, q, mcmc_n)
        data['a1'] = np.append(data['a1'], np.array([a1 for _ in cos1s]))
        data['cos1'] = np.append(data['cos1'], cos1s)
        data['p_cos1'] = np.append(data['p_cos1'], p_cos1_for_a1)
    return data


def joblib_get_p_cos1(cos1s, a1, xeff, q, mcmc_n):
    p_cos1_for_a1 = Parallel(n_jobs=num_cores, verbose=1)(
        delayed(get_p_cos1_given_xeff_q_a1)(cos1, a1, xeff, q, mcmc_n) for cos1 in cos1s)
    return p_cos1_for_a1

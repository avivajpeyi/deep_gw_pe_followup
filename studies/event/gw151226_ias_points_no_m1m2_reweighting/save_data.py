import bilby
import os
import numpy as np
from deep_gw_pe_followup.restricted_prior.conversions import calc_a2, calc_xp
import pandas as pd
from tqdm import tqdm

HIGH_Q = dict(fname="outdir_highq/result/highq_0_result.json", color="tab:green")
LOW_Q = dict(fname="outdir_loqq/result/lowq_0_result.json", color="tab:green")

PLOT_PARAMS = ["chi_p", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
N = 50000


def sample_in_batches(p, N):
    num_batches = 20
    batch_size = int(N / num_batches)
    combined = []
    for batch_i in tqdm(range(num_batches)):
        b = pd.DataFrame(p.sample(batch_size))
        combined.append(b)
    return pd.concat(combined)


def load_prior_samples(r):
    smps = sample_in_batches(r.priors, N)
    smps = add_params_to_df(smps)
    return smps


def add_params_to_df(p):
    p['tilt_1'] = np.arccos(p.cos_tilt_1)
    p['tilt_2'] = np.arccos(p.cos_tilt_2)
    p['a_2'] = calc_a2(
        xeff=p.chi_eff,
        q=p.mass_ratio, cos1=p.cos_tilt_1, cos2=p.cos_tilt_2, a1=p.a_1
    )
    sin1 = np.sin(p['tilt_1'])
    sin2 = np.sin(p['tilt_1'])
    p['chi_p'] = calc_xp(a1=p.a_1, a2=p.a_2, q=p.mass_ratio, sin1=sin1, sin2=sin2)
    return p


def load_res(fname):
    r = bilby.gw.result.CBCResult.from_json(fname)
    r.outdir = os.path.dirname(fname)
    r.posterior = add_params_to_df(r.posterior)
    return r


def main():
    # r = load_res(LOW_Q['fname'])
    # prior_samp = load_prior_samples(r)
    # r.posterior[PLOT_PARAMS].to_parquet('posterior.parquet.gzip', compression='gzip')
    # prior_samp[PLOT_PARAMS].to_parquet('prior.parquet.gzip', compression='gzip')
    r = load_res(HIGH_Q['fname'])
    prior_samp = load_prior_samples(r)
    r.posterior[PLOT_PARAMS].to_parquet('high_q_posterior.parquet.gzip', compression='gzip')
    prior_samp[PLOT_PARAMS].to_parquet('high_q_prior.parquet.gzip', compression='gzip')

if __name__ == '__main__':
    main()

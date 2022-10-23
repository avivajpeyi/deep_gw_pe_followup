import bilby
import os
import numpy as np
from deep_gw_pe_followup.restricted_prior.conversions import calc_a2, calc_xp
from agn_utils.plotting.overlaid_corner_plotter import overlaid_corner, CORNER_KWARGS
import pandas as pd

LOW_Q = dict(fname="outdir_loqq/result/lowq_0_result.json", color="tab:green")
HIGH_Q = dict(fname="outdir_highq/result/highq_0_result.json", color="tab:blue")

PLOT_PARAMS = ["chi_p", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
N = 10000


def load_prior_samples(r):
    smp_file = f"{r.outdir}/prior_samps.csv"
    if os.path.exists(smp_file):
        smps = pd.read_csv(smp_file)
    else:
        print("Caching prior samps")
        smps = pd.DataFrame(r.priors.sample(N))
        smps = add_params_to_df(smps)
        smps.to_csv(smp_file, index=False)
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


def plot_samples(s, fname, color):
    r = bilby.result.Result()
    r.posterior = s[PLOT_PARAMS]
    r.outdir = os.path.dirname(fname)
    r.search_parameter_keys = PLOT_PARAMS.copy()
    r.parameter_labels_with_unit = PLOT_PARAMS.copy()
    r.plot_corner(
        parameters=PLOT_PARAMS.copy(),
        color=color,
        labels=PLOT_PARAMS.copy(),
        filename=fname,
        **CORNER_KWARGS
    )


def make_plots(fname, color):
    r = load_res(fname)
    prior_samp = load_prior_samples(r)
    overlaid_corner(
        samples_list=[r.posterior.sample(N), prior_samp],
        samples_colors=[color, "gray"],
        sample_labels=["Posterior", "Prior"],
        params=PLOT_PARAMS,
        fname=fname.replace("result.json", "combined.png"),
        override_kwargs=CORNER_KWARGS
    )


if __name__ == '__main__':
    make_plots(LOW_Q['fname'], LOW_Q['color'])

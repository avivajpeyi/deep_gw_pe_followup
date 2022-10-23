import bilby
import os
import numpy as np
from deep_gw_pe_followup.restricted_prior.conversions import calc_a2, calc_xp
from agn_utils.plotting.overlaid_corner_plotter import overlaid_corner
import pandas as pd


LOW_Q = dict(fname="outdir_loqq/result/lowq_0_result.json", color="tab:green")
HIGH_Q = dict(fname="outdir_highq/result/highq_0_result.json", color="tab:blue")


def add_param_to_posterior(r):
    p = r.posterior
    p['a_2'] = calc_a2(
        xeff=p.chi_eff,
        q=p.mass_ratio, cos1=p.cos_tilt_1, cos2=p.cos_tilt_2, a1=p.a_1
    )
    sin1 = np.sin(p['tilt_1'])
    sin2 = np.sin(p['tilt_1'])
    p['chi_p'] = calc_xp(a1=p.a_1, a2=p.a_2, q=p.mass_ratio, sin1=sin1, sin2=sin2)
    r.posterior = p
    return r

def sample_prior(r):
    samples = r.prior.sample(5000)


def plot_res(fname, color):
    r = bilby.gw.result.CBCResult.from_json(fname)
    r.outdir = os.path.dirname(fname)
    r = add_param_to_posterior(r)
    r.plot_corner(
        parameters=["chi_p", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"],
        color=color
    )



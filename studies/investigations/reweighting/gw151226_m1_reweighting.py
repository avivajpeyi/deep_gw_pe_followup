import os

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from bilby.core.result import Result
import pandas as pd

from bilby_report.tools.image_utils import make_gif

from deep_gw_pe_followup.uniform_component_mass_prior_reweighter import (
    m1m2_prior_from_mcq_prior, m1m2_prior_with_q_cut, get_m1_weight
)
from deep_gw_pe_followup import get_mpl_style
from tqdm.auto import tqdm

get_mpl_style()

CLEAN_AFTER = False

IAS_q = 0.10537432884262372
LVK_q = 0.29103432408863716

mc_prior = Uniform(name='chirp_mass', minimum=6.0, maximum=15.0)
m1_prior = Uniform(minimum=3.022, maximum=54.398, latex_label=r"$m_1^{\rm orig}$", name='m1_orig')
m2_prior = Uniform(minimum=3.022, maximum=54.398, latex_label=r"$m_2^{\rm orig}$", name='m2_orig')

OUT = "outdir"



def main():
    ias_fnames, lvk_fnames = [],[]
    qtols = [0.05, 0.005, 0.0005, 0.00005, 0.000005]
    for i, qtol in tqdm(enumerate(qtols)):
        ias_fnames.append(run_expt(f"IAS {i}", IAS_q, qtol))
        lvk_fnames.append(run_expt(f"LVK {i}", LVK_q, qtol))
    make_gif(image_paths=ias_fnames, gif_save_path=f"{OUT}/ias.gif", duration=150)
    make_gif(image_paths=lvk_fnames, gif_save_path=f"{OUT}/lvk.gif", duration=150)

def run_expt(label, true_q, q_tolerance=0.0005):
    outdir = f"{OUT}"
    os.makedirs(outdir, exist_ok=True)
    q_prior = Uniform(minimum=true_q - q_tolerance, maximum=true_q + q_tolerance)
    return plot_weights(q_prior, fname=f"{outdir}/{label}.png", qtol=q_tolerance, label=label)

def plot_weights(q_prior, fname, qtol, label):
    m1_prior_old, _ = m1m2_prior_from_mcq_prior.get_m1m2_prior_from_mcq_prior(
        mc_bounds=[mc_prior.minimum, mc_prior.maximum],
        q_bounds=[q_prior.minimum, q_prior.maximum],
    )
    m1_prior_new, _, _ = m1m2_prior_with_q_cut.get_m1m2_prior_after_q_cut(
        m1_bounds=[m1_prior.minimum, m1_prior.maximum],
        m2_bounds=[m2_prior.minimum, m2_prior.maximum],
        q_bounds=[q_prior.minimum, q_prior.maximum],
    )

    m1 = np.linspace(m1_prior.minimum, m1_prior.maximum, 1000)
    ln_weights = get_m1_weight.get_m1_ln_weights(m1, orig_pri=m1_prior_old, new_pri=m1_prior_new)

    plt.close('all')
    plt.figure(figsize=(4, 3))
    plt.plot(m1, ln_weights, color="tab:blue")
    plt.fill_between(m1, ln_weights, min(ln_weights), alpha=0.4, color="tab:blue")
    plt.xlim(m1_prior.minimum, m1_prior.maximum)
    plt.ylim(-2, 4)
    plt.xlabel(r"$m_1$")
    plt.ylabel(r"Ln Weights")
    plt.title(f"{label} (q-tol = {qtol})")
    plt.tight_layout()
    plt.savefig(fname)
    return fname

if __name__ == '__main__':
    main()
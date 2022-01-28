import bilby
from deep_gw_pe_followup.plotting.hist2d import seaborn_plot_hist
import matplotlib.pyplot as plt
from deep_gw_pe_followup.restricted_prior import RestrictedPrior
from agn_utils.plotting.overlaid_corner_plotter import overlaid_corner
from deep_gw_pe_followup.restricted_prior.conversions import calc_xeff, calc_a2
from bilby.gw.conversion import generate_all_bbh_parameters
import pandas as pd
import os
import numpy as np

IAS = "priors/ias_restricted.prior"
LVK = "priors/lvk_restricted.prior"

def get_prior(fname):
    return RestrictedPrior(filename=fname)

def remove_mass_constraints(prior:RestrictedPrior):
    p = prior.copy()
    p.pop('mass_1')
    p.pop('mass_2')
    return p

def get_samples_df(prior):
    s = pd.DataFrame(prior.sample(1500))
    s['chi_eff'] = prior.xeff
    s['mass_ratio'] = prior.q
    s['a_2'] = calc_a2(xeff=s.chi_eff, q=s.mass_ratio, a1=s.a_1, cos1=s.cos_tilt_1, cos2=s.cos_tilt_2)
    s = generate_all_bbh_parameters(s)
    return s


def load_samples(prior:RestrictedPrior, fname):
    if os.path.isfile(fname):
        df = pd.read_csv(fname)
    else:
        df = get_samples_df(prior)
        df.to_csv(fname, index=False)
        print(f"caching {fname} samples")
    return df


def calculate_throw_fraction(samples_without_constraint, prior_with_constraints:RestrictedPrior):
    samples_without_constraint = samples_without_constraint[list(prior_with_constraints.keys())]
    samples_without_constraint = samples_without_constraint.drop(['mass_ratio', 'chi_eff'], axis=1)
    s_dict = samples_without_constraint.to_dict('records')
    ln_prob = np.array([prior_with_constraints.ln_prob(s) for s in s_dict])
    return np.sum(np.isinf(ln_prob))/len(ln_prob)




def plot_samples(prior:RestrictedPrior, name):
    without_fname = f"{name}_samples_without_constraints.csv"
    with_fname = f"{name}_samples_with_constraints.csv"

    s_with_constraint = load_samples(prior, with_fname)
    no_constraint_prior = remove_mass_constraints(prior)
    s_no_constraint = load_samples(no_constraint_prior, without_fname)

    frac = calculate_throw_fraction(s_no_constraint, prior)
    print(f"frac: {frac}")

    overlaid_corner(
        [s_with_constraint, s_no_constraint],
        ['mass_1', 'mass_2', 'a_1', 'a_2', 'chirp_mass', ],
        ["tab:blue", "tab:orange"],
        sample_labels=["m1-m2 constrained", "no m1-m2 constraint"],
        fname=f"{name}_prior_constraints.png",
        title=name,
        truths=None,
        ranges=[(16,60), (2 ,8), (0.5,1), (0,1), (5.5 ,15.5)],
        quants=False,
        override_kwargs={}
    )

    plt.hist(s_with_constraint.chirp_mass, histtype='step', density=True, label="m1-m2 constrained", color="tab:blue")
    plt.hist(s_no_constraint.chirp_mass, histtype='step', density=True, label="no m1-m2 constraint", color="tab:orange")
    plt.ylabel("Density")
    plt.xlabel("Chirp Mass")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{name}_chirpmass.png")


if __name__ == '__main__':
    print("Getting prior")
    ias_p = get_prior(IAS)
    print("plotting")
    plot_samples(ias_p, "IAS")

    print("Getting prior")
    ias_p = get_prior(LVK)
    print("plotting")
    plot_samples(ias_p, "LVK")

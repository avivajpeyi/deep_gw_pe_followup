# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: venv
#     language: python
#     name: venv
# ---

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import numpy as np
from deep_gw_pe_followup.restricted_prior import RestrictedPrior, PlaceholderPrior

from deep_gw_pe_followup.plotting import plot_probs
from deep_gw_pe_followup.plotting import plot_heatmap
import bilby
from bilby.core.prior import Cosine, Sine, Uniform
import os

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

bilby.core.utils.setup_logger(label="interactive")




prior_dict = dict(
    # The following constraints also set a1, a2, cos1, cos2
    mass_ratio=0.2,
    chi_eff=0.1,
    a_1 = PlaceholderPrior(name='a_1'),
    cos_tilt_1 = PlaceholderPrior(name='cos_tilt_1'),
    cos_tilt_2 = PlaceholderPrior(name='cos_tilt_2'),
    # remaining spin params
    phi_12 = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'),
    phi_jl = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'),
    # remaining mass params
    chirp_mass = Uniform(name='chirp_mass', minimum=25, maximum=100),
    # remaining params
    luminosity_distance = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=1e2, maximum=5e3),
    dec = Cosine(name='dec'),
    ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
    theta_jn = Sine(name='theta_jn'),
    psi = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
    phase = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic'),
)
prior = RestrictedPrior(dictionary=prior_dict, cache="cache_q0_2-xeff0_1")
# -

help(plot_probs)



from ipywidgets import *


# +
from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 8


def plot_a1_cos1(a1=0.5):
    d = prior.cached_cos1_data
    fig, axes = plt.subplots(2,2, figsize=(5,5))
    axes[0,1].remove()
    heat_ax = axes[1,0]
    a1_ax = axes[0,0]
    c1_ax = axes[1,1]
    plot_heatmap(d.a1, d.cos1, d.p_cos1, heat_ax)
    heat_ax.set_xlabel('a1')
    heat_ax.set_ylabel('cos1')
    c1_ax.set_xlabel('cos1')
    a1_ax.set_xlabel('a1')
    a1_ax.plot(prior['a_1'].xx, prior['a_1'].yy)
    c1_pr = prior.get_cos1_prior(a1)
    c1_ax.plot(c1_pr.xx, c1_pr.yy)
    c1_ax.set_xlim(-1,1)
    a1_ax.set_xlim(0,1)
    a1_ax.set_yticks([])
    c1_ax.set_yticks([])
    heat_ax.axvline(a1, c='green')
    a1_ax.axvline(a1, c='green')
    return fig, axes
    
interact(plot_a1_cos1, a1=FloatSlider(min=0, max=1, step=0.01, continuous_update=False))
# +
def plot_row(a1=0.5, c1=0.5, c2=0.1):
    d = prior.cached_cos1_data
    fig, axes = plt.subplots(1,4, figsize=(10,2.5))
    a1_ax = axes[0]
    c1_ax = axes[1]
    c2_ax = axes[2]
    a2_ax = axes[3]
    
    a1_ax.set_xlabel('a1')
    c1_ax.set_xlabel('cos1')
    c2_ax.set_xlabel('cos2')
    a2_ax.set_xlabel('a2')
    
    a1_ax.plot(prior['a_1'].xx, prior['a_1'].yy, lw=2)
    a1_ax.axvline(a1, c='C1', label=f"a1={a1:0.2f}")
    
    c1_pr = prior.get_cos1_prior(a1)
    c1_ax.plot(c1_pr.xx, c1_pr.yy, lw=2)
    c1_ax.axvline(c1, c='C1', label=f"c1={c1:0.2f}")
    
    
    c2_pr = prior.get_cos2_prior(given_a1=a1, given_cos1=c1)
    if isinstance(c2_pr, PlaceholderDelta):
        c2_ax.axvline(c2_pr.peak, c='C0', lw=2, label=f"peak={c2_pr.peak}")
    else:
        c2_ax.plot(c2_pr.xx, c2_pr.yy, lw=2)
    c2_ax.axvline(c2, c='C1', label=f"c2={c2:0.2f}")
    
    a2 = calc_a2(xeff=0.1, q=0.2, cos1=c1, cos2=c2, a1=a1)
    a2_ax.axvline(a2, c='C1', label=f"a2={a2:0.2f}")
    
    
    a1_ax.set_xlim(0,1)
    c1_ax.set_xlim(-1,1)
    c2_ax.set_xlim(-1,1)
    a2_ax.set_xlim(0,1)
    
    for ax in axes:
        ax.set_yticks([])
        ax.legend()
        ax.grid(False)




interact(plot_row, 
         a1=FloatSlider(value=0.25, min=0, max=1, step=0.01, continuous_update=False),
         c1=FloatSlider(value= 0.5, min=-1, max=1, step=0.01, continuous_update=False),
         c2=FloatSlider(value = 0.5, min=-1, max=1, step=0.01, continuous_update=False),
        )


# +
param = {'t1': 2.642520338794379, 't2': 2.381671608047824, 
'a1': 0.0931014476863745}

param['c1'] = np.cos(param['t1'])
param['c2'] = 1

plot_row(a1=param['a1'], c1=param['c1'], c2=param['c2'])

# +
param = {'theta_jn': 2.3869668452643813, 'phi_jl': 4.8788378667791, 'tilt_1': 2.362456909547408, 'tilt_2': 0.4326935468536894, 'phi_12': 3.102810772445929, 'a_1': 0.11466769198862962, 'a_2': 1.1102648659098913, 'mass_1': 3.850456300099072e+32, 'mass_2': 7.700912600198145e+31, 'reference_frequency': 50.0, 'phase': 5.366282972133813}


param['c1'] = np.cos(param['tilt_1'])
param['c2'] = np.cos(param['tilt_2'])
param['a1'] = param['a_1']



c2 = plot_row(a1=param['a1'], c1=param['c1'], c2=param['c2'])
# -



# +
from deep_gw_pe_followup.restricted_prior.prior import *

xeff = 0.1
q = 0.2

def get_cos2_prior(given_a1, given_cos1):
    cos2s = np.linspace(-1, 1, 10000)
    dc2 = cos2s[1] - cos2s[0]

    args = (given_a1, xeff, q, given_cos1)
    p_cos2 = np.array([get_p_cos2_given_xeff_q_a1_cos1(cos2, *args) for cos2 in cos2s])
    
    
    print(np.sum(p_cos2))
    p_cos2 = p_cos2 / np.sum(p_cos2) / dc2

    
    
    
    min_b, max_b = find_boundary(cos2s, p_cos2)

    if min_b == max_b:
        return PlaceholderDelta(peak=min_b, name="cos_tilt_2", latex_label=r"$\cos \theta_2$")

    return Interped(
        xx=cos2s, yy=p_cos2,
        minimum=min_b, maximum=max_b, name="cos_tilt_2",
        latex_label=r"$\cos \theta_2$"
    )

c2_p = get_cos2_prior(given_a1=param['a1'], given_cos1=param['c1'])


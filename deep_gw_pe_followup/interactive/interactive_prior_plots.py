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

# +
import numpy as np
from deep_gw_pe_followup.restricted_prior import RestrictedPrior, PlaceholderPrior

from deep_gw_pe_followup.restricted_prior.plotting import plot_ci, plot_probs
from deep_gw_pe_followup.restricted_prior.plotting.hist2d import plot_heatmap
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
# -



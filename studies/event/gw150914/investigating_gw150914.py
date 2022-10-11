# +
# %load_ext autoreload
# # %load_ext autotime
# # %load_ext jupyternotify
# %autoreload 2
# %matplotlib inline

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
import numpy as np
from scipy import stats
import os
import pandas as pd
from scipy.ndimage import gaussian_filter

from scipy.interpolate import interp1d
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import deep_gw_pe_followup
from bilby.core.prior import Uniform, DeltaFunction, Constraint
from deep_gw_pe_followup.utils.isotropic_spins_prior_odds import print_isotropic_spin_qxeff_prior_odds, get_isotropic_spin_prior_samples
from deep_gw_pe_followup.restricted_prior import RestrictedPrior
from deep_gw_pe_followup.plotting.plot_qxeff import plot_qxeff





GW150914_POSTERIOR_FN = "data/gw150914.dat"
GW150914_POSTERIOR_URL = "https://raw.githubusercontent.com/prayush/GW150914_GW170104_NRSur7dq2_Posteriors/master/GW150914/NRSur7dq2_RestrictedPriors.dat"
GW150914_TIMESERIES = dict(
    h1="https://www.gw-openscience.org/GW150914data/H-H1_LOSC_4_V2-1126259446-32.gwf",
    l1="https://www.gw-openscience.org/GW150914data/L-L1_LOSC_4_V2-1126259446-32.gwf"
)



PSD_FN = dict(h1="data/h1_psd.txt",l1="data/l1_psd.txt")
PSD_URL = "https://git.ligo.org/lscsoft/parallel_bilby/-/raw/master/examples/GW150914_IMRPhenomPv2/psd_data/{}_psd.txt?inline=false"
SIGMA_LEVELS = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)]



def save_gw150914_data():
    # can only run on machine with nds2
    from gwosc.datasets import event_gps
    from gwpy.timeseries import TimeSeries
    trigger_time = event_gps("GW150914")

    end_time = trigger_time + 2.1
    start_time = end_time - 4 - 0.1

    channel = "DCS-CALIB_STRAIN_C02"

    for d in ["H1", "L1"]:
        data = TimeSeries.get(f"{d}:channel",
                              start_time,
                              end_time,
                              allow_tape=True)
        TimeSeries.write(data,
                         target=f'.data/{d}.gwf',
                         format='gwf')


def download_file(fn, url):
    subprocess.run(
        [
            f"""wget -O "{fn}" "{url}" """
        ],
        shell=True,
    )


def load_gw150914_samples():
    if not os.path.isfile(GW150914_POSTERIOR_FN):
        download_file(GW150914_POSTERIOR_FN, GW150914_POSTERIOR_URL)
    posterior = np.genfromtxt(GW150914_POSTERIOR_FN, names=True)
    df = pd.DataFrame(posterior)
    df["xeff"] = df.chi_eff
    return df

def load_psd():
    psd_dat = {}
    for det, fn in PSD_FN.items():
        if not os.path.isfile(fn):
            download_file(fn, PSD_URL.format(det))
        psd_dat[det] = np.loadtxt(fn).T
    return psd_dat



def add_letter_marker(ax, x, y, color, letter):
    kwgs = dict(color=color, zorder=100)
    ax.scatter(x, y, marker=".", s=10, **kwgs)
    ax.annotate(
        letter,
        xy=(x, y),
        xycoords="data",
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=10,
        **kwgs,
    )

def plot_psd():
    psds = load_psd()
    fig, ax = plt.subplots(1,1)
    for det, psd_dat in psds.items():
        ax.plot(psd_dat[0], psd_dat[1], label=det)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlim(10, 1100)
    plt.ylim(1e-49, 1e-32)
    plt.tight_layout()
    plt.savefig("psds.png")



def plot_max_gw150914_params(params = ['dist', 'ra', 'dec', 'psi']):
    p = load_gw150914_samples()

    fig, axes = plt.subplots(1,len(params), figsize=(2*len(params), 2))
    for ax, pa in zip(axes, params):
        c, b, patch = ax.hist(p[pa], density=True, bins=100, histtype='step')
        ax.set_yticks([])
        max_pt = b[np.where(c == c.max())][0]
        ax.axvline(max_pt,  color='tab:green', label=f"Max: {max_pt:.2f}")
        # print(pa, max_pt)
        ax.set_xlabel(pa)
    plt.tight_layout()
    plt.savefig("max_gw150914_params.png")



def main():
    # save_gw150914_data()

    print_isotropic_spin_qxeff_prior_odds(PTS)
    # plot_max_gw150914_params()
    gw150914_samples = load_gw150914_samples()
    plot_qxeff(gridfname="gw150914_kde.npz", clean=False, heatmap=True, fname="gw150914_qxeff.png", colorbar=False, samples=gw150914_samples, pts=PTS)
    iso_spin_prior_sample = get_isotropic_spin_prior_samples()
    plot_qxeff(gridfname="isospin_prior_kde.npz", clean=False, heatmap=True, fname="isospin_prior_qxeff.png", colorbar=False, samples=iso_spin_prior_sample, pts=PTS)
    plot_psd()




PTS = dict(
    A=dict(q=0.7, xeff=0.02, color="tab:blue"),
    # B=dict(q=0.9, xeff=-0.1, color="tab:green"),
    C=dict(q=0.78, xeff=-0.04, color="tab:red"),
    # D=dict(q=0.95, xeff=-0.01, color="tab:purple"),
)


START_Q, START_XEFF = 0.85, 0

PTS = {"0":dict(q=START_Q, xeff=START_XEFF, color="k")}
for i in range(1,3):
    PTS.update({f"{i}":dict(q=START_Q+i*(0.05), xeff=START_XEFF, color="k")})
    PTS.update({f"{i+2}":dict(q=START_Q, xeff=START_XEFF+i*(0.05), color="k")})


for l, pt in PTS.items():
    print(f"{l}=dict(q={pt['q']}, xeff={pt['xeff']}), ")


if __name__ == "__main__":
    main()

# +
import glob
from bilby.gw.result import CBCResult
import os

result_files = glob.glob("out*/res*/*.json")
names = [os.path.basename(p).split("_")[0] for p in result_files]
loaded_results = {n:CBCResult.from_json(p) for p, n in zip(result_files, names)}


# -

for name, res in loaded_results.items():
    fig = res.plot_corner(parameters=["mass_1","mass_2"], save=False)
    fig.suptitle(name)


loaded_results.keys()

# +
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import numpy as np

def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)




# -

old_res = "/fred/oz980/avajpeyi/projects/GW150914_downsampled_posterior_samples.dat"
old = pd.read_csv(old_res, delimiter=" ")


# +
from corner import corner
p = ['mass_1','mass_2']
leg_colrs = dict(
    GWTC="black",
    AVI_FULL='red',
    DEEP_A="green",
    DEEP_C="purple"
)



max_len = 50000


posts = [
    loaded_results['ptA'].posterior[p],
    loaded_results['ptC'].posterior[p],
    loaded_results['full'].posterior[p]
]


KW = dict(plot_datapoints=False,smooth=0.9,max_n_ticks=3, labels=['m1','m2'], plot_density=False, levels=(0.68,))
f = corner(old[p], **KW)
f = corner(posts[0], fig=f, color="green", **KW,
          weights=get_normalisation_weight(len(posts[0]), max_len))
f = corner(posts[1], fig=f, color="purple", **KW,
          weights=get_normalisation_weight(len(posts[1]), max_len),)
f = corner(posts[2], fig=f, color="red", **KW,
          weights=get_normalisation_weight(len(posts[2]), max_len),)


leg_colrs = dict(
    GWTC="black",
    AVI_FULL='red',
    DEEP_A="green",
    DEEP_C="purple"
)

custom_lines = []
for l, cl in leg_colrs.items():
    custom_lines.append(Line2D([0], [0], color=cl, lw=4))
l = plt.legend(custom_lines, list(leg_colrs.keys()), frameon=False, fontsize=14, loc=(0.3, 1))
# -

len(old)

old.log_likelihood

loaded_results['full'].posterior.log_likelihood

plt.hist(old.log_likelihood, density=True, histtype='step', color="black", label="GWTC")
plt.hist(loaded_results['full'].posterior.log_likelihood, density=True, histtype='step', color="red", label="AVI GW150914")
plt.xlabel("Log Likelihood")
plt.legend(frameon=False, fontsize=14)


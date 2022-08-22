# ! pip install jupyter-autotime -q

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

from deep_gw_pe_followup.restricted_prior import RestrictedPrior




GW150914_POSTERIOR_FN = "data/gw150914.dat"
GW150914_POSTERIOR_URL = "https://raw.githubusercontent.com/prayush/GW150914_GW170104_NRSur7dq2_Posteriors/master/GW150914/NRSur7dq2_RestrictedPriors.dat"
GW150914_TIMESERIES = dict(
    h1="https://www.gw-openscience.org/GW150914data/H-H1_LOSC_4_V2-1126259446-32.gwf",
    l1="https://www.gw-openscience.org/GW150914data/L-L1_LOSC_4_V2-1126259446-32.gwf"
)



PSD_FN = dict(h1="data/h1_psd.txt",l1="data/l1_psd.txt")
PSD_URL = "https://git.ligo.org/lscsoft/parallel_bilby/-/raw/master/examples/GW150914_IMRPhenomPv2/psd_data/{}_psd.txt?inline=false"
SIGMA_LEVELS = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)]


QLIM, XEFFLIM = (0.5, 1), (-0.2, 0.2)

ORANGE = "#eaa800"

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


def load_samples():
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


def gauss_smooth(args, smooth):
    x, y, z = args
    z = gaussian_filter(z, smooth)
    return x, y, z


def add_contours(ax, cargs, levels, color, label_col=None, smooth=None):
    if smooth:
        cargs = gauss_smooth(cargs, smooth)
    CS = ax.contour(
        *cargs,
        levels=levels,
        linewidths=0.5,
        colors=color,
        alpha=0.7,
        zorder=100,
    )
    if label_col:
        ax.clabel(
            CS, CS.levels, inline=True, fmt=fmt_contour, fontsize=10, colors=label_col
        )


def make_colormap_to_white(color="tab:orange"):
    color_rgb = np.array(to_rgba(color))
    lower = np.ones((int(256 / 4), 4))
    for i in range(3):
        lower[:, i] = np.linspace(1, color_rgb[i], lower.shape[0])
    cmap = np.vstack(lower)
    return ListedColormap(cmap, name="myColorMap", N=cmap.shape[0])


def fmt_contour(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


def calc_and_save_kde_grid(fname, N=200j):
    print("Calculating KDE grid")
    x, y, z = evaluate_kde_on_grid(get_kde(), x=QLIM, y=XEFFLIM, num_gridpoints=N)
    np.savez_compressed(fname, x=x, y=y, z=z)


def get_kde():
    samples = load_samples()
    return stats.gaussian_kde(np.vstack([samples.q, samples.xeff]))


def evaluate_kde_on_grid(kde, x, y, num_gridpoints=200j):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    x_grid, y_grid = np.mgrid[xmin:xmax:num_gridpoints, ymin:ymax:num_gridpoints]
    xy_array = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(xy_array).T
    return x_grid, y_grid, z.reshape(x_grid.shape)


def plot_heatmap(x, y, z, ax, cmap, add_cbar=False, smooth=None):
    x = np.unique(x)
    y = np.unique(y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = z.reshape(len(x), len(y))
    args = (X, Y, Z)
    if smooth:
        args = gauss_smooth(args, smooth)
    cbar = ax.pcolor(
        *args, cmap=cmap, vmin=np.nanmin(z), vmax=np.nanmax(z), zorder=-100
    )
    if add_cbar:
        fig = ax.get_figure()
        cbar = fig.colorbar(cbar, ax=ax)
        return cbar
    return None


def query_z_value_for_xy(query_x, query_y):
    return get_kde()([query_x, query_y])[0]


def load_kde_samples(clean=False):
    KDE_GRID = "gw150914_kde.npz"
    if clean:
        os.remove(KDE_GRID)
    if not os.path.isfile(KDE_GRID):
        calc_and_save_kde_grid(KDE_GRID)
    return np.load(KDE_GRID)


def plot_qxeff(clean=False, heatmap=True, fname="qxeff.png", colorbar=False):
    grid = load_kde_samples(clean)
    fig, ax = plt.subplots(figsize=(4, 4))
    args = (grid["x"], grid["y"], grid["z"])
    levels = [1e0, 1e1]

    cmap = make_colormap_to_white(ORANGE)
    if heatmap:
        cbar = plot_heatmap(*args, ax, cmap, add_cbar=colorbar)
    else:
        ax.contourf(*args, levels=levels, cmap=cmap, zorder=-50, extend="max")

    add_contours(ax, args, levels, ORANGE, label_col="black")
    for pt_label, pt in PTS.items():
        z_val = query_z_value_for_xy(pt["q"], pt["xeff"])
        add_letter_marker(
            ax, pt["q"], pt["xeff"], pt["color"], f"{pt_label} ({z_val:0.3f})"
        )

        PTS[pt_label]["z_val"] = z_val
    ax.set(xlabel=r"$q$", ylabel=r"$\chi_{\rm eff}$")
    ax.set_xlim(*QLIM)
    ax.set_ylim(*XEFFLIM)
    ax.set_xticks([0.5, 1])
    ax.set_yticks([-0.2, 0, 0.2])
    plt.minorticks_off()
    if colorbar:
        cax = cbar.ax
        for pt_label, pt in PTS.items():
            cax.hlines(
                pt["z_val"], 0, 1, colors=pt["color"], linewidth=3, linestyles="-"
            )
            print(f"Pt {pt_label}: {pt['z_val']}")
        cax.set_yticks([p["z_val"] for p in PTS.values()])
        cax.set_ylim(0, 0.1)
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
        fig.set_size_inches([6, 4], forward=True)
    plt.tight_layout()
    plt.savefig(fname, transparent=True, dpi=200)


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
    p = load_samples()

    fig, axes = plt.subplots(1,len(params), figsize=(2*len(params), 2))
    for ax, pa in zip(axes, params):
        c, b, patch = ax.hist(p[pa], density=True, bins=100, histtype='step')
        ax.set_yticks([])
        max_pt = b[np.where(c == c.max())][0]
        ax.axvline(max_pt,  color='tab:green', label=f"Max: {max_pt:.2f}")
        print(pa, max_pt)
        ax.set_xlabel(pa)
    plt.tight_layout()
    plt.savefig("max_gw150914_params.png")


def main():
    # save_gw150914_data()
    plot_max_gw150914_params()
    plot_qxeff(clean=False, heatmap=True, fname="qxeff.png", colorbar=False)
    plot_psd()
    print(f"O(C/A) = {PTS['C']['z_val']/PTS['A']['z_val']:.2f}")
    print(f"O(A/B) = {PTS['A']['z_val']/PTS['B']['z_val']:.2f}")
    print(f"O(C/B) = {PTS['C']['z_val']/PTS['B']['z_val']:.2f}")
    print("Plotting priors:")
    for label, pt in PTS.items():
        p = RestrictedPrior(filename=f"priors/pt{label}.prior")
        p.plot_cache()


PTS = dict(
    A=dict(q=0.7, xeff=0.02, color="tab:blue"),
    B=dict(q=0.9, xeff=-0.1, color="tab:green"),
    C=dict(q=0.78, xeff=-0.04, color="tab:red"),
)


if __name__ == "__main__":
    main()
# -




from .utils import *

from deep_gw_pe_followup.sample_cacher.kde2d import load_kde_gridpts
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
from itertools import combinations

ORANGE = "#B37570"

def plot_qxeff(gridfname, samples, clean=False, heatmap=True, fname="qxeff.png", colorbar=False,  pts={}):

    kde = get_kde(samples.q, samples.xeff)
    grid = load_kde_gridpts(gridfname, clean=clean, samples=samples)
    z_vals = {l: kde_predict(kde, pt['q'], pt['xeff']) for l, pt in pts.items()}

    if len(pts)>1:
        print("KDE ODDS:")
        for combo in list(combinations(z_vals.keys(),r=2)):
            pt0_z, pt1_z = z_vals[combo[0]], z_vals[combo[1]]
            print(f">>> KDE O({combo[0]}/{combo[1]}) = {pt0_z:.2f}/{pt1_z:.2f} = {pt0_z/pt1_z:.2f}")

    fig, ax = plt.subplots(figsize=(4, 4))
    args = (grid["x"], grid["y"], grid["z"])
    levels = [1e0, 1e1]


    # cmap = make_colormap_to_white(ORANGE)
    cmap = 'hot'
    if not heatmap:
        cbar = ax.contourf(*args, levels=levels, cmap=cmap, zorder=-50, extend="max")
    else:
        add_cntr(ax, samples.q.values, samples.xeff.values, cmap, label="", add_heat=0,  plot_median=False, label_contour_cols=['k','k'])

        if colorbar:
            cax = cbar.ax
            for pt, zval in z_vals.items():
                cax.hlines(
                    zval, 0, 1, colors=pts[pt]["color"], linewidth=3, linestyles="-"
                )
            cax.set_yticks(z_vals.values())
            cax.set_ylim(0, 0.1)
            # ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
            # fig.set_size_inches([6, 4], forward=True)

    # add_contours(ax, args, levels, ORANGE, label_col="black")



    for l, pt in pts.items():
        add_letter_marker(
            ax, pt["q"], pt["xeff"], pt["color"], f"{l}"
        )


    ax.set(xlabel=r"$q$", ylabel=r"$\chi_{\rm eff}$")
    ax.set_xlim(min(samples.q), max(samples.q))
    ax.set_ylim(min(samples.xeff), max(samples.xeff))
    # ax.set_xticks([0.5, 1])
    # ax.set_yticks([-0.2, 0, 0.2])
    plt.minorticks_off()


    plt.tight_layout()

    plt.savefig(fname, transparent=True, dpi=200)

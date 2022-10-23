from deep_gw_pe_followup.restricted_prior import RestrictedPrior
import matplotlib.pyplot as plt
from deep_gw_pe_followup import get_mpl_style
from deep_gw_pe_followup.plotting.hist2d import make_colormap_to_white, plot_heatmap
import os
import numpy as np

plt.style.use(get_mpl_style())

LAB_a1 = r"$\chi_1$"
LAB_cos1 = r"$\cos\theta_1$"
LAB_pa1 = r"$p(\chi_1|q,\chi_{\rm eff})$"
LAB_pc1 = r"$p(\cos\theta_1|\chi_1, q,\chi_{\rm eff})$"
LAB_pa1c1 = r"$p(\chi_1, \cos\theta_1| q,\chi_{\rm eff})$"
FSIZE = (3, 3)

COLOR = "tab:orange"

def add_a1_line(ax, a1):
    for i, a1i in enumerate(a1):
        ax.axvline(a1i, color=f"C{i + 1}", linestyle="dashed")


def plot_p_a1(p, basepth, a1=[]):
    y = p['a_1'].yy
    fig, ax = plt.subplots(1, 1, figsize=FSIZE)
    ax.set_xlabel(LAB_a1)
    ax.set_ylabel(LAB_pa1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([min(y), max(y) * 1.1])
    ax.set_xlim([0, 1])
    ax.plot(p['a_1'].xx, y, color=COLOR)
    if len(a1) > 0:
        add_a1_line(ax, a1)
    plt.minorticks_off()
    plt.tight_layout()
    plt.savefig(f"{basepth}/p_a1.png")


def plot_p_a1_cos1(p, basepth, a1=[]):
    data = p.cached_cos1_data
    fig, ax = plt.subplots(1, 1, figsize=FSIZE)
    ax.set_xlabel(LAB_a1, labelpad=-10)
    ax.set_ylabel(LAB_cos1, labelpad=-10)

    ax.set_xticks([0, 1])
    ax.set_yticks([-1, 1])
    cmap = make_colormap_to_white(COLOR)
    plot_heatmap(x=data['a1'], y=data['cos1'], p=data['p_cos1'], ax=ax, cmap=cmap)
    if len(a1) > 0:
        add_a1_line(ax, a1)
    plt.tight_layout()
    plt.minorticks_off()
    plt.savefig(f"{basepth}/p_a1c1.png")


def plot_p_cos1_given_a1(p, a1, basepth):
    fig, ax = plt.subplots(1, 1, figsize=FSIZE)
    ax.set_xlabel(LAB_cos1)
    ax.set_ylabel(LAB_pc1)
    ax.set_yticks([])
    ax.set_xticks([-1, 0, 1])
    ax.set_xlim([-1, 1])
    for i, a1i in enumerate(a1):
        c1_p = p.get_cos1_prior(a1i)
        y = c1_p.yy
        ax.plot(c1_p.xx, y, color=f"C{i + 1}")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.minorticks_off()
    plt.savefig(f"{basepth}/p_c1.png")


def plot_cache(p, a1):
    pth = f"{p.cache}/given_a1"
    os.makedirs(pth, exist_ok=True)
    # plot_p_a1(p, pth, a1)
    plot_p_a1_cos1(p, pth, a1)
    # plot_p_cos1_given_a1(p, a1, pth)
    plt.tight_layout()
    plt.savefig(f"{p.cache}/plot.png")


if __name__ == "__main__":
    highq_prior = RestrictedPrior(filename="../../gw151226_ias_points_no_m1m2_reweighting/priors/high_q.prior")
    plot_cache(highq_prior, a1=[])
    # ias_prior = RestrictedPrior(filename="priors/low_q.prior")
    # plot_cache(ias_prior, a1=0.8)
    # plot_cache(ias_prior, a1=0.6)

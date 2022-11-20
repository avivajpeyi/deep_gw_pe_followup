
import numpy as np

import matplotlib
import bilby

from collections import namedtuple

import corner
from corner.core import overplot_lines, overplot_points
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging


CORNER_KWARGS = dict(
    bins=50,
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color="black",
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=False,
)

LABELS = dict(
    q=r"$q$",
    xeff=r"$\chi_{\rm eff}$",
    a_1=r"$a_1$",
    a_2=r"$a_2$",
    cos_tilt_1=r"$\cos \theta_1$",
    cos_tilt_2=r"$\cos \theta_2$",
)
PARAMS = dict(
    chi_eff=dict(latex_label=r"$\chi_{\rm eff}$", range=(-1, 1)),
    chi_p=dict(latex_label=r"$\chi_{p}$", range=(0, 1)),
    cos_tilt_1=dict(latex_label=r"$\cos\ \theta_1$", range=(-1, 1)),
    cos_tilt_2=dict(latex_label=r"$\cos\theta_2$", range=(-1, 1)),
    cos_theta_12=dict(latex_label=r"$\cos\ \theta_{12}$", range=(-1, 1)),
    phi_12=dict(latex_label=r"$\phi_{12}$", range=(0, np.pi * 2)),
    tilt_1=dict(latex_label=r"$tilt_{1}$", range=(0, np.pi)),
    remnant_kick_mag=dict(latex_label=r"$|\vec{v}_k|\ $km/s", range=(0, 3000)),
    chirp_mass=dict(latex_label="$M_{c}$", range=(5, 200)),
    mass_1_source=dict(latex_label="$m_1^{\\mathrm{source}}$", range=(0, 200)),
    mass_2_source=dict(latex_label="$m_2^{\\mathrm{source}}$", range=(0, 200)),
    luminosity_distance=dict(latex_label="$d_L$", range=(50, 20000)),
    log_snr=dict(latex_label="$\\rm{log}_{10}\ \\rho$)", range=(-1, 3)),
    snr=dict(latex_label="$\mathrm{SNR}$", range=(0, 20)),
    log_likelihood=dict(latex_label="$\log\mathrm{L}$", range=(40, 100)),
    mass_ratio=dict(latex_label="$q$", range=(0, 1)),
)


def plot_corner(df, fname="corner.png"):
    labels = [LABELS.get(i, i.replace("_", "")) for i in df.columns.values]
    fig = corner.corner(df, labels=labels, **CORNER_KWARGS)
    fig.savefig(fname)



matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=["k", "tab:orange", "tab:blue", "tab:green", "tab:purple"])

logger = logging.getLogger()





def _get_one_dimensional_median_and_error_bar(
        posterior, key, fmt=".2f", quantiles=(0.16, 0.84)
):
    """Calculate the median and error bar for a given key

    Parameters
    ----------
    key: str
        The parameter key for which to calculate the median and error bar
    fmt: str, ('.2f')
        A format string
    quantiles: list, tuple
        A length-2 tuple of the lower and upper-quantiles to calculate
        the errors bars for.

    Returns
    -------
    summary: namedtuple
        An object with attributes, median, lower, upper and string

    """
    summary = namedtuple("summary", ["median", "lower", "upper", "string"])

    if len(quantiles) != 2:
        raise ValueError("quantiles must be of length 2")

    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(posterior[key], quants_to_compute * 100)
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]

    fmt = "{{0:{0}}}".format(fmt).format
    string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    summary.string = string_template.format(
        fmt(summary.median), fmt(summary.minus), fmt(summary.plus)
    )
    return summary


def _add_ci_vals_to_marginalised_posteriors(
        fig, params, posterior: pd.DataFrame
):
    # plt the quantiles
    axes = fig.get_axes()
    for i, par in enumerate(params):
        ax = axes[i + i * len(params)]
        if ax.title.get_text() == "":
            ax.set_title(
                _get_one_dimensional_median_and_error_bar(
                    posterior, par, quantiles=CORNER_KWARGS["quantiles"]
                ).string,
                **CORNER_KWARGS["title_kwargs"],
            )


def overlaid_corner(
        samples_list,
        params,
        samples_colors,
        sample_labels=[],
        fname="",
        title=None,
        truths=None,
        ranges=[],
        quants=False,
        override_kwargs={}
):
    """Plots multiple corners on top of each other

    :param samples_list: list of all posteriors to be plotted ontop of each other
    :type samples_list: List[pd.DataFrame]
    :param sample_labels: posterior's labels to be put on legend
    :type sample_labels: List[str]
    :param params: posterior params names (used to access posteriors samples)
    :type params: List[str]
    :param samples_colors: Color for each posterior
    :type samples_colors: List[Color]
    :param fname: Plot's save path
    :type fname: str
    :param title: Plot's suptitle if not None
    :type title: None/str
    :param truths: posterior param true vals
    :type truths: Dict[str:float]
    :return: None
    """
    logger.info(f"Plotting {fname}")
    logger.info(f"Cols in samples: {samples_list[0].columns.values}")
    # sort the sample columns

    samples_list = [s[params].sample(1500) for s in samples_list]
    base_s = samples_list[0]

    # get plot_posterior_predictive_check range, latex labels, colors and truths
    plot_range, axis_labels = [], []
    for p in params:
        p_data = PARAMS.get(
            p,
            dict(range=(min(base_s[p]), max(base_s[p])), latex_label=f"${p}$"),
        )
        plot_range.append(p_data["range"])
        axis_labels.append(p_data["latex_label"])

    if isinstance(ranges, list):
        if len(ranges) != 0:
            plot_range = ranges
    elif isinstance(ranges, type(None)):
        plot_range = None

    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = min([len(s) for s in samples_list])

    c_kwargs = CORNER_KWARGS.copy()
    c_kwargs.update(
        range=plot_range,
        labels=axis_labels,
        truths=truths,
        truth_color=CORNER_KWARGS["truth_color"]
    )

    for k, v in override_kwargs.items():
        c_kwargs[k] = v

    hist_kwargs = dict(lw=3, histtype='step', alpha=0.5)
    if not quants:
        c_kwargs.pop("quantiles", None)

    fig = corner.corner(
        samples_list[0],
        color=samples_colors[0],
        **c_kwargs,
        hist_kwargs=dict(fc=samples_colors[0], ec=samples_colors[0], **hist_kwargs)
    )

    for idx in range(1, n):
        col = samples_colors[idx]
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=_get_normalisation_weight(len(samples_list[idx]), min_len),
            color=col,
            **c_kwargs,
            hist_kwargs=dict(fc=col, ec=col, **hist_kwargs)
        )

    if len(samples_list) == 1:
        _add_ci_vals_to_marginalised_posteriors(fig, params, samples_list[0])

    if len(sample_labels) > 1:
        plt.legend(
            handles=[
                mlines.Line2D(
                    [], [], color=samples_colors[i], label=sample_labels[i]
                )
                for i in range(len(sample_labels))
            ],
            fontsize=18,
            frameon=False,
            bbox_to_anchor=(1, ndim),
            loc="upper right",
        )
    if title:
        fig.suptitle(title, y=0.97)
        fig.subplots_adjust(top=0.75)
    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig


def _get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (
            len_of_longest_samples / len_current_samples
    )


def overplot_sample(fig, data, color):
    overplot_lines(fig, data, color=color)
    overplot_points(fig, [data, data], color=color)



from corner import corner
import numpy as np

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    color="tab:blue",
    truth_color="tab:orange",
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9.0 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)
LABELS = dict(
    q=r"$q$",
    xeff=r"$\chi_{\rm eff}$",
    a_1=r"$a_1$",
    a_2=r"$a_2$",
    cos_tilt_1=r"$\cos \theta_1$",
    cos_tilt_2=r"$\cos \theta_2$",
)


def plot_corner(df, fname="corner.png"):
    labels = [LABELS.get(i, i.replace("_", "")) for i in df.columns.values]
    fig = corner(df, labels=labels, **CORNER_KWARGS)
    fig.savefig(fname)
"""Gets two q-xeff points for deeper follow-up"""
import numpy as np
import bilby
import glob
import pandas as pd
import corner
from corner import overplot_lines

np.random.seed(100)

PARAM = ['chi_eff', 'mass_ratio']

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color="tab:orange",
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=False,
)


def load_result(filename=glob.glob("out*/*.json")[0]):
    r = bilby.gw.result.CBCResult.from_json(filename)
    r.posteror = r.posterior[PARAM]
    return r


def get_MaL_point(res):
    res.posterior = res.posterior.sort_values(by='log_likelihood')
    return res.posterior[PARAM].iloc[-1].to_dict()


def get_random_pt(res):
    return res.posterior[PARAM].sample(1).to_dict('records')[0]

def get_specific_pt(res):
    return {'chi_eff': 0.2461, 'mass_ratio': 0.3923}

def plot_corner(r, pts):
    data = r.posterior[PARAM].sample(1000)
    fig = corner.corner(
        data, color="C0", **CORNER_KWARGS,
        range=[(0,0.5), (0.1, 0.75)],
        labels=[r"$\chi_{\rm eff}$", r"$q$"]
        )
    for i, pt in enumerate(pts):
        overplot_lines(fig, [v for v in pt.values()], color=f"C{i + 1}")
    fig.savefig("follouwp_params.png")


def main():
    r = load_result()
    pt1 = get_MaL_point(r)
    pt2 = get_specific_pt(r)
    pts = [pt1, pt2]
    data = {k:[] for k in PARAM}
    for p in pts:
        for k in PARAM:
            data[k].append(p[k])
    data['label'] = ['MaL', 'Random']
    pd.DataFrame(data).to_csv("followup_pts.csv", index=False)
    plot_corner(r, pts)


if __name__ == '__main__':
    main()

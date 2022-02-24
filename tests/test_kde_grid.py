from deep_gw_pe_followup.plotting import plot_probs, plot_corner
from deep_gw_pe_followup.sample_cacher.kde2d import get_kde, evaluate_kde_on_grid

import pandas as pd
import numpy as np
import os

CLEAN = False
OUTDIR = "outdir_plot"
os.makedirs(OUTDIR, exist_ok=True)

def generate_data(n):
    x = np.random.normal(scale=1,size=n)
    y = np.random.normal(scale=5, size=n)
    df = pd.DataFrame(dict(x=x, y=y))
    kde = get_kde(df.x, df.y)
    x, y, z = evaluate_kde_on_grid(kde, df.x, df.y, num_gridpoints=100j)
    kde_df = pd.DataFrame(dict(x=x, y=y, p=z))
    return kde_df


def test_kde():
    kde_df = generate_data(int(1e4))
    sum_p = np.sum(kde_df.p)
    np.testing.assert_almost_equal(sum_p, 1.0, decimal=2, err_msg=f"sum(p)={sum_p}")
    plot_probs(kde_df.x, kde_df.y, kde_df.p, "x", "y", fname=f"{OUTDIR}/kde_grid.png")

if __name__ == '__main__':
    test_kde()
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use(
    'https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle')

DATA_KEY = "probabilities"

CMAP = "hot"

def store_probabilities(df: pd.DataFrame, fname: str):
    assert ".h5" in fname, f"{fname} is invalid"
    if os.path.isfile(fname):
        print(f"{fname} exsits. Overwritting with newly computed values.")
        os.remove(fname)
    df = clean_df(df)
    store = pd.HDFStore(fname)
    store.append(key=DATA_KEY, value=df, format="t", data_columns=True)
    store.close()


def clean_df(df):
    df = df.drop_duplicates(keep='last')
    df = df.fillna(0)
    return df


def load_probabilities(fname) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_hdf(fname, key=DATA_KEY)
    df = clean_df(df)
    return df


def plot_probs(x, y, p, xlabel, ylabel, plabel, fname):
    plt.close('all')
    fig, axes = plt.subplots(2,1, figsize=(4, 8))
    ax = axes[0]
    if isinstance(p, pd.Series):
        z = p.values
    else:
        z = p.copy()
    z[z == -np.inf] = np.nan
    try:
        # p = np.nan_to_num(p)
        ax.tricontour(x, y, z, 15, linewidths=0.5, colors='k')
        cmap = ax.tricontourf(
            x, y, z, 15,
            vmin=np.nanmin(z), vmax=np.nanmax(z),
            # norm=plt.Normalize(vmax=abs(p).max(), vmin=-abs(p).max()),
            cmap=CMAP
        )
    except Exception:
        cmap = ax.scatter(x, y, c=p, cmap=CMAP)
    # ax.colorbar(cmap, label=plabel)

    x = np.unique(x.values)
    y = np.unique(y.values)
    X, Y = np.meshgrid(x, y,indexing = 'ij')

    Z = z.reshape(len(y), len(x))

    ax = axes[1]
    ax.pcolor(X, Y, Z, cmap=CMAP, vmin=np.nanmin(z), vmax=np.nanmax(z))


    for ax in axes:
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    # fig.tight_layout()
    # fig.savefig(fname)

    return fig, axes

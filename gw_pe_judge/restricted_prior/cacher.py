import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use(
    'https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle')

DATA_KEY = "probabilities"


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
    try:
        p = np.nan_to_num(p)
        plt.tricontour(x, y, p, 15, linewidths=0.5, colors='k')
        cmap = plt.tricontourf(
            x, y, p, 15,
            norm=plt.Normalize(vmax=abs(p).max(), vmin=-abs(p).max()),
            cmap='inferno'
        )
    except Exception:
        cmap = plt.scatter(x, y, c=p, cmap='inferno')
    plt.colorbar(cmap, label=plabel)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(fname)

import os

import pandas as pd

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

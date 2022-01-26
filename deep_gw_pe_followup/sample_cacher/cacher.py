import os
from typing import Optional
import pandas as pd
from tqdm.auto import tqdm
import glob

DATA_KEY = "probabilities"


def store_probabilities(df: pd.DataFrame, fname: str, append: Optional[bool] = False,
                        no_duplicates: Optional[bool] = True):
    assert ".h5" in fname, f"{fname} is invalid"
    if no_duplicates:
        df = clean_df(df)
    store = pd.HDFStore(fname)
    store.append(key=DATA_KEY, value=df, format="t", data_columns=True, append=append)
    store.close()


def clean_df(df):
    df = df.drop_duplicates(keep='last')
    df = df.fillna(0)
    return df


def load_probabilities(fname) -> pd.DataFrame:
    df = pd.read_hdf(fname, key=DATA_KEY)
    return df

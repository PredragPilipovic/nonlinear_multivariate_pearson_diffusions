"""
data_loading.py — Download, cache, and preprocess the Greenland Ca2+ ice core series
"""
import os
import numpy as np
import pandas as pd
import requests

from config_app import DATA_URL, DATA_CACHE, AGE_PREFILTER, AGE_WINDOW


def load_ca2(cache_path: str = DATA_CACHE) -> pd.DataFrame:
    """
    Return a preprocessed DataFrame with columns [age_ka, X_Ca2].
    Downloads and caches the raw Excel file on first call.
    """
    if os.path.exists(cache_path):
        raw = pd.read_csv(cache_path)
    else:
        raw = _download_and_parse(cache_path)
    return _preprocess(raw)


def build_data_matrix(ca2_series: np.ndarray, h: float):
    """
    Build the (N-1, 2) observation matrix [X_t, (X_{t+1} - X_t)/h].
    ca2_series must already be time-reversed (oldest observation first).
    """
    import jax.numpy as jnp
    return jnp.column_stack((ca2_series[:-1], np.diff(ca2_series) / h))


def _download_and_parse(cache_path: str) -> pd.DataFrame:
    tmp = "temp_ice_data.xlsx"
    with open(tmp, "wb") as f:
        f.write(requests.get(DATA_URL).content)
    df = pd.read_excel(tmp, sheet_name=2, skiprows=49, usecols=[0, 8])
    df.columns = ["age", "Ca2"]
    df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.to_csv(cache_path, index=False)
    os.remove(tmp)
    return df


def _preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[(raw["age"] >= AGE_PREFILTER[0]) & (raw["age"] <= AGE_PREFILTER[1])].copy()
    df["Ca2"] = df["Ca2"].interpolate(method="linear")
    df = df.groupby("age", as_index=False)["Ca2"].mean()
    df["age_ka"] = df["age"] / 1000
    df["Ca2"]    = -np.log(df["Ca2"])
    df = df[(df["age"] >= AGE_WINDOW[0]) & (df["age"] <= AGE_WINDOW[1])].copy()
    df["age"]    = df["age"] / 1000
    df["X_Ca2"]  = df["Ca2"] - df["Ca2"].mean()
    return df[["age_ka", "X_Ca2"]].reset_index(drop=True)

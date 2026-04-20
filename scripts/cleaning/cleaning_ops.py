"""Pure transformations applied by step1b to the dirty DataFrame."""
from __future__ import annotations

import pandas as pd

from lookup_store import TEST_USER_IDS
from url_helpers import build_wikipedia_url, normalise_timestamp


def filter_test_users(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["ID"].isin(TEST_USER_IDS)].copy()


def normalise_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Time"] = out["Time"].apply(normalise_timestamp)
    return out


def merge_revids(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError  # Task 8

def build_wikipedia_urls_col(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError  # Task 8

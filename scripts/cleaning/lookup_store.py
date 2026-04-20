"""Read/write the revid lookup CSV and extract unique (slug, timestamp) pairs."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from url_helpers import normalise_timestamp

LOOKUP_COLUMNS = ["slug", "timestamp", "revid", "status", "fetched_at"]
TEST_USER_IDS = {69, 70}


def load_lookup(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        df = pd.DataFrame(columns=LOOKUP_COLUMNS)
        df["revid"] = df["revid"].astype("Int64")
        return df
    df = pd.read_csv(path, dtype={"slug": str, "timestamp": str, "status": str, "fetched_at": str})
    for col in LOOKUP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["revid"] = pd.to_numeric(df["revid"], errors="coerce").astype("Int64")
    return df[LOOKUP_COLUMNS]


def save_lookup(df: pd.DataFrame, path) -> None:
    missing = set(LOOKUP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"save_lookup: DataFrame missing columns: {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[LOOKUP_COLUMNS].to_csv(path, index=False)


def extract_unique_pairs(dirty_df: pd.DataFrame) -> List[Tuple[str, str]]:
    mask = (dirty_df["Action"] == "article_open") & (~dirty_df["ID"].isin(TEST_USER_IDS))
    sub = dirty_df.loc[mask, ["ArticleSlug", "Time"]].dropna()
    pairs = {
        (str(row["ArticleSlug"]), normalise_timestamp(row["Time"]))
        for _, row in sub.iterrows()
    }
    return sorted(pairs)

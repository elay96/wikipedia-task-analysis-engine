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
    """Left-join ok-status revids from `lookup` onto article_open rows in `df`.

    Matches on ArticleSlug==lookup.slug AND Time==lookup.timestamp.
    Only lookup rows with status=='ok' are used; others leave ArticleRevid null.
    """
    out = df.copy()
    ok = lookup[lookup["status"] == "ok"][["slug", "timestamp", "revid"]]
    merged = out.merge(
        ok.rename(columns={"slug": "ArticleSlug", "timestamp": "Time", "revid": "_revid_merged"}),
        on=["ArticleSlug", "Time"],
        how="left",
    )
    mask_article_open = merged["Action"] == "article_open"
    merged.loc[mask_article_open & merged["_revid_merged"].notna(), "ArticleRevid"] = \
        merged.loc[mask_article_open & merged["_revid_merged"].notna(), "_revid_merged"]
    return merged.drop(columns=["_revid_merged"])


def build_wikipedia_urls_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = out["ArticleRevid"].notna() & out["ArticleSlug"].notna()
    out.loc[mask, "WikipediaUrl"] = [
        build_wikipedia_url(slug, int(rev))
        for slug, rev in zip(out.loc[mask, "ArticleSlug"], out.loc[mask, "ArticleRevid"])
    ]
    return out

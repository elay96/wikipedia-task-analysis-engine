"""Read/write articles.jsonl and extract unique revid records from cleaned CSV."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set

import pandas as pd

ARTICLE_FIELDS = ("article_slug", "revid", "timestamp", "content")


def load_existing_revids(path) -> Set[int]:
    """Return the set of revids already present in an articles.jsonl file.

    Returns an empty set if the file does not exist or is empty.
    """
    p = Path(path)
    if not p.exists():
        return set()
    revids: Set[int] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            revids.add(int(record["revid"]))
    return revids


def append_article(path, record: dict) -> None:
    """Append a single article record as one JSON line. Creates parent dir if needed.

    Record must have exactly the fields in ARTICLE_FIELDS.
    """
    missing = set(ARTICLE_FIELDS) - set(record.keys())
    if missing:
        raise ValueError(f"append_article: record missing fields: {sorted(missing)}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_unique_revid_records(cleaned_df: pd.DataFrame) -> List[dict]:
    """From a cleaned Game.csv DataFrame, emit one record per unique non-null revid.

    Each record has article_slug + revid + timestamp (the earliest Time across all
    article_open rows with that revid). Output is sorted by revid ascending.
    """
    mask = (cleaned_df["Action"] == "article_open") & cleaned_df["ArticleRevid"].notna()
    sub = cleaned_df.loc[mask, ["ArticleSlug", "ArticleRevid", "Time"]].copy()
    sub["ArticleRevid"] = sub["ArticleRevid"].astype("int64")
    sub = sub.sort_values("Time")
    grouped = sub.groupby("ArticleRevid", sort=True, as_index=False).first()

    records: List[dict] = []
    for _, row in grouped.iterrows():
        records.append({
            "article_slug": str(row["ArticleSlug"]),
            "revid": int(row["ArticleRevid"]),
            "timestamp": str(row["Time"]),
        })
    return records

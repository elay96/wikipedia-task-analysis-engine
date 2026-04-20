"""Validation gate: asserts the cleaned CSV satisfies our invariants."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from lookup_store import TEST_USER_IDS

MISSING_REVID_TOLERANCE = 0.05  # 5%
URL_PATTERN = re.compile(r"^https://en\.wikipedia\.org/wiki/.+\?oldid=\d+$")


class ValidationError(Exception):
    pass


def validate_cleaned(df: pd.DataFrame, *, original_row_count: int, removed_rows: int) -> None:
    if df["ID"].isin(TEST_USER_IDS).any():
        raise ValidationError("Cleaned DataFrame still contains test user rows (IDs 69/70).")

    expected = original_row_count - removed_rows
    if len(df) != expected:
        raise ValidationError(
            f"Unexpected row count: got {len(df)}, expected {expected} "
            f"(original={original_row_count} - removed={removed_rows})."
        )

    is_practice = pd.to_numeric(df["IsPractice"], errors="coerce") == 1
    is_article_open = df["Action"] == "article_open"
    nonprac_opens = df[is_article_open & ~is_practice]
    if len(nonprac_opens) > 0:
        missing = nonprac_opens["ArticleRevid"].isna().sum()
        rate = missing / len(nonprac_opens)
        if rate > MISSING_REVID_TOLERANCE:
            raise ValidationError(
                f"{missing}/{len(nonprac_opens)} non-practice article_open rows missing "
                f"ArticleRevid ({rate:.1%}) exceeds tolerance ({MISSING_REVID_TOLERANCE:.0%})."
            )

    urls = df["WikipediaUrl"].dropna()
    bad = urls[~urls.apply(lambda u: bool(URL_PATTERN.match(str(u))))]
    if len(bad) > 0:
        sample = bad.iloc[0]
        raise ValidationError(f"{len(bad)} rows have malformed URL (WikipediaUrl); e.g. {sample!r}")


def validate_articles(cleaned_df: pd.DataFrame, articles_path) -> None:
    """Assert every unique non-null revid in the cleaned CSV is represented by a
    non-empty content line in the articles JSONL.
    """
    p = Path(articles_path)
    if not p.exists():
        raise ValidationError(f"articles file does not exist: {p}")

    required = set()
    mask = (cleaned_df["Action"] == "article_open") & cleaned_df["ArticleRevid"].notna()
    for r in cleaned_df.loc[mask, "ArticleRevid"]:
        required.add(int(r))

    present: dict = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                present[int(rec["revid"])] = rec.get("content") or ""
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue  # skip corrupted/partial lines — validation reports missing

    missing = sorted(required - set(present.keys()))
    if missing:
        sample = missing[:5]
        raise ValidationError(
            f"articles JSONL missing {len(missing)} revids (first 5: {sample})"
        )

    empties = sorted(r for r in required if not present[r].strip())
    if empties:
        sample = empties[:5]
        raise ValidationError(
            f"articles JSONL has empty content for {len(empties)} revids "
            f"(first 5: {sample})"
        )

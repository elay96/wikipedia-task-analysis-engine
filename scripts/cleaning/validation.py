"""Validation gate: asserts the cleaned CSV satisfies our invariants."""
from __future__ import annotations

import re

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

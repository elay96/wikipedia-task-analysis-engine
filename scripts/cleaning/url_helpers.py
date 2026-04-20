"""Pure helpers for URL construction and timestamp normalisation."""
from __future__ import annotations

import re

import pandas as pd

_URL_TEMPLATE = "https://en.wikipedia.org/wiki/{slug}?oldid={revid}"

# Matches strings that have an explicit timezone marker (Z or +/-HH:MM or UTC)
_TZ_PATTERN = re.compile(r"(Z|[+-]\d{2}:?\d{2}|UTC)$")


def build_wikipedia_url(slug: str, revid: int) -> str:
    return _URL_TEMPLATE.format(slug=slug, revid=int(revid))


def normalise_timestamp(raw) -> str:
    if isinstance(raw, str) and not _TZ_PATTERN.search(raw):
        raise ValueError(f"Naive timestamp rejected: {raw!r}")
    if isinstance(raw, pd.Timestamp) and raw.tzinfo is None:
        raise ValueError(f"Naive timestamp rejected: {raw!r}")
    ts = pd.to_datetime(raw, utc=True, errors="raise")
    ms = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"
    return ms

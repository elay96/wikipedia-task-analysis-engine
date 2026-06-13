"""Resolve and cache the historical Wikipedia revision each visit saw.

Reuses the repo's api_client (revision-at-timestamp resolver + extracts +
outlinks). Adds a current-revision category fetch (categories are stable, so
this approximates the historical category set; documented in the report)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from api_client import (  # from scripts/cleaning
    API_ENDPOINT, TIMEOUT_SEC, build_session, fetch_extract_by_revid,
    fetch_outlinks, fetch_revision_at,
)


def epoch_ms_to_iso(ms) -> str | None:
    """Epoch milliseconds -> ISO-8601 UTC string suitable for rvstart."""
    if ms is None or (isinstance(ms, float) and pd.isna(ms)):
        return None
    dt = datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def cache_path(cache_dir, revid: int) -> Path:
    return Path(cache_dir) / f"{revid}.json"


def fetch_categories(session, slug: str) -> list:
    """Visible (non-hidden) categories for the current revision of `slug`."""
    params = {
        "action": "query", "prop": "categories", "titles": slug,
        "clshow": "!hidden", "cllimit": "max", "redirects": 1,
        "format": "json", "formatversion": 1,
    }
    try:
        resp = session.get(API_ENDPOINT, params=params, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        pages = (resp.json().get("query") or {}).get("pages") or {}
        cats = []
        for _, page in pages.items():
            for c in page.get("categories") or []:
                cats.append(c.get("title", "").replace("Category:", ""))
        return cats
    except Exception:
        return []

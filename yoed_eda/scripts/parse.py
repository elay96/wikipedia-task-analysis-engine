"""Flatten the wiki_behavior CSV into tidy participant/visit/search tables."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

PROXY_PREFIX = "/proxy/wiki/wiki/"
MEASURE_AND_META_DROP = ["page_visits_rows_json", "search_queries_rows_json"]
ISO_COLS = ["started_at", "ended_at"]
DISAMBIG_RE = re.compile(r"\(disambiguation\)", re.IGNORECASE)


def clean_iso(value) -> str | float:
    """Strip wrapping quote characters from a CSV-embedded ISO timestamp."""
    if not isinstance(value, str):
        return value
    return value.strip().strip('"').strip()


def decode_article(url: str) -> str:
    """Map a proxied URL to the bare article slug."""
    if not isinstance(url, str):
        return ""
    if PROXY_PREFIX in url:
        return url.split(PROXY_PREFIX, 1)[1]
    return url.rsplit("/", 1)[-1]


def _iso_to_ms(iso: str):
    if not isinstance(iso, str) or not iso:
        return np.nan
    ts = pd.to_datetime(iso, utc=True, errors="coerce")
    if pd.isna(ts):
        return np.nan
    return ts.value // 1_000_000  # ns -> ms


def flatten_visits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        try:
            visits = json.loads(r["page_visits_rows_json"])
        except (TypeError, ValueError):
            continue
        if not visits:
            continue
        visits = sorted(visits, key=lambda v: v.get("start_time", 0))
        end_ms = _iso_to_ms(clean_iso(r.get("ended_at")))
        for i, v in enumerate(visits):
            slug = decode_article(str(v.get("url", "")))
            start = v.get("start_time")
            nxt = visits[i + 1]["start_time"] if i + 1 < len(visits) else end_ms
            dwell = (nxt - start) if (pd.notna(nxt) and pd.notna(start)) else np.nan
            rows.append({
                "participant_id": r["participant_id"],
                "session_id": r.get("session_id"),
                "visit_id": v.get("id"),
                "order_in_session": i,
                "article": slug,
                "title": v.get("title"),
                "url": v.get("url"),
                "start_time_ms": start,
                "dwell_ms": dwell,
                "is_main_page": slug == "Main_Page",
                "is_disambiguation": bool(DISAMBIG_RE.search(slug)),
            })
    return pd.DataFrame(rows)


def flatten_searches(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        try:
            queries = json.loads(r["search_queries_rows_json"])
        except (TypeError, ValueError):
            continue
        for q in queries:
            rows.append({
                "participant_id": r["participant_id"],
                "session_id": r.get("session_id"),
                "query": q.get("query"),
                "timestamp_ms": q.get("timestamp"),
                "results_count": q.get("results_count"),
            })
    return pd.DataFrame(rows)


def participants_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[c for c in MEASURE_AND_META_DROP if c in df.columns]).copy()
    for c in ISO_COLS:
        if c in out.columns:
            out[c] = out[c].map(clean_iso)
    return out

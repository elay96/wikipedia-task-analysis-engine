#!/usr/bin/env python3
"""Stage 1a: resolve MediaWiki revision IDs for every unique (slug, timestamp)
pair found in article_open rows of the dirty Game.csv.

Usage:
    py scripts/cleaning/step1a_resolve_revids.py
    py scripts/cleaning/step1a_resolve_revids.py --dirty path/to/dirty.csv --lookup path/to/lookup.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from api_client import build_session, fetch_revision_at
from lookup_store import LOOKUP_COLUMNS, extract_unique_pairs, load_lookup, save_lookup

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DIRTY = PROJECT_ROOT / "data" / "dirty_data" / "Game.csv"
DEFAULT_LOOKUP = PROJECT_ROOT / "data" / "cleaned" / "revid_lookup.csv"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def resolve_all(
    dirty_df: pd.DataFrame,
    lookup_path,
    *,
    session,
    fetch_fn: Callable[[str, str], dict] = None,
) -> pd.DataFrame:
    if fetch_fn is None:
        def fetch_fn(slug, timestamp):
            return fetch_revision_at(session, slug, timestamp)

    existing = load_lookup(lookup_path)
    resolved_ok = set(
        zip(existing.loc[existing["status"] == "ok", "slug"],
            existing.loc[existing["status"] == "ok", "timestamp"])
    )

    pairs = extract_unique_pairs(dirty_df)
    todo = [p for p in pairs if p not in resolved_ok]

    print(f"[step1a] {len(pairs)} unique pairs; {len(pairs) - len(todo)} already resolved; "
          f"{len(todo)} to fetch")

    new_rows = []
    counts = {"ok": 0, "not_found": 0, "error": 0}
    for i, (slug, ts) in enumerate(todo, 1):
        res = fetch_fn(slug, ts)
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        new_rows.append({
            "slug": slug,
            "timestamp": ts,
            "revid": res["revid"],
            "status": res["status"],
            "fetched_at": _now_iso(),
        })
        if i % 50 == 0 or i == len(todo):
            print(f"[step1a] [{i}/{len(todo)}] {slug} @ {ts} -> "
                  f"revid={res['revid']} status={res['status']}")

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=LOOKUP_COLUMNS)
        if existing.empty:
            combined = new_df
        else:
            keys_added = set(zip(new_df["slug"], new_df["timestamp"]))
            keep_mask = ~existing[["slug", "timestamp"]].apply(tuple, axis=1).isin(keys_added)
            combined = pd.concat([existing[keep_mask], new_df], ignore_index=True)
    else:
        combined = existing

    save_lookup(combined, lookup_path)
    print(f"[step1a] summary: ok={counts['ok']} not_found={counts['not_found']} "
          f"error={counts['error']}; total rows in lookup: {len(combined)}")
    return combined


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Resolve Wikipedia revids for article_open rows.")
    parser.add_argument("--dirty", type=Path, default=DEFAULT_DIRTY)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    args = parser.parse_args(argv)

    dirty_df = pd.read_csv(args.dirty, low_memory=False)
    session = build_session()
    resolve_all(dirty_df, args.lookup, session=session)
    return 0


if __name__ == "__main__":
    sys.exit(_main())

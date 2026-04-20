#!/usr/bin/env python3
"""Stage 2: fetch plain-text article content for every unique revid in the cleaned
Game.csv and write one JSON object per line to data/cleaned/articles.jsonl.

Usage:
    py scripts/cleaning/step2_fetch_articles.py
    py scripts/cleaning/step2_fetch_articles.py --cleaned X --out Y
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

from api_client import build_session, fetch_extract_by_revid
from article_store import (
    append_article,
    extract_unique_revid_records,
    load_existing_revids,
)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CLEANED = PROJECT_ROOT / "data" / "cleaned" / "Game.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "cleaned" / "articles.jsonl"


def resolve_all_articles(
    cleaned_df: pd.DataFrame,
    out_path,
    *,
    session,
    fetch_fn: Callable[[int], dict] = None,
) -> Dict[str, int]:
    if fetch_fn is None:
        def fetch_fn(revid):
            return fetch_extract_by_revid(session, revid)

    all_records = extract_unique_revid_records(cleaned_df)
    already = load_existing_revids(out_path)
    todo = [r for r in all_records if r["revid"] not in already]

    print(f"[step2] {len(all_records)} unique revids; {len(already)} already fetched; "
          f"{len(todo)} to fetch")

    counts = {"ok": 0, "not_found": 0, "error": 0, "skipped": len(already)}
    for i, record in enumerate(todo, 1):
        res = fetch_fn(record["revid"])
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        if res["status"] == "ok":
            full = {**record, "content": res["content"]}
            append_article(out_path, full)
        if i % 50 == 0 or i == len(todo):
            print(f"[step2] [{i}/{len(todo)}] revid={record['revid']} "
                  f"slug={record['article_slug']} status={res['status']}")

    print(f"[step2] summary: ok={counts['ok']} not_found={counts['not_found']} "
          f"error={counts['error']} skipped={counts['skipped']}")
    return counts


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch article extracts per unique revid.")
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    cleaned = pd.read_csv(args.cleaned, low_memory=False)
    cleaned["ArticleRevid"] = pd.to_numeric(cleaned["ArticleRevid"], errors="coerce").astype("Int64")
    session = build_session()
    resolve_all_articles(cleaned, args.out, session=session)
    return 0


if __name__ == "__main__":
    sys.exit(_main())

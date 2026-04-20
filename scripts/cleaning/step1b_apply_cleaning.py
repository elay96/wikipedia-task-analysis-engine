#!/usr/bin/env python3
"""Stage 1b: apply cleaning to the dirty Game.csv using revid_lookup.csv.

Usage:
    py scripts/cleaning/step1b_apply_cleaning.py
    py scripts/cleaning/step1b_apply_cleaning.py --dirty X --lookup Y --out Z
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from cleaning_ops import (
    build_wikipedia_urls_col,
    filter_test_users,
    merge_revids,
    normalise_timestamp_col,
)
from lookup_store import TEST_USER_IDS, load_lookup
from validation import validate_cleaned

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DIRTY = PROJECT_ROOT / "data" / "dirty_data" / "Game.csv"
DEFAULT_LOOKUP = PROJECT_ROOT / "data" / "cleaned" / "revid_lookup.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "cleaned" / "Game.csv"


def apply_cleaning(dirty: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    original_columns = list(dirty.columns)
    original_len = len(dirty)

    removed = int(dirty["ID"].isin(TEST_USER_IDS).sum())

    df = filter_test_users(dirty)
    df = normalise_timestamp_col(df)
    df = merge_revids(df, lookup)
    df = build_wikipedia_urls_col(df)

    df = df[original_columns].reset_index(drop=True)

    validate_cleaned(df, original_row_count=original_len, removed_rows=removed)
    return df


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Apply cleaning to produce cleaned Game.csv.")
    parser.add_argument("--dirty", type=Path, default=DEFAULT_DIRTY)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    dirty = pd.read_csv(args.dirty, low_memory=False)
    lookup = load_lookup(args.lookup)

    cleaned = apply_cleaning(dirty, lookup)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False)
    print(f"[step1b] wrote {len(cleaned)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

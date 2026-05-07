#!/usr/bin/env python3
"""One-shot export of the spatial-search BSON dumps to Game.csv + KeyTable.csv,
keyed by the wikipedia integer participant ID.

Inputs:
    data/Spatial Search Data/1/spatial_search_users_records.bson
    data/Spatial Search Data/2/spatial_search_users_records.bson
    data/Spatial Search Data/KeyTable.csv  (wiki backend KeyTable)

Outputs:
    data/cleaned/spatial_search/Game.csv
    data/cleaned/spatial_search/KeyTable.csv

Usage:
    py scripts/cleaning/spatial_search_export.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import bson
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SS_DIR = PROJECT_ROOT / "data" / "Spatial Search Data"
BSON_PATHS = [
    SS_DIR / "1" / "spatial_search_users_records.bson",
    SS_DIR / "2" / "spatial_search_users_records.bson",
]
WIKI_KEYTABLE_PATH = SS_DIR / "KeyTable.csv"

OUT_DIR = PROJECT_ROOT / "data" / "cleaned" / "spatial_search"
OUT_GAME = OUT_DIR / "Game.csv"
OUT_KEYTABLE = OUT_DIR / "KeyTable.csv"

EXPECTED_VERSION = "Wikipedia_Exp_13.04"
EXPECTED_RUNNING_NAME = "Wikipedia_Exp_14.04"
WIKI_TEST_VERSION = "Updated Test"

GAME_COLUMNS = [
    "ID", "RunningName", "Version", "Age", "Gender",
    "GameCondition", "Trial", "Action", "Time",
    "X", "Y", "Heading", "Speed",
    "MapID", "ResourceX", "ResourceY",
    "TotalCollected", "CollisionCount",
]

KEYTABLE_COLUMNS = [
    "ID", "UserID", "RunningName", "Version", "Mode", "Date",
    "SpatialSearchID",
    "SpatialSearchClientStartTime", "SpatialSearchClientEndTime",
    "SpatialSearchSummaryStartTime", "SpatialSearchSummaryEndTime",
    "SpatialSearchBonusPayment",
]


def load_wiki_keytable(path: Path) -> pd.DataFrame:
    """Load the wiki backend KeyTable and filter to real Wikipedia_Exp_14.04 sessions.

    Drops rows whose Version is 'Updated Test' (elay96 IDs 69, 70).
    Asserts no duplicate (ID, UserID) rows remain.
    """
    df = pd.read_csv(path)
    df = df[df["RunningName"] == EXPECTED_RUNNING_NAME].copy()
    df = df[df["Version"] != WIKI_TEST_VERSION].copy()
    assert df["ID"].is_unique, f"duplicate IDs in wiki KeyTable: {df[df['ID'].duplicated()]}"
    assert df["UserID"].is_unique, f"duplicate UserIDs in wiki KeyTable: {df[df['UserID'].duplicated()]}"
    return df.reset_index(drop=True)


def load_spatial_records(paths: list[Path]) -> list[dict]:
    """Load and concatenate the spatial-search BSON dumps."""
    records: list[dict] = []
    for p in paths:
        with open(p, "rb") as f:
            records += bson.decode_all(f.read())
    return records


def filter_to_wiki_sessions(records: list[dict]) -> list[dict]:
    """Keep only the spatial-search sessions belonging to the wikipedia run."""
    return [
        d for d in records
        if d.get("Version") == EXPECTED_VERSION
        and d.get("RunningName") == EXPECTED_RUNNING_NAME
    ]


def build_uid_lookup(records: list[dict]) -> dict[str, dict]:
    """Map Prolific UserId -> spatial record. Hard-fail on any duplicates."""
    lookup: dict[str, dict] = {}
    for d in records:
        uid = d.get("UserId")
        if uid in lookup:
            raise AssertionError(
                f"duplicate UserId {uid!r} in spatial-search records "
                f"(this should not happen for Wikipedia_Exp_13.04 + Wikipedia_Exp_14.04)"
            )
        lookup[uid] = d
    return lookup


def assert_full_match(wiki_kt: pd.DataFrame, lookup: dict[str, dict]) -> None:
    """Every wiki UID must appear exactly once in the spatial-search lookup."""
    missing = set(wiki_kt["UserID"]) - set(lookup.keys())
    if missing:
        raise AssertionError(
            f"{len(missing)} wiki participants missing from spatial-search BSON: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )


def event_row(event: dict, wiki_id: int) -> dict[str, Any]:
    """Project one BSON event onto the canonical Game.csv schema.

    The integer ID inside the BSON event payload is the spatial-search internal
    counter; we override it with the wikipedia ID so the output joins cleanly
    against data/cleaned_new/Game.csv.
    """
    return {
        "ID": wiki_id,
        "RunningName": event.get("RunningName"),
        "Version": event.get("Version"),
        "Age": event.get("Age"),
        "Gender": event.get("Gender"),
        "GameCondition": event.get("GameCondition"),
        "Trial": event.get("Trial"),
        "Action": event.get("Action"),
        "Time": event.get("Time"),
        "X": event.get("X"),
        "Y": event.get("Y"),
        "Heading": event.get("Heading"),
        "Speed": event.get("Speed"),
        "MapID": event.get("MapID"),
        "ResourceX": event.get("ResourceX"),
        "ResourceY": event.get("ResourceY"),
        "TotalCollected": event.get("TotalCollected"),
        "CollisionCount": event.get("CollisionCount"),
    }


def build_game_df(wiki_kt: pd.DataFrame, lookup: dict[str, dict]) -> pd.DataFrame:
    """Flatten every spatial-search session's Records.Game[] into one DataFrame."""
    rows: list[dict[str, Any]] = []
    for _, kt_row in wiki_kt.iterrows():
        wiki_id = int(kt_row["ID"])
        rec = lookup[kt_row["UserID"]]
        events = (rec.get("Records") or {}).get("Game") or []
        for ev in events:
            if isinstance(ev, dict):
                rows.append(event_row(ev, wiki_id))
    df = pd.DataFrame(rows, columns=GAME_COLUMNS)
    df = df.sort_values(["ID", "Time"], kind="stable").reset_index(drop=True)
    return df


def build_keytable_df(wiki_kt: pd.DataFrame, lookup: dict[str, dict]) -> pd.DataFrame:
    """One row per wiki participant, with spatial-search session metadata appended."""
    rows: list[dict[str, Any]] = []
    for _, kt_row in wiki_kt.iterrows():
        rec = lookup[kt_row["UserID"]]
        records_block = rec.get("Records") or {}
        ss_kt = records_block.get("KeyTable") or {}
        summary = records_block.get("Summary") or {}
        payment = records_block.get("Payment") or {}
        rows.append({
            "ID": int(kt_row["ID"]),
            "UserID": kt_row["UserID"],
            "RunningName": kt_row["RunningName"],
            "Version": kt_row["Version"],
            "Mode": kt_row["Mode"],
            "Date": kt_row["Date"],
            "SpatialSearchID": ss_kt.get("ID"),
            "SpatialSearchClientStartTime": rec.get("ClientStartTime"),
            "SpatialSearchClientEndTime": rec.get("ClientEndTime"),
            "SpatialSearchSummaryStartTime": summary.get("StartTime"),
            "SpatialSearchSummaryEndTime": summary.get("EndTime"),
            "SpatialSearchBonusPayment": payment.get("bonus_payment"),
        })
    df = pd.DataFrame(rows, columns=KEYTABLE_COLUMNS)
    return df.sort_values("ID", kind="stable").reset_index(drop=True)


def validate_outputs(game_df: pd.DataFrame, keytable_df: pd.DataFrame,
                     wiki_kt: pd.DataFrame, lookup: dict[str, dict]) -> None:
    """Print a PASS/FAIL report on structural invariants. Hard-fail on red checks."""
    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "KeyTable row count == 163",
        len(keytable_df) == 163,
        f"got {len(keytable_df)}",
    ))

    game_ids = set(game_df["ID"].unique())
    kt_ids = set(keytable_df["ID"].unique())
    checks.append((
        "every Game.csv ID is present in KeyTable.csv",
        game_ids.issubset(kt_ids),
        f"orphans: {sorted(game_ids - kt_ids)[:5]}",
    ))

    expected_total = sum(
        len((lookup[uid].get("Records") or {}).get("Game") or [])
        for uid in wiki_kt["UserID"]
    )
    checks.append((
        "Game.csv row count matches sum of consumed BSON Records.Game lengths",
        len(game_df) == expected_total,
        f"got {len(game_df)}, expected {expected_total}",
    ))

    # Each participant should be in exactly one experimental condition.
    real_cond = game_df[game_df["GameCondition"].isin(["Diffuse", "Clumpy"])]
    cond_per_pid = real_cond.groupby("ID")["GameCondition"].nunique()
    multi_cond = cond_per_pid[cond_per_pid > 1]
    checks.append((
        "each participant has events from exactly one of {Diffuse, Clumpy}",
        len(multi_cond) == 0,
        f"{len(multi_cond)} participants with both: {multi_cond.head().to_dict()}",
    ))

    # Most participants should have 5 Round_end events. Soft warning otherwise.
    round_ends = (
        game_df[game_df["Action"] == "Round_end"]
        .groupby("ID")
        .size()
    )
    expected_round_ends = 5
    bad_pids = round_ends[round_ends != expected_round_ends]
    if len(bad_pids):
        print(f"  [warn] {len(bad_pids)}/163 participants do not have exactly "
              f"{expected_round_ends} Round_end events: "
              f"{bad_pids.value_counts().to_dict()}")
    else:
        print(f"  trial structure: every participant has {expected_round_ends} Round_end events")

    print("\nValidation:")
    failed = []
    for name, ok, detail in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}" + (f" -- {detail}" if not ok else ""))
        if not ok:
            failed.append(name)
    if failed:
        raise AssertionError(f"{len(failed)} validation check(s) failed: {failed}")


def main() -> int:
    print("[spatial_search_export] starting")
    wiki_kt = load_wiki_keytable(WIKI_KEYTABLE_PATH)
    print(f"  wiki KeyTable: {len(wiki_kt)} real Wikipedia_Exp_14.04 participants")

    raw_records = load_spatial_records(BSON_PATHS)
    print(f"  spatial-search BSON: {len(raw_records)} raw records "
          f"(across {len(BSON_PATHS)} dumps)")

    wiki_records = filter_to_wiki_sessions(raw_records)
    print(f"  filtered to wikipedia sessions: {len(wiki_records)} records")

    lookup = build_uid_lookup(wiki_records)
    assert_full_match(wiki_kt, lookup)
    print(f"  Prolific UserId lookup: {len(lookup)} entries, full match against wiki KeyTable")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    game_df = build_game_df(wiki_kt, lookup)
    game_df.to_csv(OUT_GAME, index=False, encoding="utf-8")
    print(f"  wrote {OUT_GAME.relative_to(PROJECT_ROOT)} ({len(game_df):,} rows, "
          f"{game_df['ID'].nunique()} participants)")

    keytable_df = build_keytable_df(wiki_kt, lookup)
    keytable_df.to_csv(OUT_KEYTABLE, index=False, encoding="utf-8")
    print(f"  wrote {OUT_KEYTABLE.relative_to(PROJECT_ROOT)} ({len(keytable_df)} rows)")

    validate_outputs(game_df, keytable_df, wiki_kt, lookup)
    print("[spatial_search_export] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

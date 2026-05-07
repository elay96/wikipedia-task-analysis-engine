# Spatial Search Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-shot Python script that converts the two spatial-search MongoDB BSON dumps into a clean `Game.csv` + `KeyTable.csv` pair under `data/cleaned/spatial_search/`, keyed by the wikipedia integer participant `ID`.

**Architecture:** Single self-contained script `scripts/cleaning/spatial_search_export.py`. Loads the two BSON files with `bson.decode_all`, joins to the wiki KeyTable on Prolific UserID, flattens the embedded `Records.Game[]` event arrays, overrides the BSON-side integer ID with the wiki ID, writes two CSVs, and runs an end-of-script validation block. No external Mongo dependency, no Parquet, no analysis-level metrics.

**Tech Stack:** Python 3.13, `pymongo` (only for `bson.decode_all`), `pandas`. All already installed.

**Reference spec:** `docs/superpowers/specs/2026-05-07-spatial-search-export-design.html`.

---

## File Structure

| File | Status | Purpose |
|---|---|---|
| `scripts/cleaning/spatial_search_export.py` | Create | The export script (single file, ~250 lines). |
| `data/cleaned/spatial_search/Game.csv` | Create (output) | Long-format event log, ~154,574 rows. |
| `data/cleaned/spatial_search/KeyTable.csv` | Create (output) | Per-participant lookup, 163 rows. |
| `data/Spatial Search Data/KeyTable.csv` | Already in place | Wiki backend KeyTable (the bridge UID -> wiki ID). |
| `data/Spatial Search Data/{1,2}/spatial_search_users_records.bson` | Already in place | Source BSON dumps. |

The script is intentionally single-file (rather than split like `step1a/step1b/...`) because it has one responsibility (one-shot dump conversion) and the existing `cleaning/` subdirectory uses multi-file decomposition only for the multi-stage wiki article pipeline.

---

## Task 1: Script skeleton and constants

**Files:**
- Create: `scripts/cleaning/spatial_search_export.py`

- [ ] **Step 1: Create the file with the header, constants, and an empty `main()`**

```python
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
WIKI_TEST_VERSION = "Updated Test"  # elay96 IDs 69, 70 — drop these.

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


def main() -> int:
    print("[spatial_search_export] starting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the skeleton and confirm it executes**

Run: `py scripts/cleaning/spatial_search_export.py`
Expected stdout: `[spatial_search_export] starting`
Expected exit code: 0

- [ ] **Step 3: Commit**

```bash
git add scripts/cleaning/spatial_search_export.py
git commit -m "chore(spatial-search): scaffold export script skeleton"
```

---

## Task 2: Load wiki KeyTable + filter to real participants

**Files:**
- Modify: `scripts/cleaning/spatial_search_export.py` (add `load_wiki_keytable` function and call it from `main`)

- [ ] **Step 1: Add the loader and a printed sanity check**

Add right above `main()`:

```python
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
```

Update `main()`:

```python
def main() -> int:
    print("[spatial_search_export] starting")
    wiki_kt = load_wiki_keytable(WIKI_KEYTABLE_PATH)
    print(f"  wiki KeyTable: {len(wiki_kt)} real Wikipedia_Exp_14.04 participants")
    return 0
```

- [ ] **Step 2: Run and verify count is 163**

Run: `py scripts/cleaning/spatial_search_export.py`
Expected stdout includes: `wiki KeyTable: 163 real Wikipedia_Exp_14.04 participants`

- [ ] **Step 3: Commit**

```bash
git add scripts/cleaning/spatial_search_export.py
git commit -m "feat(spatial-search): load + filter wiki KeyTable"
```

---

## Task 3: Load BSON, filter to wiki sessions, build the lookup

**Files:**
- Modify: `scripts/cleaning/spatial_search_export.py`

- [ ] **Step 1: Add the loader and lookup builder**

Add above `main()`:

```python
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
```

Update `main()`:

```python
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

    return 0
```

- [ ] **Step 2: Run and verify expected counts**

Run: `py scripts/cleaning/spatial_search_export.py`
Expected stdout includes:
- `spatial-search BSON: 495 raw records (across 2 dumps)`
- `filtered to wikipedia sessions: 163 records`
- `Prolific UserId lookup: 163 entries, full match against wiki KeyTable`

If the match assertion fires, the wiki KeyTable was probably copied from a stale Drive snapshot — re-copy the latest version and re-run.

- [ ] **Step 3: Commit**

```bash
git add scripts/cleaning/spatial_search_export.py
git commit -m "feat(spatial-search): load BSON dumps and verify 1:1 UID match"
```

---

## Task 4: Build Game.csv

**Files:**
- Modify: `scripts/cleaning/spatial_search_export.py`

- [ ] **Step 1: Add the row-flattener and the per-participant frame builder**

Add above `main()`:

```python
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
```

Update `main()` to build and write `Game.csv`:

```python
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

    return 0
```

- [ ] **Step 2: Run and verify Game.csv counts**

Run: `py scripts/cleaning/spatial_search_export.py`
Expected stdout includes: `wrote data/cleaned/spatial_search/Game.csv (154,574 rows, 163 participants)` (the exact total may differ by ~few rows if any non-dict items slipped through; off by more than a few rows means a filter regression).

- [ ] **Step 3: Spot-check the output**

Run:
```bash
py -3 -c "import pandas as pd; df = pd.read_csv(r'data/cleaned/spatial_search/Game.csv'); print(df.shape); print(df['Action'].value_counts()); print(df['GameCondition'].value_counts(dropna=False))"
```
Expected: shape `(154574, 18)`; Action and GameCondition counts matching the spec breakdown.

- [ ] **Step 4: Commit**

```bash
git add scripts/cleaning/spatial_search_export.py data/cleaned/spatial_search/Game.csv
git commit -m "feat(spatial-search): emit Game.csv keyed by wiki ID"
```

---

## Task 5: Build KeyTable.csv

**Files:**
- Modify: `scripts/cleaning/spatial_search_export.py`

- [ ] **Step 1: Add the KeyTable builder**

Add above `main()`:

```python
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
```

Update `main()` (insert before the final `return 0`):

```python
    keytable_df = build_keytable_df(wiki_kt, lookup)
    keytable_df.to_csv(OUT_KEYTABLE, index=False, encoding="utf-8")
    print(f"  wrote {OUT_KEYTABLE.relative_to(PROJECT_ROOT)} ({len(keytable_df)} rows)")
```

- [ ] **Step 2: Run and verify KeyTable count is 163**

Run: `py scripts/cleaning/spatial_search_export.py`
Expected stdout includes: `wrote data/cleaned/spatial_search/KeyTable.csv (163 rows)`

- [ ] **Step 3: Spot-check the KeyTable**

Run:
```bash
py -3 -c "import pandas as pd; df = pd.read_csv(r'data/cleaned/spatial_search/KeyTable.csv'); print(df.shape); print(df.head(5).to_string()); print('SpatialSearchID nulls:', df['SpatialSearchID'].isna().sum())"
```
Expected: shape `(163, 12)`; `SpatialSearchID nulls: 0`.

- [ ] **Step 4: Commit**

```bash
git add scripts/cleaning/spatial_search_export.py data/cleaned/spatial_search/KeyTable.csv
git commit -m "feat(spatial-search): emit KeyTable.csv with spatial metadata"
```

---

## Task 6: End-of-script validation block

**Files:**
- Modify: `scripts/cleaning/spatial_search_export.py`

- [ ] **Step 1: Add a `validate_outputs` function and call it from `main`**

Add above `main()`:

```python
def validate_outputs(game_df: pd.DataFrame, keytable_df: pd.DataFrame,
                     wiki_records: list[dict]) -> None:
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
        len((d.get("Records") or {}).get("Game") or [])
        for d in wiki_records
    )
    checks.append((
        "Game.csv row count matches sum of BSON Records.Game lengths",
        len(game_df) == expected_total,
        f"got {len(game_df)}, expected {expected_total}",
    ))

    # Per-participant trial structure (warning only, not hard-fail).
    trial_check = (
        game_df[game_df["Action"] == "Round_start"]
        .groupby(["ID", "GameCondition"])
        .size()
        .unstack(fill_value=0)
    )
    expected_per_cond = {"Diffuse": 5, "Clumpy": 5}
    anomalies = []
    for cond, expected in expected_per_cond.items():
        if cond not in trial_check.columns:
            anomalies.append(f"no {cond} trials at all")
            continue
        bad = trial_check[trial_check[cond] != expected]
        if len(bad):
            anomalies.append(f"{len(bad)} participants with {cond} != {expected}")
    if anomalies:
        print(f"  [warn] trial structure anomalies: {anomalies}")
    else:
        print("  trial structure: 5 Diffuse + 5 Clumpy per participant")

    print("\nValidation:")
    failed = []
    for name, ok, detail in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}" + (f" -- {detail}" if not ok else ""))
        if not ok:
            failed.append(name)
    if failed:
        raise AssertionError(f"{len(failed)} validation check(s) failed: {failed}")
```

Update `main()` to call it (replace the body of `main`'s tail with):

```python
    validate_outputs(game_df, keytable_df, wiki_records)
    print("[spatial_search_export] done")
    return 0
```

- [ ] **Step 2: Run end-to-end and verify all checks pass**

Run: `py scripts/cleaning/spatial_search_export.py`
Expected tail:
```
Validation:
  [PASS] KeyTable row count == 163
  [PASS] every Game.csv ID is present in KeyTable.csv
  [PASS] Game.csv row count matches sum of BSON Records.Game lengths
[spatial_search_export] done
```
The trial-structure line may print as a `[warn]` if any participant has unusual trial counts; that is informational only.

- [ ] **Step 3: Cross-check the wiki join works**

Run:
```bash
py -3 -c "import pandas as pd; ss = pd.read_csv(r'data/cleaned/spatial_search/Game.csv'); wk = pd.read_csv(r'data/cleaned_new/Game.csv'); ss_ids = set(ss['ID'].unique()); wk_ids = set(wk['ID'].unique()); print('ss ids:', len(ss_ids)); print('wk ids:', len(wk_ids)); print('ss subset of wk:', ss_ids.issubset(wk_ids)); print('ss not in wk:', sorted(ss_ids - wk_ids))"
```
Expected: `ss subset of wk: True`, empty `ss not in wk` list.

- [ ] **Step 4: Commit**

```bash
git add scripts/cleaning/spatial_search_export.py
git commit -m "feat(spatial-search): add end-of-script validation block"
```

---

## Self-Review

**Spec coverage (against `2026-05-07-spatial-search-export-design.html`):**

| Spec section | Plan task |
|---|---|
| Inputs (2 BSONs + wiki KeyTable) | Task 1 (constants), Tasks 2-3 (loaders) |
| Output `Game.csv` 18 columns | Task 4 (`GAME_COLUMNS` + `event_row`) |
| Output `KeyTable.csv` 12 columns | Task 5 (`KEYTABLE_COLUMNS` + `build_keytable_df`) |
| Pipeline step 1 (filter wiki KeyTable) | Task 2 |
| Pipeline step 2 (load + filter BSON) | Task 3 |
| Pipeline step 3 (build UID lookup, hard-fail dup) | Task 3 (`build_uid_lookup`) |
| Pipeline step 4 (assert full UID match) | Task 3 (`assert_full_match`) |
| Pipeline step 5 (flatten Records.Game; override ID) | Task 4 (`event_row` overrides ID with wiki ID) |
| Pipeline step 6 (write KeyTable with merged metadata) | Task 5 |
| Pipeline step 7 (UTF-8, no index) | Tasks 4 & 5 (`encoding="utf-8", index=False`) |
| Pipeline step 8 (validation block) | Task 6 |
| Acceptance criterion 1 (Game.csv 154,574 rows / 18 cols) | Task 4 step 3 + Task 6 |
| Acceptance criterion 2 (KeyTable.csv 163 rows / 12 cols) | Task 5 step 3 + Task 6 |
| Acceptance criterion 3 (validation block PASS) | Task 6 |
| Acceptance criterion 4 (join-compat with wiki Game.csv) | Task 6 step 3 |

**Placeholder scan:** No TBD/TODO. Every step shows the exact code or command. Counts and column lists are concrete throughout.

**Type consistency:** `wiki_kt`, `lookup`, `wiki_records`, `game_df`, `keytable_df` are reused identically across Tasks 2-6. `event_row` returns `dict[str, Any]`; `build_game_df` consumes it; column order matches `GAME_COLUMNS`. `KEYTABLE_COLUMNS` matches `build_keytable_df`'s row dict keys 1:1.

---

## Execution choice

Plan complete. The script is ~250 lines in a single file, with five integration checkpoints (one per task) plus a final validation block. For a one-shot data export the practical fit is **inline execution** — there is no parallelizable work, no repeated test/code cycle, and the validation block at the end is the integration test.

Recommended: execute inline, task-by-task, with the user inspecting the printed counts after each step.

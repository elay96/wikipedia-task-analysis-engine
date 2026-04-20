# Cleaning Pipeline

Two-stage pipeline that produces `data/cleaned/Game.csv` from the raw experiment
logs at `data/dirty_data/Game.csv`.

## Overview

```
data/dirty_data/Game.csv
        │
        ▼   step1a_resolve_revids.py  (network: MediaWiki API)
data/cleaned/revid_lookup.csv
        │
        ▼   step1b_apply_cleaning.py  (deterministic CSV transform + validation)
data/cleaned/Game.csv
```

Stage 2 (article content fetch) is designed but not implemented here - see
`docs/superpowers/specs/2026-04-20-data-cleaning-design.html` section 6.

## Prerequisites

```
pip install -r requirements.txt
```

Requires `requests`, `pandas`, `pytest`. Python 3.9+.

## Running

### Stage 1a - resolve revids (slow, network-bound)

```
py scripts/cleaning/step1a_resolve_revids.py
```

- Reads `data/dirty_data/Game.csv`.
- Writes / updates `data/cleaned/revid_lookup.csv`.
- Resumable: re-runs skip entries already resolved with `status=ok`.
- Expected runtime: ~30-90s on the current dataset (~250 unique pairs).
- Progress is printed every 50 calls; final summary shows `ok / not_found / error` counts.

Custom paths:
```
py scripts/cleaning/step1a_resolve_revids.py --dirty X --lookup Y
```

### Stage 1b - apply cleaning (fast, deterministic)

```
py scripts/cleaning/step1b_apply_cleaning.py
```

- Reads `data/dirty_data/Game.csv` and `data/cleaned/revid_lookup.csv`.
- Writes `data/cleaned/Game.csv`.
- Runs a validation gate; exits non-zero if any invariant is violated.

## Lookup CSV schema

| Column | Example | Notes |
|---|---|---|
| `slug` | `Capybara` | Unchanged from the source row's `ArticleSlug` |
| `timestamp` | `2026-04-14T13:15:03.415Z` | Normalised ISO-8601 UTC, ms precision |
| `revid` | `1349431902` | Empty when `status != ok` |
| `status` | `ok` / `not_found` / `error` | Audit value |
| `fetched_at` | `2026-04-20T12:00:00.000Z` | Debug / cache invalidation |

## Tests

```
pytest scripts/cleaning/            # unit tests (fast, no network)
pytest scripts/cleaning/ -m live    # live MediaWiki API tests (opt-in)
```

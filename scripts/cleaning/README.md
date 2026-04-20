# Cleaning Pipeline

Three-stage pipeline that produces the canonical cleaned dataset +
article-content artefacts in `data/cleaned/`. The raw experiment logs at
`data/dirty_data/Game.csv` are the single source of truth for input; all
derived artefacts live in `data/cleaned/`.

## Overview

```
data/dirty_data/Game.csv              (raw experiment logs, COMPOUND)
        │
        ▼   step1a_resolve_revids.py          (MediaWiki API - resumable)
data/cleaned/revid_lookup.csv
        │
        ▼   step1b_apply_cleaning.py          (deterministic + validation gate)
data/cleaned/Game.csv                 (canonical input for analysis scripts)
        │
        ▼   step2_fetch_articles.py           (MediaWiki API - resumable)
data/cleaned/articles.jsonl           (revid -> plain-text content)
        │
        ▼   step3_build_wiki_texts.py         (slug -> latest content)
data/cleaned/wiki_texts.json
        │
        ▼   scripts/compute_topics.py         (LDA)
        ▼   scripts/compute_similarity.py     (tf-idf cosine)
        ▼   scripts/compute_bertopic.py       (optional - BERTopic variants)
data/cleaned/{topic_model,similarity_matrix,bertopic_*}.json
```

**Rule:** every derived artefact (anything that can be regenerated from
`data/dirty_data/Game.csv`) lives in `data/cleaned/`. Nothing should ever
be written to the root of `data/`.

## Prerequisites

```
pip install -r requirements.txt
```

Requires `requests`, `pandas`, `pytest`. Python 3.9+.

## Full re-run sequence (drop-in for new compound data)

When new participants finish and a new compound `Game.csv` arrives:

```bash
# 1. replace the raw input (compound = old rows preserved + new rows appended)
cp /path/to/new/Game.csv data/dirty_data/Game.csv

# 2. regenerate the cleaned dataset (resumable where possible)
py scripts/cleaning/step1a_resolve_revids.py     # only new (slug, time) pairs hit the API
py scripts/cleaning/step1b_apply_cleaning.py     # regenerate data/cleaned/Game.csv
py scripts/cleaning/step2_fetch_articles.py      # only new revids hit the API
py scripts/cleaning/step3_build_wiki_texts.py    # rebuild slug -> content map

# 3. regenerate the semantic artefacts (full rebuild; fast)
py scripts/compute_topics.py                     # LDA -> topic_model.json
py scripts/compute_similarity.py                 # tf-idf -> similarity_matrix.json
# py scripts/compute_bertopic.py                 # optional, slower (BERTopic)
```

All stages are idempotent. `step1a` and `step2` also resume cleanly after
a crash or SIGINT.

## Per-stage notes

### Stage 1a - resolve revids (slow, network-bound)

- Reads `data/dirty_data/Game.csv`.
- Writes / updates `data/cleaned/revid_lookup.csv`.
- Resumable: re-runs skip entries already resolved with `status=ok`.
- Progress is printed every 50 calls; final summary shows `ok / not_found / error` counts.

Custom paths:
```
py scripts/cleaning/step1a_resolve_revids.py --dirty X --lookup Y
```

### Stage 1b - apply cleaning (fast, deterministic)

- Reads `data/dirty_data/Game.csv` and `data/cleaned/revid_lookup.csv`.
- Writes `data/cleaned/Game.csv`.
- Runs a validation gate; exits non-zero if any invariant is violated.

### Stage 2 - fetch article content

- Input:  `data/cleaned/Game.csv`
- Output: `data/cleaned/articles.jsonl` (schema: `{article_slug, revid, timestamp, content}`)
- Resumable: any revid already present in the JSONL is skipped.
- Validation at end: every unique non-null `ArticleRevid` in cleaned CSV
  must have a non-empty `content` line in the JSONL.

### Stage 3 - build wiki_texts.json

- Input:  `data/cleaned/articles.jsonl`
- Output: `data/cleaned/wiki_texts.json` (slug -> latest content)
- For each slug, picks the content from the latest (timestamp, revid).
- Full rebuild each run; fast.

## Lookup CSV schema (`revid_lookup.csv`)

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

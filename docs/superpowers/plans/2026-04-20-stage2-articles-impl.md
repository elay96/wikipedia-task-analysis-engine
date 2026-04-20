# Stage 2: Article Content Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch Wikipedia plain-text article content for every unique revid in `data/cleaned/Game.csv` and emit one JSON object per line to `data/cleaned/articles.jsonl`, ready for downstream LDA / embeddings / PCA analyses.

**Architecture:** Mirror stage 1a's two-layer split. `api_client.py` gains a new `fetch_extract_by_revid(session, revid)` function that reuses the session factory, retry, backoff, and throttle contract already in place. A new `article_store.py` module reads/writes the JSONL artefact and extracts the `(slug, revid, earliest_timestamp)` triples from the cleaned CSV. `step2_fetch_articles.py` is the orchestrator: resumable, idempotent, sequential. Validation in `validation.py` gains an assertion that every unique revid in the cleaned CSV has a non-empty content line in the JSONL.

**Tech Stack:** Python 3.11+, `requests` (already in stage-1 api_client), `pandas` (already used), `pytest` (stage-1 test harness, colocated `test_*.py`), MediaWiki action API `prop=extracts&explaintext=1`.

**Reuses from stage 1 (`scripts/cleaning/`):**
- `api_client.build_session()` — User-Agent `WikipediaTaskAnalysis/1.0 (elay96@gmail.com)`
- Retry pattern: 3 attempts, backoff 1s/2s/4s on 429/5xx, throttle 50 ms between successes
- Pytest live marker (`-m live`) from `pytest.ini` for opt-in integration tests
- `pytest.ini` `pythonpath = scripts scripts/cleaning` — no config changes needed

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `scripts/cleaning/api_client.py` | MODIFY | Add `ExtractResult` TypedDict + `fetch_extract_by_revid(session, revid)` |
| `scripts/cleaning/test_api_client.py` | MODIFY | Add `TestFetchExtractByRevid` class |
| `scripts/cleaning/article_store.py` | CREATE | JSONL read/append; extract unique revid records from cleaned CSV |
| `scripts/cleaning/test_article_store.py` | CREATE | Unit tests for the new module |
| `scripts/cleaning/step2_fetch_articles.py` | CREATE | Orchestrator + CLI entrypoint |
| `scripts/cleaning/test_step2.py` | CREATE | Orchestrator tests with fake fetcher |
| `scripts/cleaning/validation.py` | MODIFY | Add `validate_articles(cleaned_df, articles_path)` |
| `scripts/cleaning/test_validation.py` | MODIFY | Add tests for `validate_articles` |
| `scripts/cleaning/test_integration_live.py` | MODIFY | Add `test_capybara_extract_has_content` |
| `scripts/cleaning/README.md` | MODIFY | Document stage 2 usage |
| `data/cleaned/articles.jsonl` | OUTPUT | Produced by real-data dry run; committed at end |

---

## Prerequisites (must already hold)

- `data/cleaned/Game.csv` exists from stage 1 (4,745 rows, 36 users, integer `ArticleRevid`).
- `scripts/cleaning/api_client.py` has `build_session` and `fetch_revision_at`.
- `pytest.ini` includes `pythonpath = scripts scripts/cleaning` and a `live` marker.
- `data/cleaned/revid_lookup.csv` exists (584 rows, all `status=ok`).

Run these checks before starting:

```bash
py -c "import pandas as pd; df = pd.read_csv('data/cleaned/Game.csv'); print('rows:', len(df), 'unique_revids:', df['ArticleRevid'].dropna().nunique())"
```
Expected: `rows: 4745 unique_revids: 584`

```bash
pytest scripts/cleaning -q
```
Expected: all stage-1 tests pass (live tests skipped).

---

## Task 1: `fetch_extract_by_revid` — happy path + not_found + retry

Add a new API function alongside `fetch_revision_at`. Same retry/backoff/throttle contract. Returns plain-text content for a given revid.

**Files:**
- Modify: `scripts/cleaning/api_client.py`
- Test: `scripts/cleaning/test_api_client.py`

- [ ] **Step 1.1: Write failing tests**

Append the following class to `scripts/cleaning/test_api_client.py` (do NOT remove any existing content):

```python
class TestFetchExtractByRevid:
    def test_happy_path_returns_content(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "1234": {
                        "pageid": 1234,
                        "title": "Capybara",
                        "extract": "The capybara (Hydrochoerus hydrochaeris) is the largest living rodent...",
                    }
                }
            }
        })
        from api_client import fetch_extract_by_revid
        result = fetch_extract_by_revid(session, 1349431902)
        assert result["status"] == "ok"
        assert result["content"].startswith("The capybara")
        assert result["error"] is None

    def test_missing_extract_returns_not_found(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "-1": {"ns": 0, "title": "NonexistentRev", "missing": ""}
                }
            }
        })
        from api_client import fetch_extract_by_revid
        result = fetch_extract_by_revid(session, 999999999999)
        assert result["status"] == "not_found"
        assert result["content"] is None

    def test_empty_extract_string_returns_not_found(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {"pages": {"1": {"pageid": 1, "title": "X", "extract": ""}}}
        })
        from api_client import fetch_extract_by_revid
        result = fetch_extract_by_revid(session, 42)
        assert result["status"] == "not_found"
        assert result["content"] is None

    def test_retries_then_succeeds_on_5xx(self):
        session = MagicMock()
        ok = _mock_response({
            "query": {"pages": {"1": {"pageid": 1, "title": "X", "extract": "body"}}}
        })
        fail = _mock_response({}, status_code=503)
        session.get.side_effect = [fail, fail, ok]
        from api_client import fetch_extract_by_revid
        with patch("api_client.time.sleep"):
            result = fetch_extract_by_revid(session, 42)
        assert result["status"] == "ok"
        assert result["content"] == "body"
        assert session.get.call_count == 3

    def test_gives_up_after_3_retries(self):
        session = MagicMock()
        session.get.return_value = _mock_response({}, status_code=500)
        from api_client import fetch_extract_by_revid
        with patch("api_client.time.sleep"):
            result = fetch_extract_by_revid(session, 42)
        assert result["status"] == "error"
        assert result["error"] == "HTTP 500"
        assert session.get.call_count == 3

    def test_sends_correct_params(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {"pages": {"1": {"pageid": 1, "title": "X", "extract": "body"}}}
        })
        from api_client import fetch_extract_by_revid
        fetch_extract_by_revid(session, 1349431902)
        _, kwargs = session.get.call_args
        params = kwargs["params"]
        assert params["action"] == "query"
        assert params["prop"] == "extracts"
        assert params["explaintext"] == 1
        assert params["exsectionformat"] == "plain"
        assert params["revids"] == 1349431902
        assert params["redirects"] == 1
        assert params["format"] == "json"
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `pytest scripts/cleaning/test_api_client.py::TestFetchExtractByRevid -v`
Expected: 6 tests ERROR/FAIL with `ImportError: cannot import name 'fetch_extract_by_revid'`.

- [ ] **Step 1.3: Implement `fetch_extract_by_revid`**

In `scripts/cleaning/api_client.py`, add the following AFTER the existing `fetch_revision_at` function (keep everything else unchanged):

```python
class ExtractResult(TypedDict):
    content: Optional[str]
    status: str  # 'ok' | 'not_found' | 'error'
    error: Optional[str]


def fetch_extract_by_revid(session: requests.Session, revid: int) -> ExtractResult:
    """Return the plain-text extract of the article version identified by `revid`.

    Follows redirects (redirects=1). Retries on 429/5xx with exponential backoff.
    Returns ok/not_found/error per the ExtractResult contract. An empty extract
    string is treated as not_found (revid exists but API returned no content).
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "revids": revid,
        "redirects": 1,
        "format": "json",
        "formatversion": 1,
    }

    last_error: Optional[str] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(API_ENDPOINT, params=params, timeout=TIMEOUT_SEC)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_BASE * (2 ** attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE * (2 ** attempt))
    else:
        return {"content": None, "status": "error", "error": last_error}

    time.sleep(THROTTLE_SEC)

    pages = (data.get("query") or {}).get("pages") or {}
    for _, page in pages.items():
        if "missing" in page:
            return {"content": None, "status": "not_found", "error": None}
        extract = page.get("extract")
        if not extract:
            return {"content": None, "status": "not_found", "error": None}
        return {"content": extract, "status": "ok", "error": None}

    return {"content": None, "status": "error", "error": "unexpected response shape"}
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `pytest scripts/cleaning/test_api_client.py -v`
Expected: all tests pass (existing + 6 new).

- [ ] **Step 1.5: Commit**

```bash
git add scripts/cleaning/api_client.py scripts/cleaning/test_api_client.py
git commit -m "feat(cleaning): add fetch_extract_by_revid to api_client"
```

---

## Task 2: `article_store.py` — JSONL read/append

New module for reading existing `articles.jsonl` (to support resume) and appending new records.

**Files:**
- Create: `scripts/cleaning/article_store.py`
- Create: `scripts/cleaning/test_article_store.py`

- [ ] **Step 2.1: Write failing tests**

Create `scripts/cleaning/test_article_store.py` with the following content:

```python
import json
from pathlib import Path

import pandas as pd
import pytest

from article_store import (
    ARTICLE_FIELDS,
    append_article,
    extract_unique_revid_records,
    load_existing_revids,
)


class TestLoadExistingRevids:
    def test_returns_empty_set_when_file_missing(self, tmp_path: Path):
        result = load_existing_revids(tmp_path / "does_not_exist.jsonl")
        assert result == set()

    def test_returns_empty_set_when_file_empty(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_existing_revids(p) == set()

    def test_parses_one_revid_per_line(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        lines = [
            json.dumps({"article_slug": "A", "revid": 1, "timestamp": "t", "content": "x"}),
            json.dumps({"article_slug": "B", "revid": 2, "timestamp": "t", "content": "y"}),
        ]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert load_existing_revids(p) == {1, 2}

    def test_ignores_blank_lines(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        p.write_text(
            json.dumps({"article_slug": "A", "revid": 1, "timestamp": "t", "content": "x"}) + "\n\n",
            encoding="utf-8",
        )
        assert load_existing_revids(p) == {1}


class TestAppendArticle:
    def test_creates_parent_directory_if_missing(self, tmp_path: Path):
        p = tmp_path / "nested" / "dir" / "a.jsonl"
        append_article(p, {"article_slug": "A", "revid": 1, "timestamp": "t", "content": "x"})
        assert p.exists()

    def test_appends_one_line_per_call(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        append_article(p, {"article_slug": "A", "revid": 1, "timestamp": "t", "content": "x"})
        append_article(p, {"article_slug": "B", "revid": 2, "timestamp": "t", "content": "y"})
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["revid"] == 1
        assert json.loads(lines[1])["revid"] == 2

    def test_each_line_is_valid_json_with_all_fields(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        record = {"article_slug": "A", "revid": 1, "timestamp": "t", "content": "body"}
        append_article(p, record)
        parsed = json.loads(p.read_text(encoding="utf-8").strip())
        assert set(parsed.keys()) == set(ARTICLE_FIELDS)
        assert parsed == record

    def test_rejects_record_with_missing_fields(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        with pytest.raises(ValueError, match="missing"):
            append_article(p, {"revid": 1, "content": "x"})


class TestExtractUniqueRevidRecords:
    def test_groups_by_revid_and_takes_earliest_time(self):
        df = pd.DataFrame({
            "Action": ["article_open", "article_open", "article_open"],
            "ArticleSlug": ["Capybara", "Capybara", "Beaver"],
            "ArticleRevid": pd.array([100, 100, 200], dtype="Int64"),
            "Time": [
                "2026-04-14T13:15:03.415Z",
                "2026-04-14T13:10:00.000Z",
                "2026-04-14T14:00:00.000Z",
            ],
        })
        records = extract_unique_revid_records(df)
        assert len(records) == 2
        by_revid = {r["revid"]: r for r in records}
        assert by_revid[100]["timestamp"] == "2026-04-14T13:10:00.000Z"
        assert by_revid[100]["article_slug"] == "Capybara"
        assert by_revid[200]["timestamp"] == "2026-04-14T14:00:00.000Z"

    def test_skips_null_revids(self):
        df = pd.DataFrame({
            "Action": ["article_open", "article_open"],
            "ArticleSlug": ["X", "Y"],
            "ArticleRevid": pd.array([100, pd.NA], dtype="Int64"),
            "Time": ["2026-04-14T13:00:00.000Z", "2026-04-14T14:00:00.000Z"],
        })
        records = extract_unique_revid_records(df)
        assert len(records) == 1
        assert records[0]["revid"] == 100

    def test_ignores_non_article_open_rows(self):
        df = pd.DataFrame({
            "Action": ["article_open", "navigation", "article_close"],
            "ArticleSlug": ["A", "B", "C"],
            "ArticleRevid": pd.array([1, 2, 3], dtype="Int64"),
            "Time": [
                "2026-04-14T13:00:00.000Z",
                "2026-04-14T13:01:00.000Z",
                "2026-04-14T13:02:00.000Z",
            ],
        })
        records = extract_unique_revid_records(df)
        assert [r["revid"] for r in records] == [1]

    def test_returns_records_sorted_by_revid(self):
        df = pd.DataFrame({
            "Action": ["article_open"] * 3,
            "ArticleSlug": ["C", "A", "B"],
            "ArticleRevid": pd.array([300, 100, 200], dtype="Int64"),
            "Time": ["2026-04-14T13:00:00.000Z"] * 3,
        })
        records = extract_unique_revid_records(df)
        assert [r["revid"] for r in records] == [100, 200, 300]

    def test_record_has_all_four_fields_except_content(self):
        df = pd.DataFrame({
            "Action": ["article_open"],
            "ArticleSlug": ["A"],
            "ArticleRevid": pd.array([100], dtype="Int64"),
            "Time": ["2026-04-14T13:00:00.000Z"],
        })
        records = extract_unique_revid_records(df)
        assert set(records[0].keys()) == {"article_slug", "revid", "timestamp"}
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest scripts/cleaning/test_article_store.py -v`
Expected: all ERROR with `ModuleNotFoundError: No module named 'article_store'`.

- [ ] **Step 2.3: Implement `article_store.py`**

Create `scripts/cleaning/article_store.py` with:

```python
"""Read/write articles.jsonl and extract unique revid records from cleaned CSV."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd

ARTICLE_FIELDS = ("article_slug", "revid", "timestamp", "content")


def load_existing_revids(path) -> Set[int]:
    """Return the set of revids already present in an articles.jsonl file.

    Returns an empty set if the file does not exist or is empty.
    """
    p = Path(path)
    if not p.exists():
        return set()
    revids: Set[int] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            revids.add(int(record["revid"]))
    return revids


def append_article(path, record: dict) -> None:
    """Append a single article record as one JSON line. Creates parent dir if needed.

    Record must have exactly the fields in ARTICLE_FIELDS.
    """
    missing = set(ARTICLE_FIELDS) - set(record.keys())
    if missing:
        raise ValueError(f"append_article: record missing fields: {sorted(missing)}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_unique_revid_records(cleaned_df: pd.DataFrame) -> List[dict]:
    """From a cleaned Game.csv DataFrame, emit one record per unique non-null revid.

    Each record has article_slug + revid + timestamp (the earliest Time across all
    article_open rows with that revid). Output is sorted by revid ascending.
    """
    mask = (cleaned_df["Action"] == "article_open") & cleaned_df["ArticleRevid"].notna()
    sub = cleaned_df.loc[mask, ["ArticleSlug", "ArticleRevid", "Time"]].copy()
    sub["ArticleRevid"] = sub["ArticleRevid"].astype("int64")
    sub = sub.sort_values("Time")
    grouped = sub.groupby("ArticleRevid", sort=True, as_index=False).first()

    records: List[dict] = []
    for _, row in grouped.iterrows():
        records.append({
            "article_slug": str(row["ArticleSlug"]),
            "revid": int(row["ArticleRevid"]),
            "timestamp": str(row["Time"]),
        })
    return records
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `pytest scripts/cleaning/test_article_store.py -v`
Expected: all 12 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add scripts/cleaning/article_store.py scripts/cleaning/test_article_store.py
git commit -m "feat(cleaning): add article_store for JSONL persistence"
```

---

## Task 3: `step2_fetch_articles.py` — orchestrator with resume

The orchestrator iterates over unique revid records, fetches extracts, appends to JSONL. Skips revids already present. Uses a `fetch_fn` parameter so tests can inject a fake.

**Files:**
- Create: `scripts/cleaning/step2_fetch_articles.py`
- Create: `scripts/cleaning/test_step2.py`

- [ ] **Step 3.1: Write failing tests**

Create `scripts/cleaning/test_step2.py` with:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from step2_fetch_articles import resolve_all_articles


def _cleaned_df(pairs):
    """Build a minimal cleaned Game.csv-shaped DataFrame from (slug, revid, time) triples."""
    return pd.DataFrame({
        "Action": ["article_open"] * len(pairs),
        "ArticleSlug": [p[0] for p in pairs],
        "ArticleRevid": pd.array([p[1] for p in pairs], dtype="Int64"),
        "Time": [p[2] for p in pairs],
    })


class TestResolveAllArticles:
    def test_writes_one_line_per_unique_revid(self, tmp_path: Path):
        df = _cleaned_df([
            ("Capybara", 100, "2026-04-14T13:00:00.000Z"),
            ("Beaver", 200, "2026-04-14T14:00:00.000Z"),
        ])
        fetch_fn = MagicMock(side_effect=[
            {"content": "capybara body", "status": "ok", "error": None},
            {"content": "beaver body", "status": "ok", "error": None},
        ])
        out_path = tmp_path / "articles.jsonl"

        resolve_all_articles(df, out_path, session=MagicMock(), fetch_fn=fetch_fn)

        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        parsed = [json.loads(l) for l in lines]
        by_revid = {r["revid"]: r for r in parsed}
        assert by_revid[100]["article_slug"] == "Capybara"
        assert by_revid[100]["content"] == "capybara body"
        assert by_revid[200]["content"] == "beaver body"

    def test_resume_skips_already_fetched_revids(self, tmp_path: Path):
        out_path = tmp_path / "articles.jsonl"
        out_path.write_text(
            json.dumps({
                "article_slug": "Capybara",
                "revid": 100,
                "timestamp": "2026-04-14T13:00:00.000Z",
                "content": "already fetched",
            }) + "\n",
            encoding="utf-8",
        )
        df = _cleaned_df([
            ("Capybara", 100, "2026-04-14T13:00:00.000Z"),
            ("Beaver", 200, "2026-04-14T14:00:00.000Z"),
        ])
        fetch_fn = MagicMock(return_value={
            "content": "beaver body", "status": "ok", "error": None,
        })

        resolve_all_articles(df, out_path, session=MagicMock(), fetch_fn=fetch_fn)

        assert fetch_fn.call_count == 1  # only Beaver fetched
        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        by_revid = {json.loads(l)["revid"]: json.loads(l) for l in lines}
        assert by_revid[100]["content"] == "already fetched"
        assert by_revid[200]["content"] == "beaver body"

    def test_skips_not_found_and_error_results(self, tmp_path: Path):
        df = _cleaned_df([
            ("A", 1, "2026-04-14T13:00:00.000Z"),
            ("B", 2, "2026-04-14T13:01:00.000Z"),
            ("C", 3, "2026-04-14T13:02:00.000Z"),
        ])
        fetch_fn = MagicMock(side_effect=[
            {"content": "a body", "status": "ok", "error": None},
            {"content": None, "status": "not_found", "error": None},
            {"content": None, "status": "error", "error": "HTTP 500"},
        ])
        out_path = tmp_path / "articles.jsonl"

        resolve_all_articles(df, out_path, session=MagicMock(), fetch_fn=fetch_fn)

        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["revid"] == 1

    def test_passes_revid_to_fetch_fn(self, tmp_path: Path):
        df = _cleaned_df([("A", 42, "2026-04-14T13:00:00.000Z")])
        fetch_fn = MagicMock(return_value={
            "content": "body", "status": "ok", "error": None,
        })

        resolve_all_articles(df, tmp_path / "a.jsonl", session=MagicMock(), fetch_fn=fetch_fn)

        fetch_fn.assert_called_once_with(42)

    def test_returns_summary_counts(self, tmp_path: Path):
        df = _cleaned_df([
            ("A", 1, "2026-04-14T13:00:00.000Z"),
            ("B", 2, "2026-04-14T13:01:00.000Z"),
            ("C", 3, "2026-04-14T13:02:00.000Z"),
        ])
        fetch_fn = MagicMock(side_effect=[
            {"content": "x", "status": "ok", "error": None},
            {"content": None, "status": "not_found", "error": None},
            {"content": None, "status": "error", "error": "HTTP 500"},
        ])

        summary = resolve_all_articles(df, tmp_path / "a.jsonl",
                                       session=MagicMock(), fetch_fn=fetch_fn)

        assert summary == {"ok": 1, "not_found": 1, "error": 1, "skipped": 0}

    def test_empty_cleaned_df_writes_no_file(self, tmp_path: Path):
        df = _cleaned_df([])
        fetch_fn = MagicMock()
        out_path = tmp_path / "a.jsonl"

        summary = resolve_all_articles(df, out_path, session=MagicMock(), fetch_fn=fetch_fn)

        assert summary == {"ok": 0, "not_found": 0, "error": 0, "skipped": 0}
        assert fetch_fn.call_count == 0
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `pytest scripts/cleaning/test_step2.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'step2_fetch_articles'`.

- [ ] **Step 3.3: Implement `step2_fetch_articles.py`**

Create `scripts/cleaning/step2_fetch_articles.py` with:

```python
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
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `pytest scripts/cleaning/test_step2.py -v`
Expected: all 6 tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/cleaning/step2_fetch_articles.py scripts/cleaning/test_step2.py
git commit -m "feat(cleaning): add step2 orchestrator for article fetch"
```

---

## Task 4: `validate_articles` — assertion that JSONL matches cleaned CSV

After `step2` finishes, we want a single function that answers: is every unique revid in the cleaned CSV present in the JSONL with non-empty content? This is the stage-2 equivalent of `validate_cleaned` from stage 1b.

**Files:**
- Modify: `scripts/cleaning/validation.py`
- Modify: `scripts/cleaning/test_validation.py`

- [ ] **Step 4.1: Write failing tests**

First, add these imports at the top of `scripts/cleaning/test_validation.py` (the file already imports `pandas as pd` and `pytest`; leave those alone):

```python
import json
from pathlib import Path
```

Then append the following at the end of the file:

```python
from validation import validate_articles


def _cleaned(revids):
    return pd.DataFrame({
        "Action": ["article_open"] * len(revids),
        "ArticleSlug": [f"A{r}" for r in revids],
        "ArticleRevid": pd.array(revids, dtype="Int64"),
        "Time": ["2026-04-14T13:00:00.000Z"] * len(revids),
    })


def _write_jsonl(path: Path, rows):
    lines = [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class TestValidateArticles:
    def test_passes_when_every_revid_has_non_empty_content(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [
            {"article_slug": "A1", "revid": 1, "timestamp": "t", "content": "body"},
            {"article_slug": "A2", "revid": 2, "timestamp": "t", "content": "body"},
        ])
        validate_articles(_cleaned([1, 2]), p)  # no raise

    def test_fails_when_revid_missing_from_jsonl(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [
            {"article_slug": "A1", "revid": 1, "timestamp": "t", "content": "body"},
        ])
        with pytest.raises(ValidationError, match="missing"):
            validate_articles(_cleaned([1, 2]), p)

    def test_fails_when_content_is_empty_string(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [
            {"article_slug": "A1", "revid": 1, "timestamp": "t", "content": ""},
        ])
        with pytest.raises(ValidationError, match="empty content"):
            validate_articles(_cleaned([1]), p)

    def test_ignores_extra_revids_in_jsonl(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [
            {"article_slug": "A1", "revid": 1, "timestamp": "t", "content": "body"},
            {"article_slug": "A2", "revid": 2, "timestamp": "t", "content": "body"},
            {"article_slug": "A3", "revid": 3, "timestamp": "t", "content": "body"},
        ])
        validate_articles(_cleaned([1, 2]), p)  # no raise even though revid=3 is extra

    def test_fails_when_jsonl_file_missing(self, tmp_path: Path):
        with pytest.raises(ValidationError, match="not exist"):
            validate_articles(_cleaned([1]), tmp_path / "missing.jsonl")
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `pytest scripts/cleaning/test_validation.py::TestValidateArticles -v`
Expected: ERROR `ImportError: cannot import name 'validate_articles'`.

- [ ] **Step 4.3: Implement `validate_articles`**

Read current `scripts/cleaning/validation.py` first to see its structure, then append the following at the end of the file (keep all existing imports and code):

```python
def validate_articles(cleaned_df: pd.DataFrame, articles_path) -> None:
    """Assert every unique non-null revid in the cleaned CSV is represented by a
    non-empty content line in the articles JSONL.
    """
    p = Path(articles_path)
    if not p.exists():
        raise ValidationError(f"articles file does not exist: {p}")

    required = set()
    mask = (cleaned_df["Action"] == "article_open") & cleaned_df["ArticleRevid"].notna()
    for r in cleaned_df.loc[mask, "ArticleRevid"]:
        required.add(int(r))

    present: dict = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            present[int(rec["revid"])] = rec.get("content") or ""

    missing = sorted(required - set(present.keys()))
    if missing:
        sample = missing[:5]
        raise ValidationError(
            f"articles JSONL missing {len(missing)} revids (first 5: {sample})"
        )

    empties = sorted(r for r in required if not present[r].strip())
    if empties:
        sample = empties[:5]
        raise ValidationError(
            f"articles JSONL has empty content for {len(empties)} revids "
            f"(first 5: {sample})"
        )
```

Also add these imports at the top of `validation.py` if not already present (check the existing file first):

```python
import json
from pathlib import Path
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `pytest scripts/cleaning/test_validation.py -v`
Expected: all tests pass (existing + 5 new).

- [ ] **Step 4.5: Commit**

```bash
git add scripts/cleaning/validation.py scripts/cleaning/test_validation.py
git commit -m "feat(cleaning): add validate_articles stage-2 gate"
```

---

## Task 5: Wire validation into `step2` CLI

The CLI should run `validate_articles` after the fetch loop completes, so a successful exit means the JSONL is guaranteed to cover every revid in the cleaned CSV.

**Files:**
- Modify: `scripts/cleaning/step2_fetch_articles.py`
- Modify: `scripts/cleaning/test_step2.py`

- [ ] **Step 5.1: Write failing test**

Append to `scripts/cleaning/test_step2.py`:

```python
class TestStep2MainValidation:
    def test_main_runs_validation_after_fetch(self, tmp_path: Path, monkeypatch):
        # Build a tiny cleaned.csv fixture
        cleaned_path = tmp_path / "Game.csv"
        pd.DataFrame({
            "Action": ["article_open", "article_open"],
            "ArticleSlug": ["A", "B"],
            "ArticleRevid": pd.array([1, 2], dtype="Int64"),
            "Time": ["2026-04-14T13:00:00.000Z", "2026-04-14T13:01:00.000Z"],
        }).to_csv(cleaned_path, index=False)
        out_path = tmp_path / "articles.jsonl"

        calls = {"resolve": 0, "validate": 0}

        def fake_resolve(df, out, *, session, fetch_fn=None):
            calls["resolve"] += 1
            out_path.write_text(
                json.dumps({"article_slug": "A", "revid": 1, "timestamp": "t",
                            "content": "body"}) + "\n" +
                json.dumps({"article_slug": "B", "revid": 2, "timestamp": "t",
                            "content": "body"}) + "\n",
                encoding="utf-8",
            )
            return {"ok": 2, "not_found": 0, "error": 0, "skipped": 0}

        def fake_validate(df, path):
            calls["validate"] += 1

        monkeypatch.setattr("step2_fetch_articles.resolve_all_articles", fake_resolve)
        monkeypatch.setattr("step2_fetch_articles.validate_articles", fake_validate)

        from step2_fetch_articles import _main
        rc = _main(["--cleaned", str(cleaned_path), "--out", str(out_path)])

        assert rc == 0
        assert calls["resolve"] == 1
        assert calls["validate"] == 1

    def test_main_propagates_validation_error(self, tmp_path: Path, monkeypatch):
        cleaned_path = tmp_path / "Game.csv"
        pd.DataFrame({
            "Action": ["article_open"], "ArticleSlug": ["A"],
            "ArticleRevid": pd.array([1], dtype="Int64"),
            "Time": ["2026-04-14T13:00:00.000Z"],
        }).to_csv(cleaned_path, index=False)
        out_path = tmp_path / "articles.jsonl"

        from validation import ValidationError

        monkeypatch.setattr("step2_fetch_articles.resolve_all_articles",
                            lambda df, out, *, session, fetch_fn=None: {
                                "ok": 0, "not_found": 1, "error": 0, "skipped": 0,
                            })

        def raise_validation(df, path):
            raise ValidationError("boom")
        monkeypatch.setattr("step2_fetch_articles.validate_articles", raise_validation)

        from step2_fetch_articles import _main
        with pytest.raises(ValidationError, match="boom"):
            _main(["--cleaned", str(cleaned_path), "--out", str(out_path)])
```

Also add `import pytest` at the top of `scripts/cleaning/test_step2.py` (Task 3 created the file without it; the new `test_main_propagates_validation_error` test uses `pytest.raises`).

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `pytest scripts/cleaning/test_step2.py::TestStep2MainValidation -v`
Expected: FAIL `AttributeError: module 'step2_fetch_articles' has no attribute 'validate_articles'`.

- [ ] **Step 5.3: Modify `step2_fetch_articles.py` to call validation from `_main`**

Add the import near the top (with other local imports):

```python
from validation import validate_articles
```

And replace the body of `_main` with:

```python
def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch article extracts per unique revid.")
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    cleaned = pd.read_csv(args.cleaned, low_memory=False)
    cleaned["ArticleRevid"] = pd.to_numeric(cleaned["ArticleRevid"], errors="coerce").astype("Int64")
    session = build_session()
    resolve_all_articles(cleaned, args.out, session=session)
    validate_articles(cleaned, args.out)
    print(f"[step2] validation OK -> {args.out}")
    return 0
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `pytest scripts/cleaning/test_step2.py -v`
Expected: all tests pass (existing + 2 new).

- [ ] **Step 5.5: Commit**

```bash
git add scripts/cleaning/step2_fetch_articles.py scripts/cleaning/test_step2.py
git commit -m "feat(cleaning): wire validate_articles into step2 CLI"
```

---

## Task 6: Live integration test + README update

One small live-API test that fetches the real Capybara extract by a known revid (1349431902) and asserts non-empty content. Opt-in via `-m live` (matches stage 1 convention).

**Files:**
- Modify: `scripts/cleaning/test_integration_live.py`
- Modify: `scripts/cleaning/README.md`

- [ ] **Step 6.1: Add live test**

Read `scripts/cleaning/test_integration_live.py` and append:

```python
from api_client import fetch_extract_by_revid


@pytest.mark.live
class TestLiveExtractFetch:
    def test_capybara_extract_has_content(self):
        session = build_session()
        result = fetch_extract_by_revid(session, 1349431902)
        assert result["status"] == "ok"
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 100
        assert "capybara" in result["content"].lower()

    def test_nonexistent_revid_returns_not_found(self):
        session = build_session()
        result = fetch_extract_by_revid(session, 999999999999)
        assert result["status"] == "not_found"
        assert result["content"] is None
```

- [ ] **Step 6.2: Run the live tests to verify they pass**

Run: `pytest scripts/cleaning/test_integration_live.py -m live -v`
Expected: all tests pass (hits the real MediaWiki API; needs internet). If they fail, investigate — do NOT proceed to Task 7.

- [ ] **Step 6.3: Update README**

Read `scripts/cleaning/README.md` and add a new section at the end (do NOT remove any existing content):

```markdown

## Stage 2: fetch article content

Stage 2 downloads the plain-text body of each unique Wikipedia revision referenced in
`data/cleaned/Game.csv` and writes `data/cleaned/articles.jsonl` — one JSON object per
line, schema `{article_slug, revid, timestamp, content}`.

### Run

```bash
py scripts/cleaning/step2_fetch_articles.py
```

- Input:  `data/cleaned/Game.csv`
- Output: `data/cleaned/articles.jsonl`
- Expected: ~584 unique revids → ~584 HTTP calls, ~1-3 minutes total.

### Resumable

On re-run, any revid already present in the JSONL is skipped. When new participants
are added, the full pipeline is:

```bash
py scripts/cleaning/step1a_resolve_revids.py   # backfills revid_lookup.csv
py scripts/cleaning/step1b_apply_cleaning.py   # regenerates cleaned Game.csv
py scripts/cleaning/step2_fetch_articles.py    # fills in any new articles
```

### Validation

After the fetch loop completes, the script asserts that every unique non-null
`ArticleRevid` in the cleaned CSV has a non-empty `content` line in the JSONL.
Non-zero exit means validation failed — inspect the summary counts at the end.
```

- [ ] **Step 6.4: Commit**

```bash
git add scripts/cleaning/test_integration_live.py scripts/cleaning/README.md
git commit -m "test(cleaning): add live integration test for step2 + README"
```

---

## Task 7: Real-data dry run

Run the full stage-2 pipeline against the real cleaned CSV, produce the final JSONL, and commit it. This is the stage-2 equivalent of stage 1's final dry run that produced `data/cleaned/Game.csv`.

**Files:**
- Output: `data/cleaned/articles.jsonl`

- [ ] **Step 7.1: Confirm prerequisites**

Run: `py -c "import pandas as pd; df = pd.read_csv('data/cleaned/Game.csv'); print(df['ArticleRevid'].dropna().nunique())"`
Expected: `584`

- [ ] **Step 7.2: Run the stage-2 script**

Run: `py scripts/cleaning/step2_fetch_articles.py`
Expected output pattern:
```
[step2] 584 unique revids; 0 already fetched; 584 to fetch
[step2] [50/584] revid=... slug=... status=ok
...
[step2] [584/584] revid=... slug=... status=ok
[step2] summary: ok=584 not_found=0 error=0 skipped=0
[step2] validation OK -> data/cleaned/articles.jsonl
```

Runtime: 1-3 minutes. If any rows show `status=error` or `status=not_found`, re-run the script — errors often come from transient 429s; the resume will pick up where it left off. If persistent errors remain (>5 revids), investigate before proceeding.

- [ ] **Step 7.3: Sanity-check the output**

Run:
```bash
wc -l data/cleaned/articles.jsonl
```
Expected: `584 data/cleaned/articles.jsonl` (one line per unique revid).

Run:
```bash
py -c "import json; lines=open('data/cleaned/articles.jsonl',encoding='utf-8').readlines(); recs=[json.loads(l) for l in lines]; print('records:',len(recs)); print('fields:',sorted(recs[0].keys())); print('median content len:',sorted(len(r['content']) for r in recs)[len(recs)//2])"
```
Expected:
- `records: 584`
- `fields: ['article_slug', 'content', 'revid', 'timestamp']`
- `median content len:` a number greater than 500 (most Wikipedia article extracts are at least a few hundred characters; a tiny median would indicate empty-body issues).

- [ ] **Step 7.4: Commit the artefact**

```bash
git add data/cleaned/articles.jsonl
git commit -m "data(cleaned): stage 2 output — 584 article extracts"
```

- [ ] **Step 7.5: Run the full test suite one last time**

Run: `pytest scripts/cleaning -q`
Expected: all stage-1 + stage-2 unit tests pass; live tests skipped.

---

## Summary of what will exist when all tasks pass

**New source files:**
- `scripts/cleaning/article_store.py`
- `scripts/cleaning/step2_fetch_articles.py`

**Modified source files:**
- `scripts/cleaning/api_client.py` (gains `fetch_extract_by_revid` + `ExtractResult`)
- `scripts/cleaning/validation.py` (gains `validate_articles`)

**New tests:**
- `scripts/cleaning/test_article_store.py`
- `scripts/cleaning/test_step2.py`

**Modified tests:**
- `scripts/cleaning/test_api_client.py` (new `TestFetchExtractByRevid` class)
- `scripts/cleaning/test_validation.py` (new `TestValidateArticles` class)
- `scripts/cleaning/test_integration_live.py` (new live extract tests)

**Documentation:**
- `scripts/cleaning/README.md` (new "Stage 2" section)

**Data artefact:**
- `data/cleaned/articles.jsonl` (584 lines, ~2-10 MB depending on article lengths)

**Commits (7, one per task):**
1. `feat(cleaning): add fetch_extract_by_revid to api_client`
2. `feat(cleaning): add article_store for JSONL persistence`
3. `feat(cleaning): add step2 orchestrator for article fetch`
4. `feat(cleaning): add validate_articles stage-2 gate`
5. `feat(cleaning): wire validate_articles into step2 CLI`
6. `test(cleaning): add live integration test for step2 + README`
7. `data(cleaned): stage 2 output — 584 article extracts`

---

## Self-Review Checklist (run before handoff)

- **Spec coverage:** Every bullet in spec section 6 (inputs/outputs, API call, dedup/resume, reliability, validation) is implemented by one of Tasks 1-5. Dry run (Task 7) verifies real-world behavior. ✓
- **Placeholder scan:** No `TBD`, `fill in details`, `handle edge cases` as a step. Every step has concrete code or an exact command. ✓
- **Type consistency:** `fetch_extract_by_revid` returns `ExtractResult{content, status, error}` — referenced uniformly in Tasks 1, 3. `ARTICLE_FIELDS = ("article_slug", "revid", "timestamp", "content")` — referenced uniformly in Tasks 2, 3, 4. `resolve_all_articles(cleaned_df, out_path, *, session, fetch_fn=None)` — signature stable across Tasks 3, 5. `validate_articles(cleaned_df, articles_path)` — stable across Tasks 4, 5. ✓
- **YAGNI:** No parallelism, no async, no caching beyond the JSONL-as-resume-log. Reuses the stage-1 retry/backoff code pattern (paste-level duplication of ~15 lines is fine for the second — and only other — API function; abstract on rule of three).
- **Re-runnability:** Task 3's resume test (`test_resume_skips_already_fetched_revids`) enforces that re-running the script is a no-op for already-fetched revids. Directly supports the user's constraint that more participants will be added and the pipeline must stay re-runnable.

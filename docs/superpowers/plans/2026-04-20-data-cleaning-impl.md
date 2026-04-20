# Data Cleaning Pipeline (Stages 1a + 1b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `data/cleaned/Game.csv` where every non-practice `article_open` row has a correct `ArticleRevid` and `WikipediaUrl`, resolved via the MediaWiki API using the timestamp-bracketed revision query, with test users (IDs 69, 70) removed.

**Architecture:** Two-script pipeline under `scripts/cleaning/`. `step1a_resolve_revids.py` resolves unique `(slug, timestamp)` pairs into revids via MediaWiki and writes a resumable `revid_lookup.csv`. `step1b_apply_cleaning.py` joins the lookup back into the dirty CSV, removes test users, normalises timestamps, builds URLs, and runs a validation gate before writing `data/cleaned/Game.csv`. Stage 2 (article content fetch) is deferred - design only, not implemented here.

**Tech Stack:** Python 3, `requests` (HTTP), `pandas` (CSV), `pytest` (tests). Spec file: `docs/superpowers/specs/2026-04-20-data-cleaning-design.html`.

---

## File Structure

New files:
```
pytest.ini                                          ← pytest config (pythonpath, markers)
requirements.txt                                    ← add requests, pytest
scripts/cleaning/
  __init__.py                                       ← empty, marks package
  url_helpers.py                                    ← build_wikipedia_url, normalise_timestamp
  api_client.py                                     ← session factory, fetch_revision_at (with retry)
  lookup_store.py                                   ← load/save revid_lookup.csv, extract_unique_pairs
  cleaning_ops.py                                   ← filter_test_users, normalise_timestamp_col,
                                                      merge_revids, build_wikipedia_urls_col
  validation.py                                     ← validate_cleaned (raises on violation)
  step1a_resolve_revids.py                          ← CLI: orchestrator for stage 1a
  step1b_apply_cleaning.py                          ← CLI: orchestrator for stage 1b
  test_url_helpers.py
  test_api_client.py
  test_lookup_store.py
  test_cleaning_ops.py
  test_validation.py
  test_step1a.py
  test_step1b.py
  test_integration_live.py                          ← @pytest.mark.live, opt-in
  README.md
```

Note: tests are colocated with source (each file `foo.py` has `test_foo.py` next to it), matching user convention. Imports use unqualified module names (`from api_client import ...`) matching the existing `from helpers import ...` pattern in `scripts/`.

---

## Task 1: Bootstrap package + test config

**Files:**
- Create: `scripts/cleaning/__init__.py`
- Create: `pytest.ini`
- Modify: `requirements.txt`

- [ ] **Step 1: Create empty package init**

Create `scripts/cleaning/__init__.py` with content:
```python
```
(empty file; marks the directory as a package for tooling but imports stay unqualified.)

- [ ] **Step 2: Create pytest.ini at project root**

Create `pytest.ini`:
```ini
[pytest]
pythonpath = scripts/cleaning
testpaths = scripts/cleaning
markers =
    live: tests that hit the real MediaWiki API (opt-in; may be slow). Run with `pytest -m live`.
addopts = -m "not live"
```

- [ ] **Step 3: Add runtime deps to requirements.txt**

Append to `requirements.txt`:
```
requests
pytest
```

- [ ] **Step 4: Verify pytest discovers the empty package**

Run: `pytest scripts/cleaning/ -v --collect-only`
Expected: exits 0, prints `no tests ran in X.XXs` (since no tests exist yet). If it errors, the config is wrong.

- [ ] **Step 5: Commit**

```bash
git add pytest.ini requirements.txt scripts/cleaning/__init__.py
git commit -m "chore(cleaning): bootstrap scripts/cleaning package + pytest config"
```

---

## Task 2: Pure helpers (URL + timestamp)

**Files:**
- Create: `scripts/cleaning/url_helpers.py`
- Create: `scripts/cleaning/test_url_helpers.py`

- [ ] **Step 1: Write failing tests**

Create `scripts/cleaning/test_url_helpers.py`:
```python
import pandas as pd
import pytest
from url_helpers import build_wikipedia_url, normalise_timestamp


class TestBuildWikipediaUrl:
    def test_simple_slug(self):
        assert build_wikipedia_url("Capybara", 1349431902) == \
            "https://en.wikipedia.org/wiki/Capybara?oldid=1349431902"

    def test_slug_with_underscores(self):
        assert build_wikipedia_url("Aquatic_plants", 999) == \
            "https://en.wikipedia.org/wiki/Aquatic_plants?oldid=999"

    def test_slug_with_parentheses(self):
        assert build_wikipedia_url("Browsing_(herbivory)", 1341346387) == \
            "https://en.wikipedia.org/wiki/Browsing_(herbivory)?oldid=1341346387"

    def test_revid_as_int_not_float(self):
        result = build_wikipedia_url("X", 1234567890)
        assert "1234567890" in result
        assert ".0" not in result


class TestNormaliseTimestamp:
    def test_already_canonical(self):
        assert normalise_timestamp("2026-04-14T13:15:03.415Z") == \
            "2026-04-14T13:15:03.415Z"

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2026-04-14T13:15:03.415Z")
        assert normalise_timestamp(ts) == "2026-04-14T13:15:03.415Z"

    def test_microsecond_precision_truncated_to_ms(self):
        # 6 decimals in → 3 decimals out
        assert normalise_timestamp("2026-04-14T13:15:03.415678Z") == \
            "2026-04-14T13:15:03.415Z"

    def test_no_fractional_seconds_zero_padded(self):
        assert normalise_timestamp("2026-04-14T13:15:03Z") == \
            "2026-04-14T13:15:03.000Z"

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError):
            normalise_timestamp("2026-04-14 13:15:03")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest scripts/cleaning/test_url_helpers.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'url_helpers'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/cleaning/url_helpers.py`:
```python
"""Pure helpers for URL construction and timestamp normalisation."""
from __future__ import annotations

import pandas as pd

_URL_TEMPLATE = "https://en.wikipedia.org/wiki/{slug}?oldid={revid}"


def build_wikipedia_url(slug: str, revid: int) -> str:
    return _URL_TEMPLATE.format(slug=slug, revid=int(revid))


def normalise_timestamp(raw) -> str:
    ts = pd.to_datetime(raw, utc=True, errors="raise")
    if ts.tzinfo is None:
        raise ValueError(f"Naive timestamp rejected: {raw!r}")
    ms = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"
    return ms
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest scripts/cleaning/test_url_helpers.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/cleaning/url_helpers.py scripts/cleaning/test_url_helpers.py
git commit -m "feat(cleaning): add URL builder and timestamp normaliser helpers"
```

---

## Task 3: API client (session + fetch_revision_at with retry)

**Files:**
- Create: `scripts/cleaning/api_client.py`
- Create: `scripts/cleaning/test_api_client.py`

- [ ] **Step 1: Write failing tests (session factory + happy path + not_found + redirect + retry)**

Create `scripts/cleaning/test_api_client.py`:
```python
from unittest.mock import MagicMock, patch

import pytest
import requests

from api_client import build_session, fetch_revision_at


def _mock_response(json_data, status_code=200):
    m = MagicMock(spec=requests.Response)
    m.status_code = status_code
    m.json.return_value = json_data
    m.raise_for_status = MagicMock()
    if status_code >= 500 or status_code == 429:
        m.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return m


class TestBuildSession:
    def test_has_user_agent(self):
        s = build_session()
        assert "WikipediaTaskAnalysis" in s.headers["User-Agent"]
        assert "elay96@gmail.com" in s.headers["User-Agent"]


class TestFetchRevisionAt:
    def test_happy_path(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "1234": {
                        "pageid": 1234,
                        "title": "Capybara",
                        "revisions": [{"revid": 1349431902, "timestamp": "2026-04-14T13:00:00Z"}],
                    }
                }
            }
        })
        result = fetch_revision_at(session, "Capybara", "2026-04-14T13:15:03.415Z")
        assert result["revid"] == 1349431902
        assert result["status"] == "ok"
        assert result["error"] is None

    def test_redirect_is_transparently_followed(self):
        # redirects=1 means the API response has already resolved to target.
        # We assert we accept the response when the title differs from the input slug.
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "redirects": [{"from": "Aquatic_plants", "to": "Aquatic_plant"}],
                "pages": {
                    "9876": {
                        "pageid": 9876,
                        "title": "Aquatic_plant",
                        "revisions": [{"revid": 1340000000, "timestamp": "2026-04-14T12:00:00Z"}],
                    }
                }
            }
        })
        result = fetch_revision_at(session, "Aquatic_plants", "2026-04-14T13:18:30.751Z")
        assert result["revid"] == 1340000000
        assert result["status"] == "ok"

    def test_not_found_returns_status_not_found(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "-1": {"ns": 0, "title": "NonexistentPage_xyz", "missing": ""}
                }
            }
        })
        result = fetch_revision_at(session, "NonexistentPage_xyz", "2026-04-14T13:00:00Z")
        assert result["revid"] is None
        assert result["status"] == "not_found"

    def test_page_exists_but_no_revision_before_timestamp(self):
        # Page exists but no revision at-or-before the timestamp (edge case).
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "1": {"pageid": 1, "title": "X", "revisions": []}
                }
            }
        })
        result = fetch_revision_at(session, "X", "2000-01-01T00:00:00Z")
        assert result["revid"] is None
        assert result["status"] == "not_found"

    def test_retries_then_succeeds(self):
        session = MagicMock()
        # First two calls raise 503, third succeeds.
        ok_response = _mock_response({
            "query": {"pages": {"1": {"pageid": 1, "title": "X",
                                        "revisions": [{"revid": 42, "timestamp": "2026-01-01T00:00:00Z"}]}}}
        })
        fail = _mock_response({}, status_code=503)
        session.get.side_effect = [fail, fail, ok_response]
        with patch("api_client.time.sleep"):  # don't actually sleep
            result = fetch_revision_at(session, "X", "2026-01-02T00:00:00Z")
        assert result["revid"] == 42
        assert session.get.call_count == 3

    def test_gives_up_after_3_retries(self):
        session = MagicMock()
        session.get.return_value = _mock_response({}, status_code=500)
        with patch("api_client.time.sleep"):
            result = fetch_revision_at(session, "X", "2026-01-02T00:00:00Z")
        assert result["status"] == "error"
        assert result["revid"] is None
        assert "500" in (result["error"] or "")
        assert session.get.call_count == 3
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest scripts/cleaning/test_api_client.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'api_client'`

- [ ] **Step 3: Write implementation**

Create `scripts/cleaning/api_client.py`:
```python
"""MediaWiki API client: session factory + revision-at-timestamp resolver."""
from __future__ import annotations

import time
from typing import Optional, TypedDict

import requests

API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WikipediaTaskAnalysis/1.0 (elay96@gmail.com)"
TIMEOUT_SEC = 30
THROTTLE_SEC = 0.05
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # 1s, 2s, 4s


class RevisionResult(TypedDict):
    revid: Optional[int]
    status: str  # 'ok' | 'not_found' | 'error'
    error: Optional[str]


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_revision_at(session: requests.Session, slug: str, timestamp: str) -> RevisionResult:
    """Return the revision that was current at `timestamp` for `slug`.

    Follows redirects automatically (redirects=1). Retries on 429/5xx with
    exponential backoff. Returns ok/not_found/error per the RevisionResult contract.
    """
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": slug,
        "rvstart": timestamp,
        "rvdir": "older",
        "rvlimit": 1,
        "rvprop": "ids|timestamp",
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
                time.sleep(BACKOFF_BASE * (2 ** attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_error = str(e)
            time.sleep(BACKOFF_BASE * (2 ** attempt))
    else:
        return {"revid": None, "status": "error", "error": last_error}

    time.sleep(THROTTLE_SEC)

    pages = (data.get("query") or {}).get("pages") or {}
    for _, page in pages.items():
        if "missing" in page:
            return {"revid": None, "status": "not_found", "error": None}
        revisions = page.get("revisions") or []
        if not revisions:
            return {"revid": None, "status": "not_found", "error": None}
        return {"revid": int(revisions[0]["revid"]), "status": "ok", "error": None}

    return {"revid": None, "status": "error", "error": "unexpected response shape"}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest scripts/cleaning/test_api_client.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/cleaning/api_client.py scripts/cleaning/test_api_client.py
git commit -m "feat(cleaning): add MediaWiki API client with retry/backoff"
```

---

## Task 4: Lookup store (load + save)

**Files:**
- Create: `scripts/cleaning/lookup_store.py`
- Create: `scripts/cleaning/test_lookup_store.py`

- [ ] **Step 1: Write failing tests**

Create `scripts/cleaning/test_lookup_store.py`:
```python
import pandas as pd
import pytest

from lookup_store import LOOKUP_COLUMNS, extract_unique_pairs, load_lookup, save_lookup


class TestLoadSaveRoundTrip:
    def test_empty_when_missing(self, tmp_path):
        df = load_lookup(tmp_path / "missing.csv")
        assert list(df.columns) == LOOKUP_COLUMNS
        assert len(df) == 0

    def test_round_trip(self, tmp_path):
        path = tmp_path / "lookup.csv"
        df = pd.DataFrame([{
            "slug": "Capybara",
            "timestamp": "2026-04-14T13:15:03.415Z",
            "revid": 1349431902,
            "status": "ok",
            "fetched_at": "2026-04-20T12:00:00.000Z",
        }])
        save_lookup(df, path)
        roundtripped = load_lookup(path)
        assert len(roundtripped) == 1
        assert roundtripped.iloc[0]["slug"] == "Capybara"
        assert int(roundtripped.iloc[0]["revid"]) == 1349431902

    def test_preserves_nulls_on_errored_rows(self, tmp_path):
        path = tmp_path / "lookup.csv"
        df = pd.DataFrame([{
            "slug": "Bad",
            "timestamp": "2026-04-14T13:15:03.415Z",
            "revid": None,
            "status": "error",
            "fetched_at": "2026-04-20T12:00:00.000Z",
        }])
        save_lookup(df, path)
        loaded = load_lookup(path)
        assert pd.isna(loaded.iloc[0]["revid"])
        assert loaded.iloc[0]["status"] == "error"


class TestExtractUniquePairs:
    def test_excludes_test_users(self):
        df = pd.DataFrame([
            {"ID": 1,  "Action": "article_open", "ArticleSlug": "A", "Time": "2026-04-14T13:00:00.000Z"},
            {"ID": 69, "Action": "article_open", "ArticleSlug": "A", "Time": "2026-04-14T13:00:00.000Z"},
            {"ID": 70, "Action": "article_open", "ArticleSlug": "B", "Time": "2026-04-14T13:00:00.000Z"},
        ])
        pairs = extract_unique_pairs(df)
        assert pairs == [("A", "2026-04-14T13:00:00.000Z")]

    def test_excludes_non_article_open_events(self):
        df = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "A", "Time": "2026-04-14T13:00:00.000Z"},
            {"ID": 1, "Action": "search",       "ArticleSlug": None, "Time": "2026-04-14T13:00:01.000Z"},
            {"ID": 1, "Action": "task_start",   "ArticleSlug": None, "Time": "2026-04-14T13:00:02.000Z"},
        ])
        pairs = extract_unique_pairs(df)
        assert pairs == [("A", "2026-04-14T13:00:00.000Z")]

    def test_dedupes_exact_pairs(self):
        df = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "A", "Time": "2026-04-14T13:00:00.000Z"},
            {"ID": 2, "Action": "article_open", "ArticleSlug": "A", "Time": "2026-04-14T13:00:00.000Z"},
        ])
        pairs = extract_unique_pairs(df)
        assert pairs == [("A", "2026-04-14T13:00:00.000Z")]

    def test_normalises_timestamps_before_dedup(self):
        # Same instant, different precision representations → single pair
        df = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "A", "Time": "2026-04-14T13:00:00.000Z"},
            {"ID": 2, "Action": "article_open", "ArticleSlug": "A", "Time": "2026-04-14T13:00:00.000000Z"},
        ])
        pairs = extract_unique_pairs(df)
        assert len(pairs) == 1

    def test_output_sorted_for_reproducibility(self):
        df = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "B", "Time": "2026-04-14T14:00:00.000Z"},
            {"ID": 1, "Action": "article_open", "ArticleSlug": "A", "Time": "2026-04-14T13:00:00.000Z"},
        ])
        pairs = extract_unique_pairs(df)
        assert pairs == [
            ("A", "2026-04-14T13:00:00.000Z"),
            ("B", "2026-04-14T14:00:00.000Z"),
        ]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest scripts/cleaning/test_lookup_store.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'lookup_store'`

- [ ] **Step 3: Write implementation**

Create `scripts/cleaning/lookup_store.py`:
```python
"""Read/write the revid lookup CSV and extract unique (slug, timestamp) pairs."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from url_helpers import normalise_timestamp

LOOKUP_COLUMNS = ["slug", "timestamp", "revid", "status", "fetched_at"]
TEST_USER_IDS = {69, 70}


def load_lookup(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LOOKUP_COLUMNS)
    df = pd.read_csv(path, dtype={"slug": str, "timestamp": str, "status": str, "fetched_at": str})
    for col in LOOKUP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[LOOKUP_COLUMNS]


def save_lookup(df: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[LOOKUP_COLUMNS].to_csv(path, index=False)


def extract_unique_pairs(dirty_df: pd.DataFrame) -> List[Tuple[str, str]]:
    mask = (dirty_df["Action"] == "article_open") & (~dirty_df["ID"].isin(TEST_USER_IDS))
    sub = dirty_df.loc[mask, ["ArticleSlug", "Time"]].dropna()
    pairs = {
        (str(row["ArticleSlug"]), normalise_timestamp(row["Time"]))
        for _, row in sub.iterrows()
    }
    return sorted(pairs)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest scripts/cleaning/test_lookup_store.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/cleaning/lookup_store.py scripts/cleaning/test_lookup_store.py
git commit -m "feat(cleaning): add lookup store + unique-pair extractor"
```

---

## Task 5: Step 1a orchestrator + CLI

**Files:**
- Create: `scripts/cleaning/step1a_resolve_revids.py`
- Create: `scripts/cleaning/test_step1a.py`

- [ ] **Step 1: Write failing tests**

Create `scripts/cleaning/test_step1a.py`:
```python
from unittest.mock import MagicMock

import pandas as pd

from step1a_resolve_revids import resolve_all


def _fake_api(slug, timestamp):
    # Deterministic fake: revid = hash-ish of slug
    canned = {
        "Capybara":       {"revid": 1349431902, "status": "ok",        "error": None},
        "Aquatic_plants": {"revid": 1340000000, "status": "ok",        "error": None},
        "Missing":        {"revid": None,       "status": "not_found", "error": None},
    }
    return canned.get(slug, {"revid": None, "status": "error", "error": "unknown slug"})


class TestResolveAll:
    def test_resolves_all_pairs_from_scratch(self, tmp_path):
        dirty = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "Capybara",
             "Time": "2026-04-14T13:00:00.000Z"},
            {"ID": 1, "Action": "article_open", "ArticleSlug": "Aquatic_plants",
             "Time": "2026-04-14T13:05:00.000Z"},
            {"ID": 69, "Action": "article_open", "ArticleSlug": "SHOULD_BE_IGNORED",
             "Time": "2026-04-14T13:00:00.000Z"},
        ])
        lookup_path = tmp_path / "lookup.csv"
        session = MagicMock()
        result = resolve_all(dirty, lookup_path, session=session, fetch_fn=_fake_api)
        assert len(result) == 2
        assert set(result["status"]) == {"ok"}
        assert set(result["slug"]) == {"Capybara", "Aquatic_plants"}
        assert lookup_path.exists()

    def test_resumes_by_skipping_ok_entries(self, tmp_path):
        dirty = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "Capybara",
             "Time": "2026-04-14T13:00:00.000Z"},
            {"ID": 1, "Action": "article_open", "ArticleSlug": "Aquatic_plants",
             "Time": "2026-04-14T13:05:00.000Z"},
        ])
        lookup_path = tmp_path / "lookup.csv"
        # Pre-seed: Capybara already resolved, Aquatic_plants errored (should retry)
        pre = pd.DataFrame([
            {"slug": "Capybara", "timestamp": "2026-04-14T13:00:00.000Z",
             "revid": 111, "status": "ok", "fetched_at": "2026-04-19T00:00:00.000Z"},
            {"slug": "Aquatic_plants", "timestamp": "2026-04-14T13:05:00.000Z",
             "revid": None, "status": "error", "fetched_at": "2026-04-19T00:00:00.000Z"},
        ])
        pre.to_csv(lookup_path, index=False)

        call_log = []
        def tracking_fetch(slug, timestamp):
            call_log.append(slug)
            return _fake_api(slug, timestamp)

        session = MagicMock()
        result = resolve_all(dirty, lookup_path, session=session, fetch_fn=tracking_fetch)
        assert call_log == ["Aquatic_plants"]  # Capybara skipped
        cap_row = result[result["slug"] == "Capybara"].iloc[0]
        assert int(cap_row["revid"]) == 111  # preserved
        aq_row = result[result["slug"] == "Aquatic_plants"].iloc[0]
        assert aq_row["status"] == "ok"
        assert int(aq_row["revid"]) == 1340000000

    def test_records_not_found_and_error(self, tmp_path):
        dirty = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "Missing",
             "Time": "2026-04-14T13:00:00.000Z"},
            {"ID": 1, "Action": "article_open", "ArticleSlug": "Unknown",
             "Time": "2026-04-14T13:05:00.000Z"},
        ])
        session = MagicMock()
        result = resolve_all(dirty, tmp_path / "lookup.csv", session=session, fetch_fn=_fake_api)
        statuses = dict(zip(result["slug"], result["status"]))
        assert statuses["Missing"] == "not_found"
        assert statuses["Unknown"] == "error"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest scripts/cleaning/test_step1a.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'step1a_resolve_revids'`

- [ ] **Step 3: Write implementation**

Create `scripts/cleaning/step1a_resolve_revids.py`:
```python
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
    """Resolve revids for every unique (slug, timestamp) pair. Resumable.

    Args:
        dirty_df: full dirty DataFrame (test users will be filtered internally).
        lookup_path: where to read existing lookup and write updated one.
        session: requests.Session instance (or mock).
        fetch_fn: optional override for the API call, for testing.
                  Signature: fn(slug, timestamp) -> {revid, status, error}
    """
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
            print(f"[step1a] [{i}/{len(todo)}] {slug} @ {ts} → "
                  f"revid={res['revid']} status={res['status']}")

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=LOOKUP_COLUMNS)
        # Drop stale errored rows for the same keys, then append fresh attempts
        keys_added = set(zip(new_df["slug"], new_df["timestamp"]))
        keep_mask = ~existing[["slug", "timestamp"]].apply(tuple, axis=1).isin(keys_added)
        existing = existing[keep_mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest scripts/cleaning/test_step1a.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/cleaning/step1a_resolve_revids.py scripts/cleaning/test_step1a.py
git commit -m "feat(cleaning): add step1a orchestrator + CLI (resumable revid resolution)"
```

---

## Task 6: Live integration test

**Files:**
- Create: `scripts/cleaning/test_integration_live.py`

- [ ] **Step 1: Write the test**

Create `scripts/cleaning/test_integration_live.py`:
```python
"""Live integration tests that hit the real MediaWiki API.
Opt-in: run with `pytest -m live`. Skipped by default per pytest.ini addopts.
"""
import pytest

from api_client import build_session, fetch_revision_at


@pytest.mark.live
class TestLiveMediaWikiAPI:
    def test_capybara_happy_path(self):
        session = build_session()
        result = fetch_revision_at(session, "Capybara", "2026-04-14T13:15:03.415Z")
        assert result["status"] == "ok"
        assert isinstance(result["revid"], int)
        assert result["revid"] > 0
        assert result["error"] is None

    def test_aquatic_plants_redirect_resolves(self):
        # "Aquatic_plants" redirects to "Aquatic_plant" on en.wikipedia.
        # With redirects=1, we expect an ok status and a valid revid of the target.
        session = build_session()
        result = fetch_revision_at(session, "Aquatic_plants", "2026-04-14T13:18:30.751Z")
        assert result["status"] == "ok"
        assert isinstance(result["revid"], int)

    def test_nonexistent_page_returns_not_found(self):
        session = build_session()
        result = fetch_revision_at(
            session,
            "ThisPageDefinitelyDoesNotExist_xyzqwerty_2026",
            "2026-04-14T13:00:00.000Z",
        )
        assert result["status"] == "not_found"
        assert result["revid"] is None
```

- [ ] **Step 2: Run the live tests (requires internet)**

Run: `pytest scripts/cleaning/test_integration_live.py -m live -v`
Expected: 3 passed (in ~2-5s total, depending on network)

If the Capybara or Aquatic_plants page genuinely lacks a revision before the 2026-04-14 timestamp (unlikely - these are established articles), the test will fail with `status=not_found` and you should adjust the timestamp to a known-past date like `2024-01-01T00:00:00Z`.

- [ ] **Step 3: Verify live tests are skipped by default**

Run: `pytest scripts/cleaning/ -v`
Expected: the `test_integration_live.py::*` tests appear as `deselected` or skipped; non-live tests pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/cleaning/test_integration_live.py
git commit -m "test(cleaning): add opt-in live API integration tests"
```

---

## Task 7: Cleaning ops - filter users + normalise timestamp column

**Files:**
- Create: `scripts/cleaning/cleaning_ops.py`
- Create: `scripts/cleaning/test_cleaning_ops.py`

- [ ] **Step 1: Write failing tests**

Create `scripts/cleaning/test_cleaning_ops.py`:
```python
import pandas as pd
import pytest

from cleaning_ops import (
    build_wikipedia_urls_col,
    filter_test_users,
    merge_revids,
    normalise_timestamp_col,
)


class TestFilterTestUsers:
    def test_drops_69_and_70(self):
        df = pd.DataFrame([
            {"ID": 1, "Action": "task_start"},
            {"ID": 69, "Action": "task_start"},
            {"ID": 70, "Action": "task_start"},
            {"ID": 5, "Action": "task_start"},
        ])
        result = filter_test_users(df)
        assert set(result["ID"]) == {1, 5}
        assert len(result) == 2

    def test_preserves_row_order_of_kept_rows(self):
        df = pd.DataFrame([
            {"ID": 1, "idx": 0},
            {"ID": 70, "idx": 1},
            {"ID": 2, "idx": 2},
            {"ID": 69, "idx": 3},
            {"ID": 3, "idx": 4},
        ])
        result = filter_test_users(df)
        assert list(result["idx"]) == [0, 2, 4]


class TestNormaliseTimestampCol:
    def test_canonicalises_mixed_precision(self):
        df = pd.DataFrame({"Time": [
            "2026-04-14T13:15:03.415Z",
            "2026-04-14T13:15:03.415678Z",
            "2026-04-14T13:15:03Z",
        ]})
        result = normalise_timestamp_col(df)
        assert list(result["Time"]) == [
            "2026-04-14T13:15:03.415Z",
            "2026-04-14T13:15:03.415Z",
            "2026-04-14T13:15:03.000Z",
        ]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"Time": ["2026-04-14T13:15:03Z"]})
        _ = normalise_timestamp_col(df)
        assert df.iloc[0]["Time"] == "2026-04-14T13:15:03Z"  # unchanged
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest scripts/cleaning/test_cleaning_ops.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'cleaning_ops'`

- [ ] **Step 3: Write partial implementation (users + timestamps only; merge/urls coming in Task 8)**

Create `scripts/cleaning/cleaning_ops.py`:
```python
"""Pure transformations applied by step1b to the dirty DataFrame."""
from __future__ import annotations

import pandas as pd

from lookup_store import TEST_USER_IDS
from url_helpers import build_wikipedia_url, normalise_timestamp


def filter_test_users(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["ID"].isin(TEST_USER_IDS)].copy()


def normalise_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Time"] = out["Time"].apply(normalise_timestamp)
    return out


def merge_revids(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError  # Task 8

def build_wikipedia_urls_col(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError  # Task 8
```

- [ ] **Step 4: Run the user + timestamp tests to verify pass**

Run: `pytest scripts/cleaning/test_cleaning_ops.py::TestFilterTestUsers scripts/cleaning/test_cleaning_ops.py::TestNormaliseTimestampCol -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/cleaning/cleaning_ops.py scripts/cleaning/test_cleaning_ops.py
git commit -m "feat(cleaning): add filter_test_users + normalise_timestamp_col"
```

---

## Task 8: Cleaning ops - merge revids + build URLs

**Files:**
- Modify: `scripts/cleaning/cleaning_ops.py`
- Modify: `scripts/cleaning/test_cleaning_ops.py`

- [ ] **Step 1: Add failing tests**

Append to `scripts/cleaning/test_cleaning_ops.py`:
```python
class TestMergeRevids:
    def test_fills_revid_on_article_open_rows(self):
        df = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "Capybara",
             "Time": "2026-04-14T13:00:00.000Z", "ArticleRevid": None},
            {"ID": 1, "Action": "search", "ArticleSlug": None,
             "Time": "2026-04-14T13:00:01.000Z", "ArticleRevid": None},
        ])
        lookup = pd.DataFrame([{
            "slug": "Capybara",
            "timestamp": "2026-04-14T13:00:00.000Z",
            "revid": 1349431902,
            "status": "ok",
            "fetched_at": "2026-04-20T12:00:00.000Z",
        }])
        result = merge_revids(df, lookup)
        assert int(result.iloc[0]["ArticleRevid"]) == 1349431902
        assert pd.isna(result.iloc[1]["ArticleRevid"])

    def test_leaves_revid_null_when_lookup_status_not_ok(self):
        df = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "Bad",
             "Time": "2026-04-14T13:00:00.000Z", "ArticleRevid": None},
        ])
        lookup = pd.DataFrame([{
            "slug": "Bad", "timestamp": "2026-04-14T13:00:00.000Z",
            "revid": None, "status": "error",
            "fetched_at": "2026-04-20T12:00:00.000Z",
        }])
        result = merge_revids(df, lookup)
        assert pd.isna(result.iloc[0]["ArticleRevid"])

    def test_preserves_row_order_and_length(self):
        df = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "A",
             "Time": "2026-04-14T13:00:00.000Z", "ArticleRevid": None},
            {"ID": 1, "Action": "search", "ArticleSlug": None,
             "Time": "2026-04-14T13:00:01.000Z", "ArticleRevid": None},
            {"ID": 1, "Action": "article_open", "ArticleSlug": "B",
             "Time": "2026-04-14T13:00:02.000Z", "ArticleRevid": None},
        ])
        lookup = pd.DataFrame([
            {"slug": "A", "timestamp": "2026-04-14T13:00:00.000Z",
             "revid": 111, "status": "ok", "fetched_at": "x"},
            {"slug": "B", "timestamp": "2026-04-14T13:00:02.000Z",
             "revid": 222, "status": "ok", "fetched_at": "x"},
        ])
        result = merge_revids(df, lookup)
        assert len(result) == 3
        assert int(result.iloc[0]["ArticleRevid"]) == 111
        assert pd.isna(result.iloc[1]["ArticleRevid"])
        assert int(result.iloc[2]["ArticleRevid"]) == 222


class TestBuildWikipediaUrlsCol:
    def test_populates_url_when_revid_present(self):
        df = pd.DataFrame([
            {"Action": "article_open", "ArticleSlug": "Capybara",
             "ArticleRevid": 1349431902, "WikipediaUrl": None},
        ])
        result = build_wikipedia_urls_col(df)
        assert result.iloc[0]["WikipediaUrl"] == \
            "https://en.wikipedia.org/wiki/Capybara?oldid=1349431902"

    def test_leaves_url_null_when_revid_missing(self):
        df = pd.DataFrame([
            {"Action": "article_open", "ArticleSlug": "Capybara",
             "ArticleRevid": None, "WikipediaUrl": None},
            {"Action": "search", "ArticleSlug": None,
             "ArticleRevid": None, "WikipediaUrl": None},
        ])
        result = build_wikipedia_urls_col(df)
        assert pd.isna(result.iloc[0]["WikipediaUrl"])
        assert pd.isna(result.iloc[1]["WikipediaUrl"])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest scripts/cleaning/test_cleaning_ops.py -v`
Expected: 5 FAIL (new ones) with `NotImplementedError`

- [ ] **Step 3: Implement merge_revids and build_wikipedia_urls_col**

Replace the two `raise NotImplementedError` stubs in `scripts/cleaning/cleaning_ops.py`:
```python
def merge_revids(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Left-join ok-status revids from `lookup` onto article_open rows in `df`.

    Matches on ArticleSlug==lookup.slug AND Time==lookup.timestamp.
    Only lookup rows with status=='ok' are used; others leave ArticleRevid null.
    """
    out = df.copy()
    ok = lookup[lookup["status"] == "ok"][["slug", "timestamp", "revid"]]
    merged = out.merge(
        ok.rename(columns={"slug": "ArticleSlug", "timestamp": "Time", "revid": "_revid_merged"}),
        on=["ArticleSlug", "Time"],
        how="left",
    )
    mask_article_open = merged["Action"] == "article_open"
    merged.loc[mask_article_open & merged["_revid_merged"].notna(), "ArticleRevid"] = \
        merged.loc[mask_article_open & merged["_revid_merged"].notna(), "_revid_merged"]
    return merged.drop(columns=["_revid_merged"])


def build_wikipedia_urls_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = out["ArticleRevid"].notna() & out["ArticleSlug"].notna()
    out.loc[mask, "WikipediaUrl"] = [
        build_wikipedia_url(slug, int(rev))
        for slug, rev in zip(out.loc[mask, "ArticleSlug"], out.loc[mask, "ArticleRevid"])
    ]
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest scripts/cleaning/test_cleaning_ops.py -v`
Expected: 9 passed total

- [ ] **Step 5: Commit**

```bash
git add scripts/cleaning/cleaning_ops.py scripts/cleaning/test_cleaning_ops.py
git commit -m "feat(cleaning): add merge_revids + build_wikipedia_urls_col"
```

---

## Task 9: Validation gate

**Files:**
- Create: `scripts/cleaning/validation.py`
- Create: `scripts/cleaning/test_validation.py`

- [ ] **Step 1: Write failing tests**

Create `scripts/cleaning/test_validation.py`:
```python
import pandas as pd
import pytest

from validation import ValidationError, validate_cleaned


def _good_row(**overrides):
    base = {
        "ID": 1,
        "IsPractice": None,           # real trial
        "Action": "article_open",
        "ArticleSlug": "Capybara",
        "Time": "2026-04-14T13:00:00.000Z",
        "ArticleRevid": 1349431902,
        "WikipediaUrl": "https://en.wikipedia.org/wiki/Capybara?oldid=1349431902",
    }
    base.update(overrides)
    return base


class TestValidateCleaned:
    def test_accepts_clean_df(self):
        df = pd.DataFrame([_good_row(), _good_row(ArticleSlug="Art",
                                                  ArticleRevid=999,
                                                  WikipediaUrl="https://en.wikipedia.org/wiki/Art?oldid=999")])
        validate_cleaned(df, original_row_count=3, removed_rows=1)
        # no raise = pass

    def test_rejects_test_user_rows(self):
        df = pd.DataFrame([_good_row(ID=69)])
        with pytest.raises(ValidationError, match="test user"):
            validate_cleaned(df, original_row_count=2, removed_rows=1)

    def test_rejects_when_too_many_nonpractice_article_opens_missing_revid(self):
        rows = [_good_row(ArticleRevid=None, WikipediaUrl=None) for _ in range(10)]
        rows.append(_good_row())  # 1/11 = 9% good; 91% missing ≫ 5% tolerance
        df = pd.DataFrame(rows)
        with pytest.raises(ValidationError, match="missing ArticleRevid"):
            validate_cleaned(df, original_row_count=12, removed_rows=1)

    def test_tolerates_practice_rows_missing_revid(self):
        # Practice rows are allowed to miss revids without failing the gate.
        real = [_good_row() for _ in range(20)]
        practice = [_good_row(IsPractice=1, ArticleRevid=None, WikipediaUrl=None)
                    for _ in range(5)]
        df = pd.DataFrame(real + practice)
        validate_cleaned(df, original_row_count=26, removed_rows=1)

    def test_rejects_malformed_url(self):
        df = pd.DataFrame([_good_row(WikipediaUrl="http://wikipedia.com/Capybara")])
        with pytest.raises(ValidationError, match="URL"):
            validate_cleaned(df, original_row_count=2, removed_rows=1)

    def test_rejects_wrong_row_count(self):
        df = pd.DataFrame([_good_row(), _good_row()])
        with pytest.raises(ValidationError, match="row count"):
            validate_cleaned(df, original_row_count=10, removed_rows=3)  # expect 7, got 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest scripts/cleaning/test_validation.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'validation'`

- [ ] **Step 3: Write implementation**

Create `scripts/cleaning/validation.py`:
```python
"""Validation gate: asserts the cleaned CSV satisfies our invariants."""
from __future__ import annotations

import re

import pandas as pd

from lookup_store import TEST_USER_IDS

MISSING_REVID_TOLERANCE = 0.05  # 5%
URL_PATTERN = re.compile(r"^https://en\.wikipedia\.org/wiki/.+\?oldid=\d+$")


class ValidationError(Exception):
    pass


def validate_cleaned(df: pd.DataFrame, *, original_row_count: int, removed_rows: int) -> None:
    # 1. No test user IDs
    if df["ID"].isin(TEST_USER_IDS).any():
        raise ValidationError("Cleaned DataFrame still contains test user rows (IDs 69/70).")

    # 2. Row count accounting
    expected = original_row_count - removed_rows
    if len(df) != expected:
        raise ValidationError(
            f"Unexpected row count: got {len(df)}, expected {expected} "
            f"(original={original_row_count} - removed={removed_rows})."
        )

    # 3. Non-practice article_open coverage of ArticleRevid
    is_practice = df["IsPractice"] == 1
    is_article_open = df["Action"] == "article_open"
    nonprac_opens = df[is_article_open & ~is_practice]
    if len(nonprac_opens) > 0:
        missing = nonprac_opens["ArticleRevid"].isna().sum()
        rate = missing / len(nonprac_opens)
        if rate > MISSING_REVID_TOLERANCE:
            raise ValidationError(
                f"{missing}/{len(nonprac_opens)} non-practice article_open rows missing "
                f"ArticleRevid ({rate:.1%}) exceeds tolerance ({MISSING_REVID_TOLERANCE:.0%})."
            )

    # 4. URL format
    urls = df["WikipediaUrl"].dropna()
    bad = urls[~urls.apply(lambda u: bool(URL_PATTERN.match(str(u))))]
    if len(bad) > 0:
        sample = bad.iloc[0]
        raise ValidationError(f"{len(bad)} rows have malformed WikipediaUrl; e.g. {sample!r}")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest scripts/cleaning/test_validation.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/cleaning/validation.py scripts/cleaning/test_validation.py
git commit -m "feat(cleaning): add validation gate for cleaned CSV invariants"
```

---

## Task 10: Step 1b orchestrator + CLI

**Files:**
- Create: `scripts/cleaning/step1b_apply_cleaning.py`
- Create: `scripts/cleaning/test_step1b.py`

- [ ] **Step 1: Write failing tests**

Create `scripts/cleaning/test_step1b.py`:
```python
import pandas as pd

from step1b_apply_cleaning import apply_cleaning


def _dirty_frame():
    return pd.DataFrame([
        # Real user, practice (Capybara landing)
        {"ID": 1, "IsPractice": 1, "Action": "article_open",
         "ArticleSlug": "Capybara", "Time": "2026-04-14T13:15:03.415678Z",
         "ArticleRevid": None, "WikipediaUrl": None},
        # Real user, real trial
        {"ID": 1, "IsPractice": None, "Action": "article_open",
         "ArticleSlug": "Art", "Time": "2026-04-14T13:20:00Z",
         "ArticleRevid": None, "WikipediaUrl": None},
        # Non-article_open (search)
        {"ID": 1, "IsPractice": None, "Action": "search",
         "ArticleSlug": None, "Time": "2026-04-14T13:19:00Z",
         "ArticleRevid": None, "WikipediaUrl": None},
        # Test user - should be dropped
        {"ID": 69, "IsPractice": None, "Action": "article_open",
         "ArticleSlug": "Shouldbegone", "Time": "2026-04-14T13:21:00Z",
         "ArticleRevid": None, "WikipediaUrl": None},
    ])


def _lookup_frame():
    # Note: timestamp matches the NORMALISED form of the dirty rows.
    return pd.DataFrame([
        {"slug": "Capybara", "timestamp": "2026-04-14T13:15:03.415Z",
         "revid": 111, "status": "ok", "fetched_at": "x"},
        {"slug": "Art", "timestamp": "2026-04-14T13:20:00.000Z",
         "revid": 222, "status": "ok", "fetched_at": "x"},
    ])


class TestApplyCleaning:
    def test_removes_test_users(self):
        result = apply_cleaning(_dirty_frame(), _lookup_frame())
        assert 69 not in set(result["ID"])

    def test_normalises_timestamps(self):
        result = apply_cleaning(_dirty_frame(), _lookup_frame())
        assert result.iloc[0]["Time"] == "2026-04-14T13:15:03.415Z"
        assert result.iloc[1]["Time"] == "2026-04-14T13:20:00.000Z"

    def test_populates_revid_and_url_on_article_open(self):
        result = apply_cleaning(_dirty_frame(), _lookup_frame())
        capy = result[(result["ArticleSlug"] == "Capybara")].iloc[0]
        art = result[(result["ArticleSlug"] == "Art")].iloc[0]
        assert int(capy["ArticleRevid"]) == 111
        assert capy["WikipediaUrl"] == "https://en.wikipedia.org/wiki/Capybara?oldid=111"
        assert int(art["ArticleRevid"]) == 222
        assert art["WikipediaUrl"] == "https://en.wikipedia.org/wiki/Art?oldid=222"

    def test_leaves_non_article_open_rows_null(self):
        result = apply_cleaning(_dirty_frame(), _lookup_frame())
        search = result[result["Action"] == "search"].iloc[0]
        assert pd.isna(search["ArticleRevid"])
        assert pd.isna(search["WikipediaUrl"])

    def test_validation_gate_runs(self):
        # If lookup is empty and every article_open is missing a revid, gate should fail.
        import pytest
        from validation import ValidationError
        empty_lookup = pd.DataFrame(columns=["slug", "timestamp", "revid", "status", "fetched_at"])
        with pytest.raises(ValidationError):
            apply_cleaning(_dirty_frame(), empty_lookup)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest scripts/cleaning/test_step1b.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'step1b_apply_cleaning'`

- [ ] **Step 3: Write implementation**

Create `scripts/cleaning/step1b_apply_cleaning.py`:
```python
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

    # Preserve exact original column order / set
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
    print(f"[step1b] wrote {len(cleaned)} rows → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest scripts/cleaning/test_step1b.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the full cleaning test suite (sanity check)**

Run: `pytest scripts/cleaning/ -v`
Expected: ~40 passed, 3 deselected (live tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/cleaning/step1b_apply_cleaning.py scripts/cleaning/test_step1b.py
git commit -m "feat(cleaning): add step1b orchestrator + CLI (apply cleaning + validation)"
```

---

## Task 11: README

**Files:**
- Create: `scripts/cleaning/README.md`

- [ ] **Step 1: Write README**

Create `scripts/cleaning/README.md`:
```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/cleaning/README.md
git commit -m "docs(cleaning): add README for the cleaning pipeline"
```

---

## Task 12: Real-data end-to-end dry run

**This is a manual verification task, not a code task.** Stop here for user review before continuing.

- [ ] **Step 1: Run stage 1a against the real dirty data**

Run: `py scripts/cleaning/step1a_resolve_revids.py`
Expected:
- Takes 30-90 seconds.
- Prints progress roughly every 50 pairs.
- Final summary line shows `ok=~280 not_found=0 error=0` (approximately - exact numbers depend on what Wikipedia returns).
- Writes `data/cleaned/revid_lookup.csv`.

Open `data/cleaned/revid_lookup.csv` in a spreadsheet or with `head`. Spot-check:
- Every row has a `revid` if `status=ok`.
- No `status=error` rows remain (if any, re-run the command - the retry logic + resume will handle transient issues).
- If some `not_found` appear, eyeball the slugs - they may be typos or deleted pages.

- [ ] **Step 2: Run stage 1b against the real dirty data + lookup**

Run: `py scripts/cleaning/step1b_apply_cleaning.py`
Expected:
- Exits 0.
- Prints `[step1b] wrote 4745 rows → data/cleaned/Game.csv` (or `4745 ± small delta` if any rows in dirty changed during development).
- No `ValidationError` raised.

If validation fails, the error message will tell you which invariant broke - fix the underlying issue rather than relaxing the gate.

- [ ] **Step 3: Manual spot-checks in the cleaned CSV**

Open `data/cleaned/Game.csv` and confirm:
- No rows with `ID` in {69, 70}.
- A handful of `article_open` rows have populated `ArticleRevid` + `WikipediaUrl`.
- Pick 3 random URLs and open them in a browser - the page content should look plausible for the given article (content at that revision).
- A known redirect case: find a row with `ArticleSlug=Aquatic_plants`. Its URL should open to the Aquatic plant article at the historical revision (Wikipedia silently resolves the redirect server-side).

- [ ] **Step 4: Decide about committing the data artifacts**

**Do not auto-commit `data/cleaned/` artifacts.** Show the user a git status and ask whether they want those committed. CSVs of experiment data may or may not belong in git based on repo policy.

If the user says yes, commit with:
```bash
git add data/cleaned/revid_lookup.csv data/cleaned/Game.csv
git commit -m "data(cleaning): add cleaned Game.csv + revid lookup (stage 1 output)"
```

- [ ] **Step 5: Hand off for stage 2 decision**

Stage 1 is complete. Surface the summary to the user:
- X rows in, Y rows out, Z test-user rows removed.
- N unique revids resolved; M not_found (if any).
- All validation invariants pass.

Ask: **"Stage 1 complete and verified. Should I proceed to design + implement stage 2 (article content fetch)?"** Do not start stage 2 without explicit approval - per spec checkpoint.

---

## Self-Review (performed by plan author)

**Spec coverage check:** Every section of `docs/superpowers/specs/2026-04-20-data-cleaning-design.html` has a corresponding task:
- §3 Architecture / file layout → Task 1 (bootstrap) + every subsequent file-creation task
- §4 step1a behaviour, API call, output schema, rate/retry, resume → Tasks 3, 4, 5
- §5 step1b operations (filter / normalise / join / url / validation / write) → Tasks 7, 8, 9, 10
- §6 step2 deferred - explicitly not implemented, acknowledged in Task 12 handoff
- §7 Testing strategy (unit + live integration + manual checkpoint) → Tasks 2-10 (unit), 6 (live), 12 (manual)
- §8 Performance notes → reflected in Task 12 expected runtime
- §9 Open assumptions → encoded as invariants in Task 9 validation gate

**Placeholder scan:** No `TBD`, `TODO`, "add appropriate", or "similar to Task N" references. Every code block is complete.

**Type/name consistency:**
- `fetch_revision_at(session, slug, timestamp)` - used consistently in api_client, step1a, and test_integration_live.
- `RevisionResult` TypedDict keys (`revid`, `status`, `error`) - used consistently across api_client and fake test fixtures.
- `LOOKUP_COLUMNS`, `TEST_USER_IDS` - defined in `lookup_store` and imported by cleaning_ops, validation, step1b.
- Function names in cleaning_ops (`filter_test_users`, `normalise_timestamp_col`, `merge_revids`, `build_wikipedia_urls_col`) match between declaration (Task 7/8) and usage (Task 10).
- `ValidationError` class imported in test_step1b matches the one defined in validation.

Plan is complete and internally consistent.

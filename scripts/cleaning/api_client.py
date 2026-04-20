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
        return {"revid": None, "status": "error", "error": last_error}

    time.sleep(THROTTLE_SEC)

    pages = (data.get("query") or {}).get("pages") or {}
    # Single-slug call (titles=slug), so pages dict has exactly one entry.
    for _, page in pages.items():
        if "missing" in page:
            return {"revid": None, "status": "not_found", "error": None}
        revisions = page.get("revisions") or []
        if not revisions:
            return {"revid": None, "status": "not_found", "error": None}
        return {"revid": int(revisions[0]["revid"]), "status": "ok", "error": None}

    return {"revid": None, "status": "error", "error": "unexpected response shape"}

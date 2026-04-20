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

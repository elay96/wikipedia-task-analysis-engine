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

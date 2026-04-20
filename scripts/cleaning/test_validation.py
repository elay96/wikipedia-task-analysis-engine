import json
from pathlib import Path

import pandas as pd
import pytest

from validation import ValidationError, validate_cleaned


def _good_row(**overrides):
    base = {
        "ID": 1,
        "IsPractice": None,
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

    def test_rejects_test_user_rows(self):
        df = pd.DataFrame([_good_row(ID=69)])
        with pytest.raises(ValidationError, match="test user"):
            validate_cleaned(df, original_row_count=2, removed_rows=1)

    def test_rejects_when_too_many_nonpractice_article_opens_missing_revid(self):
        rows = [_good_row(ArticleRevid=None, WikipediaUrl=None) for _ in range(10)]
        rows.append(_good_row())
        df = pd.DataFrame(rows)
        with pytest.raises(ValidationError, match="missing ArticleRevid"):
            validate_cleaned(df, original_row_count=12, removed_rows=1)

    def test_tolerates_practice_rows_missing_revid(self):
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
            validate_cleaned(df, original_row_count=10, removed_rows=3)

    def test_tolerates_practice_rows_with_string_is_practice(self):
        # IsPractice may come back from CSV as string "1" in some dtype-coercion paths.
        real = [_good_row() for _ in range(20)]
        practice = [_good_row(IsPractice="1", ArticleRevid=None, WikipediaUrl=None)
                    for _ in range(5)]
        df = pd.DataFrame(real + practice)
        validate_cleaned(df, original_row_count=26, removed_rows=1)


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

    def test_corrupted_jsonl_line_reports_missing_revid(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        good = json.dumps({"article_slug": "A1", "revid": 1, "timestamp": "t", "content": "body"})
        truncated = '{"article_slug": "A2", "revid":'  # crashed mid-write — should be skipped
        p.write_text(good + "\n" + truncated + "\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="missing"):
            validate_articles(_cleaned([1, 2]), p)

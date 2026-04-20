import json
from pathlib import Path

import pytest

from step3_build_wiki_texts import build_wiki_texts


def _write_jsonl(path: Path, rows):
    lines = [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class TestBuildWikiTexts:
    def test_single_revid_per_slug(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [
            {"article_slug": "Art", "revid": 1, "timestamp": "2026-04-20T10:00:00Z", "content": "A"},
            {"article_slug": "Biology", "revid": 2, "timestamp": "2026-04-20T10:00:00Z", "content": "B"},
        ])
        out = build_wiki_texts(p)
        assert out == {"Art": "A", "Biology": "B"}

    def test_picks_latest_by_timestamp_for_duplicate_slug(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [
            {"article_slug": "Art", "revid": 1, "timestamp": "2026-04-14T10:00:00Z", "content": "old"},
            {"article_slug": "Art", "revid": 2, "timestamp": "2026-04-20T10:00:00Z", "content": "new"},
        ])
        out = build_wiki_texts(p)
        assert out == {"Art": "new"}

    def test_tiebreak_on_revid_when_timestamps_equal(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [
            {"article_slug": "Art", "revid": 2, "timestamp": "2026-04-20T10:00:00Z", "content": "higher"},
            {"article_slug": "Art", "revid": 1, "timestamp": "2026-04-20T10:00:00Z", "content": "lower"},
        ])
        out = build_wiki_texts(p)
        assert out == {"Art": "higher"}

    def test_skips_malformed_lines(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        good = json.dumps({"article_slug": "Art", "revid": 1, "timestamp": "t", "content": "body"})
        p.write_text(good + "\n{\"article_slug\":\n", encoding="utf-8")
        out = build_wiki_texts(p)
        assert out == {"Art": "body"}

    def test_skips_empty_content(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [
            {"article_slug": "Art", "revid": 1, "timestamp": "t", "content": ""},
            {"article_slug": "Art", "revid": 2, "timestamp": "t", "content": "good"},
        ])
        out = build_wiki_texts(p)
        assert out == {"Art": "good"}

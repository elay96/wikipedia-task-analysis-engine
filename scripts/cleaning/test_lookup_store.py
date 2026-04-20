import pandas as pd

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

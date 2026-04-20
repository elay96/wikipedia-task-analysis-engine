import pandas as pd

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


    def test_revid_dtype_is_int64(self):
        df = pd.DataFrame([
            {"ID": 1, "Action": "article_open", "ArticleSlug": "A",
             "Time": "2026-04-14T13:00:00.000Z", "ArticleRevid": None},
        ])
        lookup = pd.DataFrame([
            {"slug": "A", "timestamp": "2026-04-14T13:00:00.000Z",
             "revid": 111, "status": "ok", "fetched_at": "x"},
        ])
        result = merge_revids(df, lookup)
        assert result["ArticleRevid"].dtype == "Int64"
        assert result.iloc[0]["ArticleRevid"] == 111


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

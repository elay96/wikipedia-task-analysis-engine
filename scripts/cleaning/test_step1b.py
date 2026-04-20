import pandas as pd

from step1b_apply_cleaning import apply_cleaning


def _dirty_frame():
    return pd.DataFrame([
        {"ID": 1, "IsPractice": 1, "Action": "article_open",
         "ArticleSlug": "Capybara", "Time": "2026-04-14T13:15:03.415678Z",
         "ArticleRevid": None, "WikipediaUrl": None},
        {"ID": 1, "IsPractice": None, "Action": "article_open",
         "ArticleSlug": "Art", "Time": "2026-04-14T13:20:00Z",
         "ArticleRevid": None, "WikipediaUrl": None},
        {"ID": 1, "IsPractice": None, "Action": "search",
         "ArticleSlug": None, "Time": "2026-04-14T13:19:00Z",
         "ArticleRevid": None, "WikipediaUrl": None},
        {"ID": 69, "IsPractice": None, "Action": "article_open",
         "ArticleSlug": "Shouldbegone", "Time": "2026-04-14T13:21:00Z",
         "ArticleRevid": None, "WikipediaUrl": None},
    ])


def _lookup_frame():
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
        import pytest
        from validation import ValidationError
        empty_lookup = pd.DataFrame(columns=["slug", "timestamp", "revid", "status", "fetched_at"])
        with pytest.raises(ValidationError):
            apply_cleaning(_dirty_frame(), empty_lookup)

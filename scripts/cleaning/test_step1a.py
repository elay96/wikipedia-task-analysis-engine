from unittest.mock import MagicMock

import pandas as pd

from step1a_resolve_revids import resolve_all


def _fake_api(slug, timestamp):
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

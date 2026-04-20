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

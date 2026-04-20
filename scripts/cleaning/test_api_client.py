from unittest.mock import MagicMock, patch

import requests

from api_client import build_session, fetch_revision_at


def _mock_response(json_data, status_code=200):
    m = MagicMock(spec=requests.Response)
    m.status_code = status_code
    m.json.return_value = json_data
    m.raise_for_status = MagicMock()
    return m


class TestBuildSession:
    def test_has_user_agent(self):
        s = build_session()
        assert "WikipediaTaskAnalysis" in s.headers["User-Agent"]
        assert "elay96@gmail.com" in s.headers["User-Agent"]


class TestFetchRevisionAt:
    def test_happy_path(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "1234": {
                        "pageid": 1234,
                        "title": "Capybara",
                        "revisions": [{"revid": 1349431902, "timestamp": "2026-04-14T13:00:00Z"}],
                    }
                }
            }
        })
        result = fetch_revision_at(session, "Capybara", "2026-04-14T13:15:03.415Z")
        assert result["revid"] == 1349431902
        assert result["status"] == "ok"
        assert result["error"] is None

    def test_redirect_is_transparently_followed(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "redirects": [{"from": "Aquatic_plants", "to": "Aquatic_plant"}],
                "pages": {
                    "9876": {
                        "pageid": 9876,
                        "title": "Aquatic_plant",
                        "revisions": [{"revid": 1340000000, "timestamp": "2026-04-14T12:00:00Z"}],
                    }
                }
            }
        })
        result = fetch_revision_at(session, "Aquatic_plants", "2026-04-14T13:18:30.751Z")
        assert result["revid"] == 1340000000
        assert result["status"] == "ok"

    def test_not_found_returns_status_not_found(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "-1": {"ns": 0, "title": "NonexistentPage_xyz", "missing": ""}
                }
            }
        })
        result = fetch_revision_at(session, "NonexistentPage_xyz", "2026-04-14T13:00:00Z")
        assert result["revid"] is None
        assert result["status"] == "not_found"

    def test_page_exists_but_no_revision_before_timestamp(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "1": {"pageid": 1, "title": "X", "revisions": []}
                }
            }
        })
        result = fetch_revision_at(session, "X", "2000-01-01T00:00:00Z")
        assert result["revid"] is None
        assert result["status"] == "not_found"

    def test_retries_then_succeeds(self):
        session = MagicMock()
        ok_response = _mock_response({
            "query": {"pages": {"1": {"pageid": 1, "title": "X",
                                        "revisions": [{"revid": 42, "timestamp": "2026-01-01T00:00:00Z"}]}}}
        })
        fail = _mock_response({}, status_code=503)
        session.get.side_effect = [fail, fail, ok_response]
        with patch("api_client.time.sleep"):
            result = fetch_revision_at(session, "X", "2026-01-02T00:00:00Z")
        assert result["revid"] == 42
        assert session.get.call_count == 3

    def test_gives_up_after_3_retries(self):
        session = MagicMock()
        session.get.return_value = _mock_response({}, status_code=500)
        with patch("api_client.time.sleep"):
            result = fetch_revision_at(session, "X", "2026-01-02T00:00:00Z")
        assert result["status"] == "error"
        assert result["revid"] is None
        assert result["error"] == "HTTP 500"
        assert session.get.call_count == 3


class TestFetchExtractByRevid:
    def test_happy_path_returns_content(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "1234": {
                        "pageid": 1234,
                        "title": "Capybara",
                        "extract": "The capybara (Hydrochoerus hydrochaeris) is the largest living rodent...",
                    }
                }
            }
        })
        from api_client import fetch_extract_by_revid
        result = fetch_extract_by_revid(session, 1349431902)
        assert result["status"] == "ok"
        assert result["content"].startswith("The capybara")
        assert result["error"] is None

    def test_missing_extract_returns_not_found(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "pages": {
                    "-1": {"ns": 0, "title": "NonexistentRev", "missing": ""}
                }
            }
        })
        from api_client import fetch_extract_by_revid
        result = fetch_extract_by_revid(session, 999999999999)
        assert result["status"] == "not_found"
        assert result["content"] is None

    def test_empty_extract_string_returns_not_found(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {"pages": {"1": {"pageid": 1, "title": "X", "extract": ""}}}
        })
        from api_client import fetch_extract_by_revid
        result = fetch_extract_by_revid(session, 42)
        assert result["status"] == "not_found"
        assert result["content"] is None

    def test_retries_then_succeeds_on_5xx(self):
        session = MagicMock()
        ok = _mock_response({
            "query": {"pages": {"1": {"pageid": 1, "title": "X", "extract": "body"}}}
        })
        fail = _mock_response({}, status_code=503)
        session.get.side_effect = [fail, fail, ok]
        from api_client import fetch_extract_by_revid
        with patch("api_client.time.sleep"):
            result = fetch_extract_by_revid(session, 42)
        assert result["status"] == "ok"
        assert result["content"] == "body"
        assert session.get.call_count == 3

    def test_gives_up_after_3_retries(self):
        session = MagicMock()
        session.get.return_value = _mock_response({}, status_code=500)
        from api_client import fetch_extract_by_revid
        with patch("api_client.time.sleep"):
            result = fetch_extract_by_revid(session, 42)
        assert result["status"] == "error"
        assert result["error"] == "HTTP 500"
        assert session.get.call_count == 3

    def test_sends_correct_params(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {"pages": {"1": {"pageid": 1, "title": "X", "extract": "body"}}}
        })
        from api_client import fetch_extract_by_revid
        fetch_extract_by_revid(session, 1349431902)
        _, kwargs = session.get.call_args
        params = kwargs["params"]
        assert params["action"] == "query"
        assert params["prop"] == "extracts"
        assert params["explaintext"] == 1
        assert params["exsectionformat"] == "plain"
        assert params["revids"] == 1349431902
        assert params["redirects"] == 1
        assert params["format"] == "json"

    def test_non_string_extract_returns_not_found(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {"pages": {"1": {"pageid": 1, "title": "X", "extract": ["not", "a", "string"]}}}
        })
        from api_client import fetch_extract_by_revid
        result = fetch_extract_by_revid(session, 42)
        assert result["status"] == "not_found"
        assert result["content"] is None

    def test_badrevids_returns_not_found(self):
        session = MagicMock()
        session.get.return_value = _mock_response({
            "query": {
                "badrevids": {"999999999999": {"revid": 999999999999, "missing": ""}}
            }
        })
        from api_client import fetch_extract_by_revid
        result = fetch_extract_by_revid(session, 999999999999)
        assert result["status"] == "not_found"
        assert result["content"] is None
        assert result["error"] is None

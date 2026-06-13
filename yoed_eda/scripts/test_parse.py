import json
import numpy as np
import pandas as pd
import parse


def test_decode_article_strips_proxy_prefix():
    assert parse.decode_article("/proxy/wiki/wiki/Chat_bot") == "Chat_bot"
    assert parse.decode_article("/proxy/wiki/wiki/AI_(disambiguation)") == "AI_(disambiguation)"


def test_clean_iso_strips_wrapping_quotes():
    assert parse.clean_iso('"""2025-11-29T10:26:31.074Z"""') == "2025-11-29T10:26:31.074Z"
    assert parse.clean_iso('"2025-11-29T10:26:31.074Z"') == "2025-11-29T10:26:31.074Z"


def test_flatten_visits_orders_by_start_and_computes_dwell():
    visits_json = json.dumps([
        {"id": 2, "session_id": "s1", "url": "/proxy/wiki/wiki/B", "title": "B",
         "start_time": 2000, "end_time": np.nan, "duration_ms": np.nan},
        {"id": 1, "session_id": "s1", "url": "/proxy/wiki/wiki/A", "title": "A",
         "start_time": 1000, "end_time": np.nan, "duration_ms": np.nan},
    ])
    row = {"participant_id": 7, "session_id": "s1",
           "ended_at": '"2025-01-01T00:00:05.000Z"',  # 5000 ms after A
           "page_visits_rows_json": visits_json}
    out = parse.flatten_visits(pd.DataFrame([row]))
    out = out.sort_values("order_in_session").reset_index(drop=True)
    assert list(out["article"]) == ["A", "B"]
    assert list(out["order_in_session"]) == [0, 1]
    # A's dwell = B.start - A.start = 1000 ms
    assert out.loc[0, "dwell_ms"] == 1000
    # B is last -> dwell from ended_at (1970 epoch parse): just assert finite & > 0
    assert out.loc[1, "dwell_ms"] > 0


def test_flatten_visits_flags_main_and_disambiguation():
    visits_json = json.dumps([
        {"id": 1, "session_id": "s1", "url": "/proxy/wiki/wiki/Main_Page", "title": "Main Page", "start_time": 1000},
        {"id": 2, "session_id": "s1", "url": "/proxy/wiki/wiki/AI_(disambiguation)", "title": "AI", "start_time": 2000},
    ])
    row = {"participant_id": 7, "session_id": "s1", "ended_at": '"2025-01-01T00:00:05.000Z"',
           "page_visits_rows_json": visits_json}
    out = parse.flatten_visits(pd.DataFrame([row])).sort_values("order_in_session")
    assert list(out["is_main_page"]) == [True, False]
    assert list(out["is_disambiguation"]) == [False, True]


def test_flatten_searches_extracts_queries():
    sq = json.dumps([{"id": 1, "session_id": "s1", "query": "Chat bot", "timestamp": 1000, "results_count": np.nan}])
    row = {"participant_id": 7, "session_id": "s1", "search_queries_rows_json": sq}
    out = parse.flatten_searches(pd.DataFrame([row]))
    assert list(out["query"]) == ["Chat bot"]
    assert list(out["participant_id"]) == [7]


def test_participants_table_keeps_measures_and_meta():
    df = pd.DataFrame([{
        "participant_id": 7, "Curiosity - Score": 23, "GF - Score": 10,
        "session_id": "s1", "started_at": '"2025-01-01T00:00:00.000Z"',
        "ended_at": '"2025-01-01T00:01:00.000Z"', "total_pages_visited": 11,
        "total_searches": 7, "user_question": "q", "page_visits_rows_json": "[]",
        "search_queries_rows_json": "[]",
    }])
    out = parse.participants_table(df)
    assert "page_visits_rows_json" not in out.columns
    assert out.loc[0, "Curiosity - Score"] == 23
    assert out.loc[0, "started_at"] == "2025-01-01T00:00:00.000Z"  # quotes stripped

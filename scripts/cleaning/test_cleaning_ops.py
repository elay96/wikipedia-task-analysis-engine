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

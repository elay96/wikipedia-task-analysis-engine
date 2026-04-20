import pandas as pd
import pytest
from url_helpers import build_wikipedia_url, normalise_timestamp


class TestBuildWikipediaUrl:
    def test_simple_slug(self):
        assert build_wikipedia_url("Capybara", 1349431902) == \
            "https://en.wikipedia.org/wiki/Capybara?oldid=1349431902"

    def test_slug_with_underscores(self):
        assert build_wikipedia_url("Aquatic_plants", 999) == \
            "https://en.wikipedia.org/wiki/Aquatic_plants?oldid=999"

    def test_slug_with_parentheses(self):
        assert build_wikipedia_url("Browsing_(herbivory)", 1341346387) == \
            "https://en.wikipedia.org/wiki/Browsing_(herbivory)?oldid=1341346387"

    def test_revid_as_int_not_float(self):
        result = build_wikipedia_url("X", 1234567890)
        assert "1234567890" in result
        assert ".0" not in result


class TestNormaliseTimestamp:
    def test_already_canonical(self):
        assert normalise_timestamp("2026-04-14T13:15:03.415Z") == \
            "2026-04-14T13:15:03.415Z"

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2026-04-14T13:15:03.415Z")
        assert normalise_timestamp(ts) == "2026-04-14T13:15:03.415Z"

    def test_microsecond_precision_truncated_to_ms(self):
        # 6 decimals in -> 3 decimals out
        assert normalise_timestamp("2026-04-14T13:15:03.415678Z") == \
            "2026-04-14T13:15:03.415Z"

    def test_no_fractional_seconds_zero_padded(self):
        assert normalise_timestamp("2026-04-14T13:15:03Z") == \
            "2026-04-14T13:15:03.000Z"

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError):
            normalise_timestamp("2026-04-14 13:15:03")

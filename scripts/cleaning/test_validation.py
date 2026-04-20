import pandas as pd
import pytest

from validation import ValidationError, validate_cleaned


def _good_row(**overrides):
    base = {
        "ID": 1,
        "IsPractice": None,
        "Action": "article_open",
        "ArticleSlug": "Capybara",
        "Time": "2026-04-14T13:00:00.000Z",
        "ArticleRevid": 1349431902,
        "WikipediaUrl": "https://en.wikipedia.org/wiki/Capybara?oldid=1349431902",
    }
    base.update(overrides)
    return base


class TestValidateCleaned:
    def test_accepts_clean_df(self):
        df = pd.DataFrame([_good_row(), _good_row(ArticleSlug="Art",
                                                  ArticleRevid=999,
                                                  WikipediaUrl="https://en.wikipedia.org/wiki/Art?oldid=999")])
        validate_cleaned(df, original_row_count=3, removed_rows=1)

    def test_rejects_test_user_rows(self):
        df = pd.DataFrame([_good_row(ID=69)])
        with pytest.raises(ValidationError, match="test user"):
            validate_cleaned(df, original_row_count=2, removed_rows=1)

    def test_rejects_when_too_many_nonpractice_article_opens_missing_revid(self):
        rows = [_good_row(ArticleRevid=None, WikipediaUrl=None) for _ in range(10)]
        rows.append(_good_row())
        df = pd.DataFrame(rows)
        with pytest.raises(ValidationError, match="missing ArticleRevid"):
            validate_cleaned(df, original_row_count=12, removed_rows=1)

    def test_tolerates_practice_rows_missing_revid(self):
        real = [_good_row() for _ in range(20)]
        practice = [_good_row(IsPractice=1, ArticleRevid=None, WikipediaUrl=None)
                    for _ in range(5)]
        df = pd.DataFrame(real + practice)
        validate_cleaned(df, original_row_count=26, removed_rows=1)

    def test_rejects_malformed_url(self):
        df = pd.DataFrame([_good_row(WikipediaUrl="http://wikipedia.com/Capybara")])
        with pytest.raises(ValidationError, match="URL"):
            validate_cleaned(df, original_row_count=2, removed_rows=1)

    def test_rejects_wrong_row_count(self):
        df = pd.DataFrame([_good_row(), _good_row()])
        with pytest.raises(ValidationError, match="row count"):
            validate_cleaned(df, original_row_count=10, removed_rows=3)

    def test_tolerates_practice_rows_with_string_is_practice(self):
        # IsPractice may come back from CSV as string "1" in some dtype-coercion paths.
        real = [_good_row() for _ in range(20)]
        practice = [_good_row(IsPractice="1", ArticleRevid=None, WikipediaUrl=None)
                    for _ in range(5)]
        df = pd.DataFrame(real + practice)
        validate_cleaned(df, original_row_count=26, removed_rows=1)

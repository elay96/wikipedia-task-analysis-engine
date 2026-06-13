"""S1: flatten the raw CSV into participants.csv, visits.csv, searches.csv."""
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

import parse

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw" / "wiki_behavior_107_data.csv"


def main() -> None:
    df = pd.read_csv(RAW)
    participants = parse.participants_table(df)
    visits = parse.flatten_visits(df)
    searches = parse.flatten_searches(df)

    participants.to_csv(DATA / "participants.csv", index=False)
    visits.to_csv(DATA / "visits.csv", index=False)
    searches.to_csv(DATA / "searches.csv", index=False)

    print(f"participants: {participants.shape}")
    print(f"visits:       {visits.shape}  (dwell present: {visits['dwell_ms'].notna().sum()})")
    print(f"searches:     {searches.shape}")
    print(f"unique articles: {visits['article'].nunique()}")


if __name__ == "__main__":
    main()

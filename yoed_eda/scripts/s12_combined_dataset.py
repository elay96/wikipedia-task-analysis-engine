"""S12: one combined per-participant dataset + a square Spearman correlation matrix.

Yoed asked for (1) a single file with every variable - cognitive AND computed -
for all participants, and (2) a square (variable x variable) correlation matrix
instead of the long-form pairwise list in spearman_correlations.csv."""
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTDIR = HERE.parent / "output"

# Free-text / id / timestamp columns: kept in the combined file, excluded from
# the numeric correlation matrix.
NON_NUMERIC = {
    "participant_id", "session_id", "started_at", "ended_at", "completion_code",
    "user_question", "interest_motivation", "perceived_learning", "new_questions",
    "discoveries", "ai_feedback",
}


def main() -> None:
    participants = pd.read_csv(DATA / "participants.csv")
    feats = pd.read_csv(OUTDIR / "participant_features.csv")
    combined = participants.merge(feats, on="participant_id", how="inner")
    combined.to_csv(OUTDIR / "combined_dataset.csv", index=False, encoding="utf-8-sig")

    num = combined.drop(columns=[c for c in NON_NUMERIC if c in combined.columns])
    num = num.apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().sum() >= 3]          # drop all/near-empty columns
    num = num.loc[:, num.nunique(dropna=True) > 1]    # drop constant columns
    corr = num.corr(method="spearman")
    corr.to_csv(OUTDIR / "correlation_matrix.csv", encoding="utf-8-sig")

    print(f"combined_dataset.csv: {combined.shape[0]} participants x "
          f"{combined.shape[1]} variables")
    print(f"correlation_matrix.csv: {corr.shape[0]} x {corr.shape[1]} (square)")


if __name__ == "__main__":
    main()

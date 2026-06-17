"""Creativity-profile predictive modeling helpers for the yoed_eda pipeline.

Pure functions: build theory-grouped creativity composites, run standardized
regression with bootstrap CIs + cross-validated R2 + permutation p, a
convergent-validity test, and a PCA robustness reduction. Orchestrated by s8."""
from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

COMPOSITES = {
    "dt_fluency": [
        "AUT Broom - Number of Answers", "AUT Belt - Number of Answers",
        "AQT Pencil - Number of Answers", "AQT Pillow - Number of Answers",
        "Verbal Fluency - Number of Answers",
    ],
    "dt_originality": [
        "AUT Belt - Originality", "AUT Broom - Originality",
        "AQT Pencil - Originality", "AQT Pillow - Originality",
        "AQT Complexity Score",
    ],
    "verbal_forward_flow": ["Verbal Fluency - Forward Flow"],
    "curiosity": ["Curiosity - Score"],
    "gf": ["GF - Score"],
}
PREDICTORS = list(COMPOSITES)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - s.mean()) / sd


def build_composites(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"participant_id": df["participant_id"].to_numpy()})
    for name, cols in COMPOSITES.items():
        z = pd.DataFrame({c: zscore(df[c]) for c in cols})
        out[name] = z.mean(axis=1, skipna=True)
    return out

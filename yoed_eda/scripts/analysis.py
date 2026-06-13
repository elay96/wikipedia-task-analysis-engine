"""Spearman correlation matrix (creativity x behavior) with BH-FDR."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import _bootstrap  # noqa: F401
from m83_utils import fdr_bh  # from repo scripts/

FDR_ALPHA = 0.05


def spearman_matrix(df: pd.DataFrame, measures: list, features: list) -> pd.DataFrame:
    rows = []
    for m in measures:
        for f in features:
            x = pd.to_numeric(df[m], errors="coerce")
            y = pd.to_numeric(df[f], errors="coerce")
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n < 3:
                rho, p = np.nan, np.nan
            else:
                res = sp_stats.spearmanr(x[mask], y[mask])
                rho, p = float(res.statistic), float(res.pvalue)
            rows.append({"measure": m, "feature": f, "n": n, "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    out["p_FDR"] = fdr_bh(out["p"].tolist())
    out["fdr_significant"] = (out["p_FDR"] < FDR_ALPHA).fillna(False)
    return out.sort_values("p").reset_index(drop=True)

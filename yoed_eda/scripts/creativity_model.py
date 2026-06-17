"""Creativity-profile predictive modeling helpers for the yoed_eda pipeline.

Pure functions: build theory-grouped creativity composites, run standardized
regression with bootstrap CIs + cross-validated R2 + permutation p, a
convergent-validity test, and a PCA robustness reduction. Orchestrated by s8."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5
Z_95 = 1.96


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


def _standardize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    sd = A.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (A - A.mean(axis=0)) / sd


def regress_with_ci(X, y, predictors, n_boot=2000, n_perm=5000, n_folds=5, seed=0) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]
    n = int(X.shape[0])
    Xs = _standardize(X)
    ys = _standardize(y.reshape(-1, 1)).ravel()

    fit = LinearRegression().fit(Xs, ys)
    betas = [float(b) for b in fit.coef_]
    r2_full = float(fit.score(Xs, ys))

    rng = np.random.RandomState(seed)
    boot = np.empty((n_boot, Xs.shape[1]))
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boot[b] = LinearRegression().fit(Xs[idx], ys[idx]).coef_
    lo = np.percentile(boot, CI_LOW_PCT, axis=0)
    hi = np.percentile(boot, CI_HIGH_PCT, axis=0)
    beta_ci = [[float(a), float(c)] for a, c in zip(lo, hi)]

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    r2_cv = float(cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean())

    count = sum(
        LinearRegression().fit(Xs, yp).score(Xs, yp) >= r2_full
        for yp in (rng.permutation(ys) for _ in range(n_perm))
    )
    p_perm = (1 + count) / (1 + n_perm)

    return {"predictors": list(predictors), "betas": betas, "beta_ci": beta_ci,
            "r2_full": r2_full, "r2_cv": r2_cv, "p_perm": p_perm, "n": n}


def convergent_validity(x, y) -> dict:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    xv, yv = x[mask].to_numpy(), y[mask].to_numpy()
    n = int(len(xv))
    sr = sp_stats.spearmanr(xv, yv)
    pr = sp_stats.pearsonr(xv, yv)
    r = float(pr.statistic)
    if n <= 3:
        ci = [float("nan"), float("nan")]
    else:
        z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
        ci = [float(np.tanh(z - Z_95 * se)), float(np.tanh(z + Z_95 * se))]
    return {"n": n, "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue),
            "pearson_r": r, "pearson_p": float(pr.pvalue), "pearson_ci": ci}

# Creativity → Browsing Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `s8` analysis stage (+ `s9` findings page) to `yoed_eda/` that tests whether trait creativity predicts browsing architecture, replicating the Zhou et al. (2024) dissociation between the Dancer (`forward_flow`) and Busybody-Hunter (`bh_score`) dimensions.

**Architecture:** Follow the existing `yoed_eda` split: pure, unit-tested logic in a module (`creativity_model.py`), orchestration in numbered scripts (`s8_*`, `s9_*`) that read CSVs and write `output/` artifacts. The new module builds 5 theory-grouped creativity composites, runs standardized multiple regression with bootstrap CIs + cross-validated R² + permutation p, a convergent-validity test, and a PCA robustness reduction.

**Tech Stack:** Python, pandas, numpy, scipy.stats, scikit-learn (`LinearRegression`, `PCA`, `KFold`, `cross_val_score`, `StandardScaler`), matplotlib, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-17-creativity-predicts-browsing-architecture-design.md`
- Use exact source column names verbatim (e.g. `"Verbal Fluency - Forward Flow"`, `"AQT Complexity Score"`). Do not rename or shorten.
- Outcomes are both continuous, run through the identical model: `forward_flow`, `bh_score`.
- 5 predictors (composites): `dt_fluency`, `dt_originality`, `verbal_forward_flow`, `curiosity`, `gf`.
- Pairwise-complete handling: drop rows with NaN in the outcome or any predictor, per outcome.
- File length ≤ 300 lines; function length ≤ 50 lines; named constants, no magic numbers.
- New modules import shared helpers via `import _bootstrap  # noqa: F401` (same as siblings).
- Tests colocated in `yoed_eda/scripts/test_*.py`; run the specific test file, not the whole suite.
- The `s9` findings page MUST follow the `html-findings-design` skill (RTL Hebrew, light mode, bottom-line-first, numbered card sections). HTML, never Markdown, for that deliverable.
- Conventional commits; work stays on branch `yoed-creativity-architecture` (already checked out); never commit to `main`.
- Commit message trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Run commands from the repo root: `C:\Users\elay9\wikipedia-task-analysis-engine`.

---

## File Structure

- Create `yoed_eda/scripts/creativity_model.py` — pure logic: composites, regression+CI, convergent validity, PCA reduction.
- Create `yoed_eda/scripts/test_creativity_model.py` — unit tests for the module.
- Create `yoed_eda/scripts/s8_creativity_predicts_architecture.py` — orchestration: reads data, runs models, writes JSON + figures.
- Create `yoed_eda/scripts/s9_build_creativity_html.py` — reads the JSON, renders the findings HTML.
- Modify `yoed_eda/README.md` — add `s8`/`s9` to the run order.
- Produced artifacts: `yoed_eda/output/creativity_architecture.json`, `output/figures/creativity_effects.png`, `output/figures/convergent_forward_flow.png`, `output/creativity_architecture_findings.html`.

---

## Task 1: Composite builder

**Files:**
- Create: `yoed_eda/scripts/creativity_model.py`
- Test: `yoed_eda/scripts/test_creativity_model.py`

**Interfaces:**
- Produces: `COMPOSITES: dict[str, list[str]]`, `PREDICTORS: list[str]` (= `list(COMPOSITES)`), `zscore(s: pd.Series) -> pd.Series`, `build_composites(df: pd.DataFrame) -> pd.DataFrame` (returns `participant_id` + one column per composite).

- [ ] **Step 1: Write the failing test**

```python
# test_creativity_model.py
import numpy as np
import pandas as pd
import creativity_model as cm


def test_zscore_centers_and_scales():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = cm.zscore(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


def test_zscore_constant_is_zero():
    z = cm.zscore(pd.Series([7.0, 7.0, 7.0]))
    assert (z == 0).all()


def test_build_composites_singleton_equals_zscore():
    df = pd.DataFrame({
        "participant_id": [1, 2, 3, 4],
        "Curiosity - Score": [1.0, 2.0, 3.0, 4.0],
        "GF - Score": [4.0, 3.0, 2.0, 1.0],
        "Verbal Fluency - Forward Flow": [0.0, 1.0, 2.0, 3.0],
        "AUT Broom - Number of Answers": [1.0, 1.0, 1.0, 1.0],
        "AUT Belt - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AQT Pencil - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AQT Pillow - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "Verbal Fluency - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AUT Belt - Originality": [1.0, 2.0, 3.0, 4.0],
        "AUT Broom - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Pencil - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Pillow - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Complexity Score": [1.0, 2.0, 3.0, 4.0],
    })
    out = cm.build_composites(df)
    assert list(out["participant_id"]) == [1, 2, 3, 4]
    assert set(cm.PREDICTORS) <= set(out.columns)
    expected = cm.zscore(df["Curiosity - Score"])
    assert np.allclose(out["curiosity"].to_numpy(), expected.to_numpy())


def test_build_composites_fluency_is_mean_of_zscores():
    df = pd.DataFrame({
        "participant_id": [1, 2, 3, 4],
        "Curiosity - Score": [1.0, 2.0, 3.0, 4.0],
        "GF - Score": [1.0, 2.0, 3.0, 4.0],
        "Verbal Fluency - Forward Flow": [1.0, 2.0, 3.0, 4.0],
        "AUT Broom - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AUT Belt - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AQT Pencil - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AQT Pillow - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "Verbal Fluency - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AUT Belt - Originality": [1.0, 2.0, 3.0, 4.0],
        "AUT Broom - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Pencil - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Pillow - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Complexity Score": [1.0, 2.0, 3.0, 4.0],
    })
    out = cm.build_composites(df)
    # all five fluency constituents identical after z-scoring -> mean equals that z-score
    expected = cm.zscore(df["AUT Belt - Number of Answers"])
    assert np.allclose(out["dt_fluency"].to_numpy(), expected.to_numpy())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest yoed_eda/scripts/test_creativity_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'creativity_model'` (or `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
# creativity_model.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest yoed_eda/scripts/test_creativity_model.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add yoed_eda/scripts/creativity_model.py yoed_eda/scripts/test_creativity_model.py
git commit -m "feat(yoed-eda): s8 creativity composites builder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Regression with bootstrap CIs, CV-R², permutation p

**Files:**
- Modify: `yoed_eda/scripts/creativity_model.py`
- Test: `yoed_eda/scripts/test_creativity_model.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `regress_with_ci(X, y, predictors, n_boot=2000, n_perm=5000, n_folds=5, seed=0) -> dict` with keys `predictors` (list[str]), `betas` (list[float], standardized), `beta_ci` (list[[lo, hi]]), `r2_full` (float), `r2_cv` (float), `p_perm` (float), `n` (int). Drops rows with NaN in `y` or any `X` column.

- [ ] **Step 1: Write the failing test**

```python
# append to test_creativity_model.py
def test_regress_recovers_strong_predictor():
    rng = np.random.RandomState(0)
    n = 200
    signal = rng.randn(n)
    X = np.column_stack([signal, rng.randn(n), rng.randn(n)])
    y = 2.0 * signal + 0.1 * rng.randn(n)
    res = cm.regress_with_ci(X, y, ["a", "b", "c"], n_boot=300, n_perm=300)
    assert res["n"] == 200
    assert res["beta_ci"][0][0] > 0  # signal predictor CI excludes zero
    assert res["r2_full"] > 0.9
    assert res["r2_cv"] > 0.8
    assert res["p_perm"] < 0.05


def test_regress_null_when_no_signal():
    rng = np.random.RandomState(1)
    X = rng.randn(120, 3)
    y = rng.randn(120)
    res = cm.regress_with_ci(X, y, ["a", "b", "c"], n_boot=200, n_perm=300)
    assert res["p_perm"] > 0.05


def test_regress_drops_nan_rows():
    rng = np.random.RandomState(2)
    X = rng.randn(50, 2)
    y = rng.randn(50)
    y[0] = np.nan
    X[1, 0] = np.nan
    res = cm.regress_with_ci(X, y, ["a", "b"], n_boot=50, n_perm=50)
    assert res["n"] == 48
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest yoed_eda/scripts/test_creativity_model.py -k regress -v`
Expected: FAIL with `AttributeError: module 'creativity_model' has no attribute 'regress_with_ci'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add imports at top of creativity_model.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# add functions
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
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest yoed_eda/scripts/test_creativity_model.py -k regress -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add yoed_eda/scripts/creativity_model.py yoed_eda/scripts/test_creativity_model.py
git commit -m "feat(yoed-eda): s8 regression with bootstrap CI, CV-R2, permutation p

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Convergent-validity test

**Files:**
- Modify: `yoed_eda/scripts/creativity_model.py`
- Test: `yoed_eda/scripts/test_creativity_model.py`

**Interfaces:**
- Produces: `convergent_validity(x, y) -> dict` with keys `n` (int), `spearman_rho`, `spearman_p`, `pearson_r`, `pearson_p` (floats), `pearson_ci` ([lo, hi], Fisher-z 95%). Pairwise-complete on `x`, `y`.

- [ ] **Step 1: Write the failing test**

```python
# append to test_creativity_model.py
def test_convergent_validity_strong_positive():
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = x + 0.3 * rng.randn(100)
    res = cm.convergent_validity(x, y)
    assert res["n"] == 100
    assert res["pearson_r"] > 0.8
    assert res["pearson_ci"][0] > 0
    assert res["spearman_rho"] > 0.7


def test_convergent_validity_pairwise_complete():
    x = [1.0, 2.0, 3.0, np.nan, 5.0]
    y = [1.0, 2.0, np.nan, 4.0, 5.0]
    res = cm.convergent_validity(x, y)
    assert res["n"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest yoed_eda/scripts/test_creativity_model.py -k convergent -v`
Expected: FAIL with `AttributeError: ... has no attribute 'convergent_validity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add import at top of creativity_model.py
from scipy import stats as sp_stats

# add function
def convergent_validity(x, y) -> dict:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    xv, yv = x[mask].to_numpy(), y[mask].to_numpy()
    n = int(len(xv))
    sr = sp_stats.spearmanr(xv, yv)
    pr = sp_stats.pearsonr(xv, yv)
    r = float(pr.statistic)
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    ci = [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
    return {"n": n, "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue),
            "pearson_r": r, "pearson_p": float(pr.pvalue), "pearson_ci": ci}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest yoed_eda/scripts/test_creativity_model.py -k convergent -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add yoed_eda/scripts/creativity_model.py yoed_eda/scripts/test_creativity_model.py
git commit -m "feat(yoed-eda): s8 convergent-validity test (verbal vs browsing forward flow)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: PCA robustness reduction

**Files:**
- Modify: `yoed_eda/scripts/creativity_model.py`
- Test: `yoed_eda/scripts/test_creativity_model.py`

**Interfaces:**
- Produces: `pca_reduce(df, measures, n_components=3) -> tuple[np.ndarray, list[float]]` returning the component scores (shape `(n, n_components)`) and the explained-variance-ratio list. Median-imputes then z-scores `measures` before PCA (`svd_solver="full"`, deterministic).

- [ ] **Step 1: Write the failing test**

```python
# append to test_creativity_model.py
def test_pca_reduce_shape_and_variance():
    rng = np.random.RandomState(0)
    latent = rng.randn(80, 1)
    cols = {f"m{i}": latent[:, 0] + 0.05 * rng.randn(80) for i in range(5)}
    df = pd.DataFrame(cols)
    comps, evr = cm.pca_reduce(df, list(cols), n_components=3)
    assert comps.shape == (80, 3)
    assert evr[0] > 0.9


def test_pca_reduce_imputes_nan():
    rng = np.random.RandomState(1)
    df = pd.DataFrame(rng.randn(40, 4), columns=list("abcd"))
    df.iloc[0, 0] = np.nan
    comps, evr = cm.pca_reduce(df, list("abcd"), n_components=2)
    assert comps.shape == (40, 2)
    assert not np.isnan(comps).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest yoed_eda/scripts/test_creativity_model.py -k pca -v`
Expected: FAIL with `AttributeError: ... has no attribute 'pca_reduce'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add import at top of creativity_model.py
from sklearn.decomposition import PCA

# add function
def pca_reduce(df, measures, n_components=3):
    X = df[measures].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    pca = PCA(n_components=n_components, svd_solver="full")
    comps = pca.fit_transform(Xs)
    return comps, [float(v) for v in pca.explained_variance_ratio_]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest yoed_eda/scripts/test_creativity_model.py -k pca -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full module test file**

Run: `python -m pytest yoed_eda/scripts/test_creativity_model.py -v`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add yoed_eda/scripts/creativity_model.py yoed_eda/scripts/test_creativity_model.py
git commit -m "feat(yoed-eda): s8 PCA robustness reduction

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: s8 orchestration script + figures

**Files:**
- Create: `yoed_eda/scripts/s8_creativity_predicts_architecture.py`
- Modify: `yoed_eda/README.md`

**Interfaces:**
- Consumes: `creativity_model` (`build_composites`, `PREDICTORS`, `regress_with_ci`, `convergent_validity`, `pca_reduce`); `MEASURES` from `s4_creativity_behavior`; `data/participants.csv`, `output/participant_features.csv`.
- Produces: `output/creativity_architecture.json` (keys `primary`, `convergent_validity`, `robustness_pca`, `pca_explained_variance`), `output/figures/creativity_effects.png`, `output/figures/convergent_forward_flow.png`.

This task is verified by running the script against real data (no unit test; matches the existing `s4`/`s6`/`s7` convention where `sN` scripts are run, and only the pure modules are unit-tested).

- [ ] **Step 1: Write the script**

```python
# s8_creativity_predicts_architecture.py
"""S8: does trait creativity predict browsing architecture?

Tests the dissociation (creativity -> forward_flow/Dancer, not bh_score/BH) and
the Verbal-Fluency-Forward-Flow -> browsing-forward_flow convergent validity,
with a PCA robustness check. See
docs/superpowers/specs/2026-06-17-creativity-predicts-browsing-architecture-design.md"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import creativity_model as cm
from s4_creativity_behavior import MEASURES

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTDIR = HERE.parent / "output"
FIGDIR = OUTDIR / "figures"
OUTCOMES = ["forward_flow", "bh_score"]
VFF_COL = "Verbal Fluency - Forward Flow"
OUTCOME_COLORS = {"forward_flow": "#4C78A8", "bh_score": "#E45756"}


def _plot_effects(primary: dict, path: Path) -> None:
    preds = primary[OUTCOMES[0]]["predictors"]
    y = np.arange(len(preds))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(preds) + 2))
    for offset, oc in [(-0.15, "forward_flow"), (0.15, "bh_score")]:
        r = primary[oc]
        betas = r["betas"]
        err = [[b - c[0] for b, c in zip(betas, r["beta_ci"])],
               [c[1] - b for b, c in zip(betas, r["beta_ci"])]]
        ax.errorbar(betas, y + offset, xerr=err, fmt="o", capsize=3,
                    color=OUTCOME_COLORS[oc], label=oc)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(preds)
    ax.set_xlabel("standardized beta (95% CI)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_convergent(df: pd.DataFrame, conv: dict, path: Path) -> None:
    x = pd.to_numeric(df[VFF_COL], errors="coerce")
    y = pd.to_numeric(df["forward_flow"], errors="coerce")
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(x, y, s=18, alpha=0.7)
    ax.set_xlabel("Verbal Fluency - Forward Flow (words)")
    ax.set_ylabel("forward_flow (browsing)")
    ax.set_title(f"r={conv['pearson_r']:.2f}, "
                 f"CI=[{conv['pearson_ci'][0]:.2f},{conv['pearson_ci'][1]:.2f}], "
                 f"n={conv['n']}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(DATA / "participants.csv")
    feats = pd.read_csv(OUTDIR / "participant_features.csv")
    df = participants.merge(feats, on="participant_id", how="inner")
    df = df.merge(cm.build_composites(participants), on="participant_id", how="inner")

    X = df[cm.PREDICTORS].to_numpy(dtype=float)
    primary = {oc: cm.regress_with_ci(X, df[oc].to_numpy(dtype=float), cm.PREDICTORS)
               for oc in OUTCOMES}
    conv = cm.convergent_validity(df[VFF_COL], df["forward_flow"])

    measures = [m for m in MEASURES if m in df.columns]
    pcs, evr = cm.pca_reduce(df, measures, n_components=3)
    pc_names = [f"PC{i + 1}" for i in range(pcs.shape[1])]
    robustness = {oc: cm.regress_with_ci(pcs, df[oc].to_numpy(dtype=float), pc_names)
                  for oc in OUTCOMES}

    results = {"primary": primary, "convergent_validity": conv,
               "robustness_pca": robustness, "pca_explained_variance": evr}
    (OUTDIR / "creativity_architecture.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_effects(primary, FIGDIR / "creativity_effects.png")
    _plot_convergent(df, conv, FIGDIR / "convergent_forward_flow.png")

    for oc in OUTCOMES:
        r = primary[oc]
        print(f"{oc}: R2_cv={r['r2_cv']:.3f} R2_full={r['r2_full']:.3f} "
              f"p_perm={r['p_perm']:.4f} n={r['n']}")
    print(f"convergent VFF->forward_flow: r={conv['pearson_r']:.3f} "
          f"CI={conv['pearson_ci']} n={conv['n']}")
    print(f"results -> {OUTDIR / 'creativity_architecture.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script against real data**

Run: `python yoed_eda/scripts/s8_creativity_predicts_architecture.py`
Expected: prints two outcome lines (e.g. `forward_flow: R2_cv=... p_perm=... n=100` and `bh_score: ... n=107`), one convergent line, and a `results -> .../creativity_architecture.json` line. No traceback.

- [ ] **Step 3: Verify the artifacts exist and parse**

Run: `python -c "import json; d=json.load(open('yoed_eda/output/creativity_architecture.json', encoding='utf-8')); print(list(d), '| primary outcomes:', list(d['primary']), '| conv n:', d['convergent_validity']['n'])"`
Expected: `['primary', 'convergent_validity', 'robustness_pca', 'pca_explained_variance'] | primary outcomes: ['forward_flow', 'bh_score'] | conv n: 100`

- [ ] **Step 4: Update README run order**

In `yoed_eda/README.md`, in the `## Run order` numbered list, after the `s5_build_html.py` line add:

```markdown
6. `python yoed_eda/scripts/s8_creativity_predicts_architecture.py`  # creativity -> architecture
7. `python yoed_eda/scripts/s9_build_creativity_html.py`             # creativity findings HTML
```

- [ ] **Step 5: Commit**

```bash
git add yoed_eda/scripts/s8_creativity_predicts_architecture.py yoed_eda/README.md yoed_eda/output/creativity_architecture.json yoed_eda/output/figures/creativity_effects.png yoed_eda/output/figures/convergent_forward_flow.png
git commit -m "feat(yoed-eda): s8 creativity->architecture analysis + figures

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: s9 findings HTML page

**Files:**
- Create: `yoed_eda/scripts/s9_build_creativity_html.py`

**Interfaces:**
- Consumes: `output/creativity_architecture.json`, `output/figures/creativity_effects.png`, `output/figures/convergent_forward_flow.png`.
- Produces: `output/creativity_architecture_findings.html`.

Before writing, invoke the `html-findings-design` skill and copy the complete `:root` + stylesheet block, the `<nav>`, `<header>`, and bottom-line markup from its `templates/base.html` (already present in this environment). Reuse the existing `yoed_eda/scripts/s5_build_html.py` as the structural pattern for generating a page from data via a Python f-string. The page is RTL Hebrew, light mode, bottom-line first.

- [ ] **Step 1: Write the script**

The script reads the JSON and emits the HTML. Section plan (numbered, one `.card` per unit):

- **0 שורה תחתונה** (`.bottom-line`): 3-4 conclusions filled from the JSON - whether creativity predicts `forward_flow` (Dancer) vs `bh_score` (BH), the convergent-validity result, and the honest framing. Build each `<li>` from the numbers (e.g. `forward_flow` `r2_cv`/`p_perm` vs `bh_score` `r2_cv`/`p_perm`; convergent `pearson_r` + CI).
- **1 הקונטקסט**: external validation of Zhou et al. (2024); trait creativity (this dataset) vs browsing-derived only (Zhou).
- **2 השערות**: H1 dissociation (creativity → Dancer, not BH), H2 convergent validity. State the descriptive-comparison choice and the "significant vs non-significant is not a difference test" caveat.
- **3 שיטה**: 5 composites (`cm.COMPOSITES`), standardized regression + bootstrap CIs + CV-R² + permutation p; PCA robustness.
- **4 ממצאים - דיסוציאציה**: embed `figures/creativity_effects.png`; a `.stat-row` per outcome with `R2_cv`, `R2_full`, `p_perm`, `n` from `primary`; per-predictor betas with CIs.
- **5 ממצאים - convergent validity**: embed `figures/convergent_forward_flow.png`; `.stat-row` with Pearson r + CI, Spearman rho + p, n.
- **6 בדיקת חוסן (PCA)**: `robustness_pca` R²/p per outcome + `pca_explained_variance`.
- **7 הסתייגויות** (`.callout warn`): N=107 power; `forward_flow` missing for 7; descriptive dissociation; "predict" = association not causation.
- **8 קישורים**: `s8`/`creativity_model.py` scripts, the JSON, the spec.

Use HTML entities for special chars (`&rho;`, `&alpha;`, `&middot;`). Images referenced relatively, e.g. `<img src="figures/creativity_effects.png" style="max-width:100%;">`.

```python
# s9_build_creativity_html.py
"""S9: render the creativity->architecture findings page (html-findings-design)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "output"
OUT = OUTDIR / "creativity_architecture_findings.html"

# Paste the full <style> block, <nav>, <header>, and .bottom-line scaffold from
# the html-findings-design templates/base.html. STYLE holds that CSS verbatim.
STYLE = """<PASTE the :root + full stylesheet from templates/base.html>"""


def _outcome_stat(label: str, r: dict) -> str:
    return (f'<div class="stat-row"><code>{label}</code>: '
            f"R&sup2;<sub>cv</sub>={r['r2_cv']:.3f}, "
            f"R&sup2;<sub>full</sub>={r['r2_full']:.3f}, "
            f"perm p={r['p_perm']:.4f}, n={r['n']}</div>")


def _beta_rows(r: dict) -> str:
    rows = []
    for name, b, ci in zip(r["predictors"], r["betas"], r["beta_ci"]):
        rows.append(f'<div class="stat-row"><code>{name}</code>: '
                    f"&beta;={b:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]</div>")
    return "\n".join(rows)


def main() -> None:
    res = json.loads((OUTDIR / "creativity_architecture.json").read_text(encoding="utf-8"))
    ff, bh = res["primary"]["forward_flow"], res["primary"]["bh_score"]
    conv = res["convergent_validity"]

    bottom = [
        f"<li><strong>יצירתיות מול Dancer:</strong> מודל היצירתיות מנבא את "
        f"<code>forward_flow</code> עם R&sup2;<sub>cv</sub>={ff['r2_cv']:.3f} "
        f"(perm p={ff['p_perm']:.3f}).</li>",
        f"<li><strong>יצירתיות מול Hunter-Busybody:</strong> אותו מודל על "
        f"<code>bh_score</code> נותן R&sup2;<sub>cv</sub>={bh['r2_cv']:.3f} "
        f"(perm p={bh['p_perm']:.3f}).</li>",
        f"<li><strong>Convergent validity:</strong> forward flow מילולי מנבא forward "
        f"flow בגלישה: r={conv['pearson_r']:.2f}, 95% CI "
        f"[{conv['pearson_ci'][0]:.2f}, {conv['pearson_ci'][1]:.2f}], n={conv['n']}.</li>",
        "<li><strong>הסתייגות:</strong> ההשוואה בין שתי המטרות תיאורית "
        "(רווחי סמך), לא מבחן הפרש פורמלי.</li>",
    ]

    rob = res["robustness_pca"]
    evr = ", ".join(f"{v:.2f}" for v in res["pca_explained_variance"])
    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>יצירתיות וארכיטקטורת גלישה &mdash; ממצאים</title>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;900&family=Rubik:wght@400;500;600;700&display=swap" rel="stylesheet">
{STYLE}
</head>
<body>
<nav><div class="nav-inner">
  <a href="#bottom-line">שורה תחתונה</a><span class="nav-sep">&middot;</span>
  <a href="#context">קונטקסט</a><span class="nav-sep">&middot;</span>
  <a href="#hyp">השערות</a><span class="nav-sep">&middot;</span>
  <a href="#method">שיטה</a><span class="nav-sep">&middot;</span>
  <a href="#dissociation">דיסוציאציה</a><span class="nav-sep">&middot;</span>
  <a href="#convergent">Convergent</a><span class="nav-sep">&middot;</span>
  <a href="#robustness">חוסן</a><span class="nav-sep">&middot;</span>
  <a href="#caveats">הסתייגויות</a><span class="nav-sep">&middot;</span>
  <a href="#links">קישורים</a>
</div></nav>
<div class="page-body"><div class="container">
<header>
  <h1>יצירתיות וארכיטקטורת גלישה</h1>
  <div class="subtitle">אימות חיצוני ל-Zhou et al. (2024): האם תכונת יצירתיות מנבאת את סגנון החיפוש בוויקיפדיה</div>
</header>

<section class="section" id="bottom-line">
  <h2 class="section-heading"><span class="section-num">0</span>שורה תחתונה</h2>
  <div class="bottom-line"><div class="card-title">מה אפשר להסיק</div>
  <ol>
    {"".join(bottom)}
  </ol></div>
</section>

<section class="section" id="context">
  <h2 class="section-heading"><span class="section-num">1</span>הקונטקסט</h2>
  <div class="card"><p>Zhou et al. (2024) גזרו את כל סגנונות הסקרנות מתוך הגלישה עצמה. במאגר הזה יש מבחני יצירתיות חיצוניים (AUT, AQT, שטף מילולי), כך שאפשר לבדוק לראשונה אם <strong>תכונת</strong> יצירתיות מנבאת את <strong>ארכיטקטורת</strong> הגלישה.</p></div>
</section>

<section class="section" id="hyp">
  <h2 class="section-heading"><span class="section-num">2</span>השערות</h2>
  <div class="card">
    <div class="card-title">H1 - דיסוציאציה</div>
    <p>יצירתיות מנבאת את <code>forward_flow</code> (Dancer) אבל לא את <code>bh_score</code> (Hunter-Busybody). מבוסס על כך שבמאמר &rho;(forward_flow, bh_score)=-0.05.</p>
    <div class="callout warn">ההשוואה תיאורית: מציגים רווחי סמך, לא מבחן הפרש. "מובהק מול לא-מובהק" אינו כשלעצמו מבחן הבדל.</div>
  </div>
  <div class="card">
    <div class="card-title">H2 - Convergent validity</div>
    <p><code>Verbal Fluency - Forward Flow</code> (על מילים) מנבא את <code>forward_flow</code> בגלישה (על עמודים): אותו construct, שני תחומים.</p>
  </div>
</section>

<section class="section" id="method">
  <h2 class="section-heading"><span class="section-num">3</span>שיטה</h2>
  <div class="card"><p>5 קומפוזיטים (z-score ואז ממוצע): חשיבה מתבדרת-כמות, חשיבה מתבדרת-איכות, Verbal Forward Flow, Curiosity, Gf (בקרה). רגרסיה מתוקננת לכל מטרה עם bootstrap CI, R&sup2; ב-cross-validation, ו-permutation ל-p. בדיקת חוסן: PCA על 13 המדדים הגולמיים.</p></div>
</section>

<section class="section" id="dissociation">
  <h2 class="section-heading"><span class="section-num">4</span>ממצאים - דיסוציאציה</h2>
  <div class="card">
    <img src="figures/creativity_effects.png" style="max-width:100%;height:auto;">
    {_outcome_stat("forward_flow (Dancer)", ff)}
    {_outcome_stat("bh_score (Hunter-Busybody)", bh)}
    <div class="sub-div"><strong>forward_flow betas:</strong>{_beta_rows(ff)}</div>
    <div class="sub-div"><strong>bh_score betas:</strong>{_beta_rows(bh)}</div>
  </div>
</section>

<section class="section" id="convergent">
  <h2 class="section-heading"><span class="section-num">5</span>ממצאים - Convergent validity</h2>
  <div class="card">
    <img src="figures/convergent_forward_flow.png" style="max-width:100%;height:auto;">
    <div class="stat-row"><code>Pearson</code>: r={conv['pearson_r']:.3f}, 95% CI [{conv['pearson_ci'][0]:.3f}, {conv['pearson_ci'][1]:.3f}], p={conv['pearson_p']:.4f}, n={conv['n']}</div>
    <div class="stat-row"><code>Spearman</code>: &rho;={conv['spearman_rho']:.3f}, p={conv['spearman_p']:.4f}</div>
  </div>
</section>

<section class="section" id="robustness">
  <h2 class="section-heading"><span class="section-num">6</span>בדיקת חוסן (PCA)</h2>
  <div class="card">
    <p>אותן רגרסיות עם רכיבי PCA במקום הקומפוזיטים. שונות מוסברת: {evr}.</p>
    {_outcome_stat("forward_flow (PCA)", rob["forward_flow"])}
    {_outcome_stat("bh_score (PCA)", rob["bh_score"])}
  </div>
</section>

<section class="section" id="caveats">
  <h2 class="section-heading"><span class="section-num">7</span>הסתייגויות</h2>
  <div class="callout warn"><ul style="margin-right:18px;">
    <li>N=107 מגביל כוח לאפקטים קטנים.</li>
    <li><code>forward_flow</code> חסר ל-7 משתתפים (פחות מ-2 עמודים נושאיים).</li>
    <li>הדיסוציאציה תיאורית (רווחי סמך), לא מבחן הפרש פורמלי.</li>
    <li>"מנבא" = אסוציאציה עם התכונה כמנבא, לא טענה סיבתית.</li>
  </ul></div>
</section>

<section class="section" id="links">
  <h2 class="section-heading"><span class="section-num">8</span>קישורים וקבצים</h2>
  <div class="card"><ul>
    <li>סקריפטים: <code>yoed_eda/scripts/s8_creativity_predicts_architecture.py</code>, <code>creativity_model.py</code></li>
    <li>תוצאות: <code>yoed_eda/output/creativity_architecture.json</code></li>
    <li>מפרט: <code>docs/superpowers/specs/2026-06-17-creativity-predicts-browsing-architecture-design.md</code></li>
  </ul></div>
</section>

</div></div>
</body>
</html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"findings -> {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Fill in STYLE**

Replace the `STYLE` placeholder string with the full `<style>...</style>` block (the `:root` variables through the `@media` rules) copied verbatim from the `html-findings-design` `templates/base.html`.

- [ ] **Step 3: Run the script**

Run: `python yoed_eda/scripts/s9_build_creativity_html.py`
Expected: prints `findings -> .../creativity_architecture_findings.html`, no traceback.

- [ ] **Step 4: Verify the page is well-formed and references figures**

Run: `python -c "t=open('yoed_eda/output/creativity_architecture_findings.html', encoding='utf-8').read(); assert 'bottom-line' in t and 'creativity_effects.png' in t and 'convergent_forward_flow.png' in t and t.strip().endswith('</html>'); print('ok', len(t), 'chars')"`
Expected: `ok <N> chars`

- [ ] **Step 5: Commit**

```bash
git add yoed_eda/scripts/s9_build_creativity_html.py yoed_eda/output/creativity_architecture_findings.html
git commit -m "feat(yoed-eda): s9 creativity->architecture findings HTML

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Motivation / external-validation framing → Tasks 5 (analysis), 6 (HTML context section). ✓
- H1 dissociation, descriptive with CIs → `regress_with_ci` (Task 2), side-by-side effect plot (Task 5), dissociation section (Task 6). ✓
- H2 convergent validity → `convergent_validity` (Task 3), scatter (Task 5), section 5 (Task 6). ✓
- 5 composites (exact membership) → `COMPOSITES` (Task 1). ✓
- Both outcomes, same model → `OUTCOMES` loop (Task 5). ✓
- CV-R² + permutation p → Task 2. ✓
- PCA robustness → `pca_reduce` (Task 4), robustness regressions + section 6 (Tasks 5, 6). ✓
- Outputs: `s8` script, JSON, figures, HTML per design system → Tasks 5, 6. ✓
- Caveats carried into writeup → Task 6 section 7. ✓
- Out-of-scope (no binary, no dancer-class, no formal difference test) → honored; nothing builds them. ✓

**Placeholder scan:** The only intentional fill-in is `STYLE` in Task 6, with an explicit copy-source (templates/base.html) and a dedicated step (6.2). All logic steps contain complete code.

**Type consistency:** `regress_with_ci` returns the same dict keys consumed by `_plot_effects`, `_outcome_stat`, `_beta_rows`. `convergent_validity` keys match `_plot_convergent` and section 5. `pca_reduce` returns `(array, list)` unpacked as `pcs, evr` in Task 5. `PREDICTORS`/`COMPOSITES` names consistent across tasks.

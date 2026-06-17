# Creativity → Browsing Architecture: External Validation of Zhou et al. (2024)

**Date:** 2026-06-17
**Scope:** New analysis stage (`s8`) in the `yoed_eda/` pipeline + findings HTML page.
**Status:** Design approved, pending spec review.

## 1. Motivation

The current `yoed_eda` analysis runs ~195 pairwise Spearman correlations between 13
creativity/cognition measures and 15 browsing features, with BH-FDR correction. The
result is an honest null (0/195 survive FDR), driven by `N=107` plus the
multiple-comparisons penalty.

This design replaces the correlation sweep with **one focused, directional question
that has a single primary outcome**, framed as an external replication of the central
claim in Zhou et al. (2024), "Architectural styles of curiosity in global Wikipedia
mobile app readership" (Sci. Adv.).

### The contribution

Zhou et al. derived *everything* from browsing behavior. Their "Dancer" creative style
was operationalized via **forward flow** computed on the page trajectory, because they
had no external creativity measures. Critically, they report that forward flow is
**nearly independent** of the Busybody-Hunter (BH) axis: `ρ(forward_flow, bh_score) =
-0.05`. So in their framework, creativity and the Hunter-Busybody dimension are
separate things.

This dataset has what Zhou lacked: **external creativity task scores** (AUT, AQT,
verbal fluency, etc.) measured independently of browsing. That lets us test, for the
first time, whether *trait* creativity predicts *behavioral* browsing architecture,
and whether it tracks the Dancer dimension (forward flow) rather than the
Hunter-Busybody dimension - an external validation of Zhou's dissociation.

## 2. Research questions and hypotheses

### Causal framing

Creativity is a stable trait measured before/independent of the browsing session;
browsing architecture is the behavioral expression. The causal direction is
**creativity → browsing**. (Correlation is symmetric; we report associations and use
"predict" in the trait-as-predictor sense, not as a causal claim.)

### H1 - Primary dissociation

Trait creativity predicts the **Dancer** dimension (`forward_flow`) but **not** the
**Busybody-Hunter** dimension (`bh_score`).

- Grounded in Zhou's own finding that the two browsing dimensions are nearly
  orthogonal (`ρ = -0.05`).
- Tested **descriptively**: the two regressions are presented side by side with
  **confidence intervals on the effect sizes**, so the reader can see whether the
  creativity→forward_flow association is larger than the creativity→bh_score
  association and whether their CIs overlap.
- Explicit caveat in the writeup: "significant vs non-significant" is not itself a
  difference test. We do not claim a formal interaction; we describe the pattern.

### H2 - Targeted convergent validity

A participant's **`Verbal Fluency - Forward Flow`** (forward flow over the *words* they
generated in the verbal fluency task, a trait) predicts their browsing **`forward_flow`**
(forward flow over the *sequence of Wikipedia pages* they visited, a behavior).

Same construct (semantic "drift" between consecutive items), two domains (words vs.
pages). This is the sharpest, most specific test of the Dancer idea: the person who
makes big semantic leaps with words also makes them while exploring Wikipedia.

### Secondary observation (not a primary hypothesis)

`Curiosity` and `Gf` are included as theoretically distinct neighbors. Curiosity is the
construct Zhou's whole framework rests on, so it is worth observing whether trait
curiosity tracks the Hunter-Busybody axis even when creativity does not. `Gf` (fluid
intelligence) is a control covariate.

## 3. Variables

### Outcomes (y) - both continuous, same model

| Outcome | Source column | Meaning |
|---|---|---|
| `forward_flow` | `participant_features.csv` | Dancer / creative-leaps dimension (browsing) |
| `bh_score` | `participant_features.csv` | Busybody-Hunter dimension (browsing) |

`forward_flow` is non-null for 100/107 participants (7 have < 2 topical pages);
`bh_score` is present for all 107. Analyses use pairwise-complete cases per outcome.

### Predictors (X) - 5 composites

Each constituent measure is z-scored across the cohort, then averaged within its
composite (mean of available z-scores, pairwise-complete).

| Composite | Constituent measures | Rationale |
|---|---|---|
| **Divergent thinking - fluency** | `AUT Broom - Number of Answers`, `AUT Belt - Number of Answers`, `AQT Pencil - Number of Answers`, `AQT Pillow - Number of Answers`, `Verbal Fluency - Number of Answers` | How many ideas (quantity) |
| **Divergent thinking - originality** | `AUT Belt - Originality`, `AUT Broom - Originality`, `AQT Pencil - Originality`, `AQT Pillow - Originality`, `AQT Complexity Score` | How novel / complex (quality) |
| **Verbal Forward Flow** (singleton) | `Verbal Fluency - Forward Flow` | Targeted H2 predictor; kept separate |
| **Curiosity** (singleton) | `Curiosity - Score` | The construct Zhou's framework rests on |
| **Gf** (control) | `GF - Score` | General fluid intelligence covariate |

This yields **5 predictors for N≈107**, an acceptable ratio (low overfitting risk).

> Note: the AQT and GF task semantics are inferred from column names. If the grouping
> mismatches the actual tasks, adjust the composite membership table; nothing else in
> the design changes.

## 4. Analysis

### Primary models

For each outcome `y ∈ {forward_flow, bh_score}`:

1. **Multiple linear regression** of `y` on the 5 composites.
2. Report **standardized betas with 95% CIs** for every predictor.
3. **Cross-validated R²** (k-fold, k=5 or 10) to report out-of-sample fit and avoid
   in-sample overfitting optimism.
4. **Permutation test** (shuffle `y`, e.g. 5000 perms) for the model-level p-value
   (CV-R² or full-model R²) and, optionally, per-predictor.

### Dissociation (H1) - descriptive

Present the two regressions side by side. Compare the **Verbal Forward Flow** and
divergent-thinking effect sizes (with CIs) across the two outcomes. Narrate the
pattern: creativity composites carry a larger, CI-separated effect on `forward_flow`
than on `bh_score`. No formal interaction/difference test is claimed.

### Convergent validity (H2) - focused

Direct association between `Verbal Fluency - Forward Flow` and browsing `forward_flow`:
Spearman + Pearson with CI, scatter plot. This is reported as a standalone result, not
buried in the regression table.

### Robustness check

Re-run the two primary regressions using **PCA components of the 13 raw measures**
(2-3 components) in place of the theory-grouped composites. If the dissociation holds
under both predictor parameterizations, the conclusion is robust to how the creativity
profile is constructed.

## 5. Outputs

- **New script:** `yoed_eda/scripts/s8_creativity_predicts_architecture.py`
  - Builds the 5 composites from `participants.csv`.
  - Runs the two primary regressions (betas + CIs + CV-R² + permutation p).
  - Runs the H2 convergent-validity test.
  - Runs the PCA robustness check.
  - Writes a results artifact (e.g. `output/creativity_architecture.json`) and figures
    (side-by-side effect-size plot with CIs; H2 scatter).
- **Findings page:** `yoed_eda/output/creativity_architecture_findings.html`, following
  the `html-findings-design` skill (RTL Hebrew, light mode, bottom-line first, numbered
  card sections).
- Reuse existing utilities where possible (`analysis.py` for correlation helpers,
  `cca.py` patterns for permutation, `m83_utils` already feeds the features).

## 6. Caveats (carry into the writeup)

- `N=107` limits power for small effects, even with a single focused question.
- `forward_flow` is missing for 7 participants (too few topical pages).
- Browsing `dwell` is derived from consecutive `start_time`, not measured directly
  (does not affect `forward_flow`/`bh_score` but noted for completeness).
- The dissociation is **descriptive**: we show CIs, not a formal difference test, so we
  describe rather than statistically assert that the two associations differ.
- "Predict" denotes trait-as-predictor association, not causal inference.

## 7. Out of scope (YAGNI)

- No binary Hunter/Busybody classification (the framework and this design treat BH as
  continuous).
- No third "Dancer class" labels; the Dancer is captured continuously by `forward_flow`.
- No formal interaction/Steiger difference test for the dissociation (descriptive by
  decision).
- No changes to existing `s1`-`s7` outputs.

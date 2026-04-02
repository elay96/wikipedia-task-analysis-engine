# Methodology Review & Recommendation — Design Spec

> **Date:** 2026-04-02
> **Branch:** feat/professors-analysis
> **Goal:** Build an HTML page (Hebrew, RTL) that explains the methodological debate, analyzes approaches for creating a unified explore/exploit metric, and includes a draft email to advisors.

---

## Context

### Research Question
Does a pre-Wikipedia-task experimental manipulation affect explore/exploit behavior during Wikipedia search?

### Design
Between-subjects (future experiment). Currently in pilot phase — analyzing Wikipedia task data only to define the analysis plan for pre-registration.

### Advisors
1. **Kineret** (primary) — wants a clear unified metric. Suggested adding combined metrics to PCA (problematic: circularity).
2. **Yuval Hart** (Hebrew University) — creative foraging paradigm (Hart 2017). Temporal dynamics: exploration/exploitation phases, switching, path lengths.
3. **Yoad Kenett** (Technion) — semantic networks (Kenett 2014). Objects to arbitrary metric combination. Sent Zhou et al. 2024 paper supporting multidimensional approach.

### Key Finding from Pilot (M29)
PCA on 3 raw continuous signals:
- PC1 = 55.7% — engagement (time on page + writing amount)
- PC2 = 33.3% — semantic breadth (topic distance)
- PC3 = 11.0% — residual

**Conclusion:** Explore/exploit is NOT unidimensional. Two partially independent dimensions.

---

## Deliverable

A single HTML file (`docs/methodology_review.html`) in Hebrew (RTL), with one section containing a draft email also in Hebrew. Self-contained (inline CSS, no external dependencies).

### Sections

#### 1. Introduction
- What the project studies (explore/exploit in Wikipedia browsing)
- The experimental setup (manipulation → Wikipedia task → measure behavior)
- Why the methodology matters for pre-registration

#### 2. Metrics Map
Table of all relevant metrics (M1–M32):
- What each measures
- Type (binary/continuous)
- Axis (time / semantics / writing / combined / PCA)
- Status (exists / new)

Focus on the key metrics: M2 (60s threshold), M4 (type/paste), M15 (median time), M16 (LSA), M18 (binary writing), M20 (LSA + type/paste), M26 (topic modeling), M27 (LSA + time median), M28 (LSA + 60s), M29 (PCA raw).

#### 3. The Problem — Three Methodological Issues
1. **Arbitrary combination (M27, M28):** Logical AND between two metrics. No justification for equal weight, threshold is arbitrary, information loss from continuous→binary→combine.
2. **PCA on binary classifications:** Circularity (combined metrics built from base metrics), PCA assumes continuous/normal variables, doesn't address construct validity.
3. **PCA on raw signals (M29):** Legitimate, but reveals 2 dimensions (55.7% + 33.3%), not 1. The semantic axis is partially independent from engagement. **This is a finding, not a failure.**

#### 4. Approach Comparison Table
| Approach | Pros | Cons | When appropriate |
|---|---|---|---|
| Simple average | Simple | Arbitrary, assumes equal weight | Theoretical justification for equal weight |
| Weighted average | Flexible | Where do weights come from? | Prior knowledge of relative importance |
| PC1 only | Data-driven, single score | Ignores 33% variance | PC1 >70% |
| PCA — 2 DVs | Preserves all info, data-driven | Requires MANOVA, larger N | 2+ significant PCs ← **our case** |
| Factor Analysis | Explicit latent variable model | Needs large N, strong assumptions | 100+ observations |
| Single representative metric | Simple, transparent | Ignores other axes | One axis theoretically dominant |

#### 5. Decision Framework
Flowchart:
```
PC1 explains >70%?
  ├── Yes → PC1 alone as DV (unified metric is legitimate)
  └── No → PC1 + PC2 significant?
        ├── Yes → MANOVA with 2 DVs ← YOU ARE HERE
        └── No → Consider FA or single representative metric
```

Additional decision criteria:
- Correlation between metrics: high r + theoretical basis → combination may work
- Latent variable: run CFA to test if a single factor fits
- Practical: if N is too small for MANOVA, PC1 alone is defensible as pre-registered secondary analysis

#### 6. Recommendation
**Approach C — Two PCA scores as DVs:**

1. Extract 3 raw continuous signals per page visit (as in M29):
   - Time on page (seconds)
   - Topic distance from previous page (JSD)
   - Writing amount (seconds of typing + weighted paste events)
2. Run PCA on standardized signals
3. Compute mean PC1 and PC2 scores **per participant**
4. MANOVA: condition (manipulation) as IV, PC1 + PC2 as DVs
5. Follow-up: separate t-tests on each PC to identify which dimension is affected

**Why this resolves the debate:**
- **Kineret:** Gets a clear analysis plan with interpretable results
- **Yoad:** No arbitrary combination — PCA defines weights from data. Consistent with Zhou et al. 2024 multidimensional approach
- **Yuval:** Engagement axis (PC1) maps to his explore/exploit temporal dynamics; semantic axis (PC2) adds the content dimension

#### 7. Draft Email to Advisors
In Hebrew. Professional, diplomatic, aimed at building consensus.

Content:
- Original thinking (unified metric from combined measures)
- What was built (M27, M28 combined metrics + M29 PCA)
- The problem with current approach (arbitrary AND, no data-driven justification)
- Yoad's PCA suggestion and why it partially works
- The key finding: 2 dimensions, not 1 (with numbers: 55.7% + 33.3%)
- Methodological recommendation: MANOVA with PC1 + PC2
- How this aligns with Zhou et al. 2024 and the creative foraging literature
- How this addresses each advisor's perspective (Kineret, Yuval, Yoad — by name)
- Proposed next steps for pre-registration

---

## Style

- Clean, minimal design
- Light background, readable typography
- RTL throughout (Hebrew)
- Navigation sidebar or top nav between sections
- Print-friendly
- Self-contained (no external CSS/JS dependencies)
- Color-coded tables (green = recommended, yellow = conditional, red = problematic)

---

## File Location

```
docs/methodology_review.html    # The deliverable
```

---

## Out of Scope

- Running new analyses or generating new plots
- Modifying existing Python scripts
- Implementing the MANOVA analysis
- English translation of the full document

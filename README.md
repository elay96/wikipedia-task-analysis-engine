# 🌐 Wikipedia Task Analysis Engine

> 🔬 A Python-based analysis pipeline for studying **explore/exploit strategies** in Wikipedia-based information foraging tasks.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-orange)]()

---

## 📖 Overview

This project analyzes how participants **explore and exploit** information while browsing Wikipedia articles to answer creative and factual questions. The engine processes behavioral data collected via Prolific and produces a comprehensive suite of **33 behavioral measures** — from basic timing and navigation metrics to advanced semantic analysis, PCA-based classification, and phase-switching dynamics.

### 🎯 Research Questions

- How do individuals **balance exploration vs. exploitation** during information search?
- Does a **pre-task experimental manipulation** affect foraging strategies?
- Is explore/exploit **unidimensional or multidimensional**? (Pilot finding: 2 dimensions — engagement + semantic breadth)

---

## 📚 Documentation

Interactive HTML guides — available online via GitHub Pages:

| Document | Description |
|----------|-------------|
| 🔬 [Methodology Review](https://elay96.github.io/wikipedia-task-analysis-engine/docs/methodology_review.html) | Methodological debate, approach comparison, recommendation (MANOVA with 2 PCA DVs), and draft email to advisors |
| 📋 [Professors Plan](https://elay96.github.io/wikipedia-task-analysis-engine/docs/professors_plan.html) | Analysis plan from advisors meeting — 3 axes, combined measures, PCA on raw signals |
| 📊 [PCA Charts Guide](https://elay96.github.io/wikipedia-task-analysis-engine/docs/pca_charts_guide.html) | How to read PCA charts M29–M32: scree plot, biplot, participant map, composite |
| 📏 [Measures Guide](https://elay96.github.io/wikipedia-task-analysis-engine/docs/measures_guide.html) | Overview of measures M1–M9 |
| 🔀 [Switch Rate Analysis](https://elay96.github.io/wikipedia-task-analysis-engine/docs/switch_rate_analysis.html) | Switch rate features (M34–M41): 3 signals, PCA attempts, composite score |

---

## 🏗️ Project Structure

```
wikipedia-task-analysis-engine/
│
├── 📂 data/                    # Raw experimental data
│   ├── Game.csv                # Primary dataset — actions, timestamps, metadata
│   ├── similarity_matrix.json  # Article-to-article cosine similarity
│   ├── wiki_texts.json         # Full article texts
│   └── topic_model.json        # LDA topic model output
│
├── 📂 scripts/                 # Analysis pipeline (M1–M33)
│   ├── helpers.py              # Shared utilities — data loading, plot helpers
│   ├── compute_similarity.py   # Pre-compute similarity matrix
│   ├── compute_topics.py       # Train LDA topic model
│   │
│   │  # ⏱️ Basic measures
│   ├── m1_time_20s.py          # Time-based filtering (20s threshold)
│   ├── m2_time_60s.py          # Time-based filtering (60s threshold)
│   ├── m3_typing.py            # Typing behavior analysis
│   ├── m4_typing_pasting.py    # Typing vs. pasting detection
│   ├── m5_navigation.py        # Navigation patterns
│   ├── m6_pages.py             # Page visit counts
│   ├── m7_heatmap.py           # Activity heatmaps
│   ├── m8_semantic.py          # Semantic similarity analysis
│   ├── m10_time_per_page.py    # Per-page time distribution
│   │
│   │  # 🕸️ Network & PCA
│   ├── m11_network.py          # Network graph analysis
│   ├── m11_pca_distance.py     # PCA-based content distance
│   ├── m11_panel_b.py          # Network panel B
│   ├── m12_pca7_distance.py    # 7-component PCA distance
│   │
│   │  # 🏷️ Classification
│   ├── m13_combined_binary.py          # Binary explore/exploit classification
│   ├── m14_combined_binary_71d.py      # Classification variant (71d)
│   ├── m15_combined_binary_median.py   # Median-based classification
│   ├── m15_diagnostics_plot.py         # Diagnostics visualization
│   ├── m16_combined_lsa_median.py      # LSA + median classification
│   ├── m18_typing_binary.py            # Binary typing classification
│   │
│   │  # 👤 Individual differences & phases
│   ├── m19_practice_duration.py        # Practice phase duration
│   ├── m20_cross_subject_median.py     # Cross-subject median analysis
│   ├── m21_tendency_dist.py            # Explore/exploit tendency distribution
│   ├── m22_tendency_analysis.py        # Individual tendency analysis
│   ├── m23_practice_vs_exploit.py      # Practice vs. exploitation
│   ├── m24_phase_duration.py           # Phase duration & switch rate
│   ├── m25_phase_duration_variants.py  # Switch rate variants
│   │
│   │  # 🧠 Topic modeling & combined measures
│   ├── m26_topic_modeling.py           # LDA topic distance visualization
│   ├── m27_lsa_time_median.py          # LSA + time-median combined
│   ├── m28_lsa_60s.py                  # LSA + 60s combined
│   │
│   │  # 📐 PCA on raw signals
│   ├── m29_pca_raw.py                  # PCA on 3 raw continuous signals
│   ├── m30_pca_biplot.py               # Enhanced PCA biplot by domain
│   ├── m31_participant_map.py          # Individual strategy map
│   ├── m32_pca_composite.py            # Composite PCA summary
│   ├── m33_analysis_pipeline.excalidraw # 🔄 Analysis pipeline diagram
│   │
│   │  # 🔀 Switch rate features & composite score
│   ├── compute_bertopic.py             # BERTopic topic model
│   ├── m34_switch_time.py              # Switch rate — time-based (60s threshold)
│   ├── m35_switch_topic.py             # Switch rate — topic-based (BERTopic + LDA)
│   ├── m36_switch_typing.py            # Switch rate — typing/paste binary
│   ├── m37_pca_switch.py               # PCA per domain on 3 switch rates
│   ├── m38_pca_avg_first.py            # PCA — average first, then PCA
│   ├── m39_pca_pool_first.py           # PCA — pool first, then average
│   ├── m40_pca_zscore.py               # PCA — z-scored
│   └── m41_composite_avg.py            # Final composite: mean of 3 switch rates
│
├── 📂 output/                  # 🖼️ Generated visualizations (.png)
├── 📂 docs/                    # 📄 HTML documentation & guides
└── 📂 references/              # 📎 Research papers
```

---

## 🧪 Methodology

### 🧩 Experiment Design

| Component | Details |
|-----------|---------|
| 🧑‍💻 **Participants** | Recruited via [Prolific](https://prolific.co) |
| 📝 **Task** | Answer questions by browsing Wikipedia articles |
| 🔀 **Design** | Between-subjects (manipulation before Wikipedia task) |
| 📊 **Data Collected** | Article opens, task start/end, paste events, answer snapshots, timestamps |

### 🔍 Explore/Exploit Measurement — PCA on Raw Signals

The recommended approach (see [Methodology Review](https://elay96.github.io/wikipedia-task-analysis-engine/docs/methodology_review.html)) uses PCA on 3 raw continuous signals per page visit:

| Signal | What it measures | Type |
|--------|-----------------|------|
| ⏱️ **Page duration** | Seconds spent on page | Continuous |
| 🧠 **Semantic distance** | Topic shift between pages (LDA topic model → JSD) | Continuous |
| ✍️ **Writing activity** | Typing duration + weighted paste events | Continuous |

**📊 Pilot results (M29):**
- PC1 = 55.7% — **Engagement** (time + writing)
- PC2 = 33.3% — **Semantic breadth** (topic distance)
- Together: 89% of variance explained

**📏 Pre-specified decision rule:**
- If PC1 > 70% → use PC1 alone as DV (t-test)
- If PC1 + PC2 both significant → use both as DVs (MANOVA)

### 📈 Key Measures

| Measure | Category | Description |
|---------|----------|-------------|
| M1–M2 | ⏱️ Timing | Time filtering at 20s and 60s thresholds |
| M3–M4 | ⌨️ Typing | Keystroke patterns, typing vs. pasting |
| M5–M6 | 🧭 Navigation | Page visit patterns and counts |
| M7 | 🔥 Heatmap | Temporal activity visualization |
| M8 | 🧠 Semantic | Article similarity analysis |
| M10 | ⏱️ Per-page | Time distribution per article |
| M11–M12 | 🕸️ Network/PCA | Network graphs, PCA-based distance |
| M13–M16 | 🏷️ Classification | Binary explore/exploit labeling |
| M18–M19 | ⌨️ Typing | Binary typing behavior, practice duration |
| M20–M22 | 👤 Individual | Cross-subject medians, tendency analysis |
| M23 | 🔄 Comparison | Practice vs. exploitation patterns |
| M24–M25 | 🔀 Switching | Phase run-lengths & switch rate variants |
| M26 | 🧠 Topic Modeling | LDA topic distance visualization |
| M27–M28 | 🏷️ Combined | LSA + time/60s combined measures |
| M29 | 📐 PCA Raw | PCA on 3 raw continuous signals |
| M30–M32 | 📊 PCA Viz | Biplot, participant map, composite |
| M33 | 🔄 Pipeline | Analysis pipeline & decision framework diagram |
| M34–M36 | 🔀 Switch Rate | 3 switch rate signals: time (60s), topic (LDA), typing |
| M37–M40 | 📐 PCA Switch | PCA variants on switch rates (per-domain, pooled, z-scored) |
| M41 | 🎯 Composite | Final composite switching score = mean of 3 switch rates |

---

## 🚀 Getting Started

### 📋 Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

### ▶️ Running an Analysis

Each measure script is standalone and can be run independently:

```bash
# Run a single measure
python scripts/m24_phase_duration.py

# Run PCA on raw signals
python scripts/m29_pca_raw.py

# Output visualizations are saved to output/
```

### 📂 Data Requirements

Place your data files in the `data/` directory:
- `Game.csv` — Primary behavioral data export
- `similarity_matrix.json` — Pre-computed article similarity matrix
- `wiki_texts.json` — Article text corpus

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.10+ | Core language |
| 🐼 Pandas | Data manipulation |
| 🔢 NumPy | Numerical computation |
| 📊 Matplotlib | Visualization |
| 🧪 scikit-learn | PCA, LDA topic modeling, similarity metrics |
| 📐 SciPy | Statistical tests (Wilcoxon, Cohen's d) |

---

## 📎 References

- Hart, Y. et al. (2017). *Creative foraging: An experimental paradigm for studying exploration and discovery.* PLOS ONE.
- Zhou, D. et al. (2024). *Architectural styles of curiosity in global Wikipedia mobile app readership.* Science Advances.
- Kenett, Y. N. et al. (2014). *Investigating the structure of semantic networks in low and high creative persons.* Frontiers in Human Neuroscience.

---

## 📄 License

This project is part of academic research. See `references/` for related publications.

---

<p align="center">
  Made with 🧠 + ☕ for behavioral research
</p>

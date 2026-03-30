# 🌐 Wikipedia Task Analysis Engine

> 🔬 A Python-based analysis pipeline for studying **explore/exploit strategies** in Wikipedia-based information foraging tasks.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-orange)]()

---

## 📖 Overview

This project analyzes how participants **explore and exploit** information while browsing Wikipedia articles to answer creative and factual questions. The engine processes behavioral data collected via Prolific and produces a comprehensive suite of **25+ behavioral measures** — from basic timing and navigation metrics to advanced semantic analysis, PCA-based classification, and phase-switching dynamics.

### 🎯 Research Questions

- How do individuals **balance exploration vs. exploitation** during information search?
- Do **high-creativity** vs. **low-creativity** prompts lead to different foraging strategies?
- Can we classify moment-to-moment behavior into **explore/exploit phases** and measure switching patterns?

---

## 🏗️ Project Structure

```
wikipedia-task-analysis-engine/
│
├── 📂 data/                    # Raw experimental data
│   ├── Game.csv                # Primary dataset — actions, timestamps, metadata
│   ├── similarity_matrix.json  # Article-to-article cosine similarity
│   └── wiki_texts.json         # Full article texts
│
├── 📂 scripts/                 # Analysis pipeline (m1–m25)
│   ├── helpers.py              # Shared utilities — data loading, plot helpers
│   ├── m1_time_20s.py          # ⏱️ Time-based filtering (20s threshold)
│   ├── m2_time_60s.py          # ⏱️ Time-based filtering (60s threshold)
│   ├── m3_typing.py            # ⌨️ Typing behavior analysis
│   ├── m4_typing_pasting.py    # 📋 Typing vs. pasting detection
│   ├── m5_navigation.py        # 🧭 Navigation patterns
│   ├── m6_pages.py             # 📄 Page visit counts
│   ├── m7_heatmap.py           # 🔥 Activity heatmaps
│   ├── m8_semantic.py          # 🧠 Semantic similarity analysis
│   ├── m10_time_per_page.py    # ⏱️ Per-page time distribution
│   ├── m11_network.py          # 🕸️ Network graph analysis
│   ├── m11_pca_distance.py     # 📐 PCA-based content distance
│   ├── m12_pca7_distance.py    # 📐 7-component PCA distance
│   ├── m13_combined_binary.py  # 🏷️ Binary explore/exploit classification
│   ├── m14_combined_binary_71d.py  # 🏷️ Classification variant (71d)
│   ├── m15_combined_binary_median.py # 📊 Median-based classification
│   ├── m16_combined_lsa_median.py    # 📊 LSA + median classification
│   ├── m18_typing_binary.py    # ⌨️ Binary typing classification
│   ├── m19_practice_duration.py # 🎯 Practice phase duration
│   ├── m20_cross_subject_median.py  # 👥 Cross-subject median analysis
│   ├── m21_tendency_dist.py    # 📈 Explore/exploit tendency distribution
│   ├── m22_tendency_analysis.py # 📈 Individual tendency analysis
│   ├── m23_practice_vs_exploit.py   # 🔄 Practice vs. exploitation
│   ├── m24_phase_duration.py   # 🔀 Phase duration & switch rate
│   └── m25_phase_duration_variants.py # 🔀 Switch rate variants
│
├── 📂 output/                  # Generated visualizations (.png)
├── 📂 docs/                    # Documentation & measure guides
├── 📂 references/              # Research papers & IRB documents
└── README.md
```

---

## 🧪 Methodology

### 🧩 Experiment Design

| Component | Details |
|-----------|---------|
| 🧑‍💻 **Participants** | Recruited via [Prolific](https://prolific.co) |
| 📝 **Task** | Answer questions by browsing Wikipedia articles |
| 🎨 **Conditions** | High-creativity vs. Low-creativity prompts |
| 📊 **Data Collected** | Article opens, task start/end, paste events, answer snapshots, timestamps |

### 🔍 Explore/Exploit Classification

The engine classifies each page visit as **explore** 🔭 or **exploit** ⛏️ using a multi-signal approach:

1. **🧠 Semantic Distance** — Cosine similarity between consecutive articles (via PCA on similarity matrix, 7 components)
2. **⏱️ Time on Page** — Duration relative to session mean
3. **📊 LSA Signals** — Latent semantic analysis for content-based distance
4. **📐 Combined Threshold** — Session-mean thresholds for both time and distance signals

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
| M20–M22 | 👥 Individual | Cross-subject medians, tendency analysis |
| M23 | 🔄 Comparison | Practice vs. exploitation patterns |
| M24–M25 | 🔀 Switching | Phase run-lengths & switch rate variants |

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

# Output visualizations are saved to output/
```

### 📂 Data Requirements

Place your data files in the `data/` directory:
- `Game.csv` — Primary behavioral data export
- `similarity_matrix.json` — Pre-computed article similarity matrix
- `wiki_texts.json` — Article text corpus

---

## 📊 Sample Output

All measure scripts generate publication-ready visualizations saved to `output/`:

| Visualization | Script |
|---------------|--------|
| 🔥 Activity heatmaps | `m7_heatmap.py` |
| 🕸️ Navigation networks | `m11_network.py` |
| 🏷️ Explore/exploit timelines | `m13–m16` |
| 📈 Tendency distributions | `m21_tendency_dist.py` |
| 🔀 Phase switching patterns | `m24_phase_duration.py` |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.10+ | Core language |
| 🐼 Pandas | Data manipulation |
| 🔢 NumPy | Numerical computation |
| 📊 Matplotlib | Visualization |
| 🧪 scikit-learn | PCA, similarity metrics |
| 📐 SciPy | Statistical tests (Wilcoxon, Cohen's d) |

---

## 📄 License

This project is part of academic research. See `references/` for related publications and IRB documentation.

---

<p align="center">
  Made with 🧠 + ☕ for behavioral research
</p>

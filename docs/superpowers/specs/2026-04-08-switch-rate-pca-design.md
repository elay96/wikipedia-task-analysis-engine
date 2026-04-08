# Switch Rate Features + PCA Design Spec

## Overview

Build 3 switch rate metrics per participant per domain, then run PCA to produce a composite explore/exploit score. Each metric captures a different facet of switching behavior between explore and exploit states.

## Decisions

- **Topic model:** Both BERTopic (new) and LDA (existing) - compare results
- **Topic switch definition:** Change in dominant topic (argmax) between consecutive pages
- **PCA scope:** Separate PCA per domain (4 PCAs)
- **PCA standardization:** None - all switch rates are on the same 0-1 scale
- **Switch rate formula:** Number of transitions between states / (N-1), where N = number of pages

## Scripts

### compute_bertopic.py (prerequisite)

- Loads `wiki_texts.json`
- Trains BERTopic on all articles
- Saves `data/bertopic_model.json` mapping article_title -> topic_id

### M34: m34_switch_time.py

**Input:** Game.csv
**Logic:**
1. Per participant + domain: get page sequence (article_open events)
2. Compute dwell time per page (time until next open or task_end)
3. Binary: >60s = Exploit, else = Explore
4. switch_rate = transitions / (N-1)

**Visualization (per domain):**
- Grid of subplots per participant (M10 style, dark theme)
- Each subplot: X = page number, colored bars (green=Explore, blue=Exploit)
- Bar height = dwell time in seconds
- Dashed horizontal line at 60s threshold
- Subplot title: `User {id} - SR: {switch_rate:.2f}`

**Output:**
- `output/m34_switch_time_{domain}.png` (4 images)
- `output/m34_switch_time.csv` (columns: participant_id, domain, switch_rate)

### M35: m35_switch_topic.py

**Input:** Game.csv, bertopic_model.json, topic_model.json
**Logic:**
1. Per participant + domain: get page sequence
2. Per page: lookup topic_id from topic model
3. Transition = topic_id differs between consecutive pages
4. switch_rate = transitions / (N-1)

**Visualization (per domain, per model):**
- Grid of subplots per participant (M10 style)
- Each subplot: X = page number, colored bars per topic_id (fixed palette ~10 colors)
- Uniform bar height (categorical, no continuous value)
- Dashed vertical separators at topic transitions
- Subplot title: `User {id} - SR: {switch_rate:.2f}`

**Output:**
- `output/m35_switch_bertopic_{domain}.png` (4 images)
- `output/m35_switch_lda_{domain}.png` (4 images)
- `output/m35_switch_bertopic.csv` (columns: participant_id, domain, switch_rate)
- `output/m35_switch_lda.csv` (columns: participant_id, domain, switch_rate)

### M36: m36_switch_typing.py

**Input:** Game.csv
**Logic:**
1. Per participant + domain: get page sequence
2. Per page: check for typing bursts (helpers.py `_detect_typing()`) or paste events (M18 logic)
3. Binary: typing/paste = Exploit, else = Explore
4. switch_rate = transitions / (N-1)

**Visualization (per domain):**
- Grid of subplots per participant (M10 style)
- Each subplot: X = page number, colored bars (green=Explore, blue=Exploit)
- Uniform bar height (binary)
- Subplot title: `User {id} - SR: {switch_rate:.2f}`

**Output:**
- `output/m36_switch_typing_{domain}.png` (4 images)
- `output/m36_switch_typing.csv` (columns: participant_id, domain, switch_rate)

### M37: m37_pca_switch.py

**Input:** CSV files from M34, M35, M36 (m34_switch_time.csv, m35_switch_bertopic.csv / m35_switch_lda.csv, m36_switch_typing.csv)
**Logic:**
1. Per domain: build table (rows=participants, cols=3 switch rates)
2. PCA on raw values (no standardization)
3. Store: loadings, explained variance, scores per participant
4. Run twice: once with BERTopic topic switch rates, once with LDA

**Visualization (per domain, per topic model):**
3 panels per image:
1. Scree plot - explained variance per PC
2. Biplot - arrows for metrics (time, topic, typing), dots for participants colored by condition (high/low creativity)
3. Loadings table - PC1, PC2, PC3 values per metric

**Output:**
- `output/m37_pca_switch_bertopic_{domain}.png` (4 images)
- `output/m37_pca_switch_lda_{domain}.png` (4 images)
- `output/m37_scores_bertopic_{domain}.csv` (4 CSVs)
- `output/m37_scores_lda_{domain}.csv` (4 CSVs)

## Data Flow

```
Game.csv + wiki_texts.json
       |
       v
compute_bertopic.py --> data/bertopic_model.json
       |
       v
  +---------+---------+---------+
  |         |         |         |
  v         v         v         |
 M34       M35       M36        |
(time)   (topic)   (typing)     |
  |         |         |         |
  +----+----+----+----+         |
       |                        |
       v                        |
      M37 (PCA)                 |
       |                        |
       v                        |
  scores CSVs                   |
  (for MANOVA)                  |
```

## Visual Style

All visualizations follow M10 dark theme:
- Background: `#0d1117`
- Text: `#e6edf3`
- Grid: `#21262d`
- Spines: `#30363d`
- Labels: `#8b949e`
- Explore color: green
- Exploit color: blue

## Dependencies

Existing: pandas, numpy, matplotlib, scikit-learn, scipy
New (for BERTopic): bertopic, sentence-transformers, umap-learn, hdbscan

# Switch Rate Features + PCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 3 switch rate metrics (time, topic, typing) per participant per domain, then run PCA to produce composite explore/exploit scores.

**Architecture:** 5 independent scripts following existing M-series pattern. `compute_bertopic.py` preprocesses BERTopic topics. M34/M35/M36 each compute one switch rate and save CSV + per-domain visualizations. M37 loads the CSVs and runs PCA per domain.

**Tech Stack:** Python 3.13, pandas, numpy, matplotlib, scikit-learn, bertopic, sentence-transformers

**Python executable:** `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe`

**Visual style constants (reuse across all scripts):**
```python
BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'
BORDER_COLOR = '#30363d'
MUTED_COLOR = '#8b949e'
EXPLOIT_COLOR = '#2196F3'
EXPLORE_COLOR = '#4CAF50'
```

**Data notes:**
- 4 real domains: art_history, ecology, economics, psychology (filter out practice where IsPractice==1)
- 20 participants in pilot data
- `load_trials()` from helpers.py returns structured trial list with page_visits, paste_times, typing_intervals
- Each trial dict has: pid, trial, domain, condition, duration, page_visits, paste_times, typing_intervals
- Each page_visit has: title, start, end, duration, nav_type
- `topic_model.json` has `topic_distributions` mapping slug -> probability array (argmax = dominant topic)

---

### Task 1: Install BERTopic dependencies

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
pandas
numpy
matplotlib
scikit-learn
scipy
bertopic
sentence-transformers
umap-learn
hdbscan
```

- [ ] **Step 2: Install dependencies**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe -m pip install bertopic sentence-transformers umap-learn hdbscan`

Expected: Successful installation (may take a few minutes for sentence-transformers model download)

- [ ] **Step 3: Verify import**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe -c "from bertopic import BERTopic; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt with BERTopic dependencies"
```

---

### Task 2: compute_bertopic.py

**Files:**
- Create: `scripts/compute_bertopic.py`
- Output: `data/bertopic_model.json`

- [ ] **Step 1: Create compute_bertopic.py**

```python
#!/usr/bin/env python3
"""
Compute BERTopic model for Wikipedia articles.
===============================================
Reads wiki_texts.json, trains BERTopic, outputs bertopic_model.json
with article_title -> topic_id mapping.
"""

import json
import numpy as np
from pathlib import Path
from bertopic import BERTopic

DATA_DIR = Path(__file__).parent / '..' / 'data'
INPUT_PATH = DATA_DIR / 'wiki_texts.json'
OUTPUT_PATH = DATA_DIR / 'bertopic_model.json'


def main():
    print("[compute_bertopic] Loading wiki texts...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        wiki = json.load(f)

    slugs = sorted(wiki.keys())
    texts = [wiki[s] for s in slugs]
    print(f"  {len(slugs)} articles")

    print("  Training BERTopic...")
    model = BERTopic(verbose=True)
    topics, probs = model.fit_transform(texts)

    topic_info = model.get_topic_info()
    n_topics = len(topic_info) - 1  # exclude -1 (outlier topic)
    print(f"  Found {n_topics} topics (+ outlier topic -1)")

    topic_assignments = {}
    for slug, topic_id in zip(slugs, topics):
        topic_assignments[slug] = int(topic_id)

    topic_words = {}
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            topic_words[str(topic_id)] = ["outlier"]
            continue
        words = model.get_topic(topic_id)
        topic_words[str(topic_id)] = [w for w, _ in words[:15]]

    result = {
        "n_topics": n_topics,
        "slugs": slugs,
        "topic_assignments": topic_assignments,
        "topic_words": topic_words,
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {OUTPUT_PATH}")

    topic_counts = {}
    for t in topics:
        topic_counts[t] = topic_counts.get(t, 0) + 1
    print("\n  Topic distribution:")
    for t in sorted(topic_counts.keys()):
        print(f"    Topic {t}: {topic_counts[t]} articles")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run compute_bertopic.py**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe scripts/compute_bertopic.py`

Expected: Creates `data/bertopic_model.json` with topic_assignments mapping. Check that most articles get a non-outlier topic.

- [ ] **Step 3: Verify output**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe -c "import json; d=json.load(open('data/bertopic_model.json')); print('Topics:', d['n_topics']); print('Articles:', len(d['topic_assignments'])); outliers=sum(1 for v in d['topic_assignments'].values() if v==-1); print('Outliers:', outliers)"`

Expected: Shows topic count, article count, and outlier count.

- [ ] **Step 4: Commit**

```bash
git add scripts/compute_bertopic.py data/bertopic_model.json
git commit -m "feat(scripts): add compute_bertopic.py and trained model"
```

---

### Task 3: m34_switch_time.py

**Files:**
- Create: `scripts/m34_switch_time.py`
- Output: `output/m34_switch_time_{domain}.png` (4 images), `output/m34_switch_time.csv`

- [ ] **Step 1: Create m34_switch_time.py**

```python
#!/usr/bin/env python3
"""
M34: Switch Rate - Time-Based (60s threshold)
=============================================
Per page: >60s dwell time = Exploit, else = Explore.
Switch rate = transitions between states / (N-1).
Output: per-domain grid plots + CSV with switch rates.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

THRESHOLD_S = 60.0

BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'
BORDER_COLOR = '#30363d'
MUTED_COLOR = '#8b949e'
EXPLOIT_COLOR = '#2196F3'
EXPLORE_COLOR = '#4CAF50'


def compute_switch_rate(labels):
    if len(labels) < 2:
        return np.nan
    transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    return transitions / (len(labels) - 1)


def classify_pages_time(page_visits):
    labels = []
    durations = []
    for pv in page_visits:
        dur = pv['duration']
        durations.append(dur)
        labels.append('exploit' if dur > THRESHOLD_S else 'explore')
    return labels, durations


def plot_domain(domain, participants_data, output_path):
    n = len(participants_data)
    cols = 4
    rows = max(1, int(np.ceil(n / cols)))

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.2))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f'M34: Switch Rate (Time > {THRESHOLD_S:.0f}s) - {domain}',
        fontsize=14, color=TEXT_COLOR, fontweight='bold', y=0.99,
    )

    axes_flat = np.array(axes).flatten() if n > 1 else [axes] if rows == 1 and cols == 1 else axes.flatten()

    for i, (pid, labels, durations, sr) in enumerate(participants_data):
        ax = axes_flat[i]
        ax.set_facecolor(BG_COLOR)

        x = np.arange(1, len(labels) + 1)
        colors = [EXPLOIT_COLOR if l == 'exploit' else EXPLORE_COLOR for l in labels]
        ax.bar(x, durations, color=colors, edgecolor='none', width=0.7, zorder=3)
        ax.axhline(y=THRESHOLD_S, color='#FF9800', linewidth=1, linestyle='--', alpha=0.8, zorder=2)

        ax.set_title(f'User {pid} - SR: {sr:.2f}', fontsize=10, color=TEXT_COLOR, fontweight='bold', pad=5)
        ax.set_xlabel('Page #', fontsize=8, color=MUTED_COLOR)
        ax.set_ylabel('Time (s)', fontsize=8, color=MUTED_COLOR)
        ax.set_xticks(x)
        ax.tick_params(colors=MUTED_COLOR, labelsize=7)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0, axis='y')
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)

    for k in range(n, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[M34] Switch Rate - Time-Based")

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    print(f"  {len(trials)} trials from {len(pids)} participants")

    rows_csv = []

    domain_data = {}
    for pid in pids:
        for tr in pid_trials[pid]:
            domain = tr['domain']
            if domain == 'practice':
                continue
            labels, durations = classify_pages_time(tr['page_visits'])
            sr = compute_switch_rate(labels)
            rows_csv.append({'participant_id': pid, 'domain': domain, 'switch_rate': sr})

            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append((pid, labels, durations, sr))

    for domain in sorted(domain_data.keys()):
        output_path = OUTPUT_DIR / f'm34_switch_time_{domain}.png'
        plot_domain(domain, domain_data[domain], output_path)

    df_out = pd.DataFrame(rows_csv)
    csv_path = OUTPUT_DIR / 'm34_switch_time.csv'
    df_out.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    print(f"\nSwitch rate stats by domain:")
    for domain in sorted(domain_data.keys()):
        rates = [r[3] for r in domain_data[domain] if not np.isnan(r[3])]
        print(f"  {domain}: mean={np.mean(rates):.3f}, std={np.std(rates):.3f}, n={len(rates)}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run m34_switch_time.py**

Run: `cd scripts && /c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe m34_switch_time.py`

Expected: Creates 4 PNG files (`m34_switch_time_{domain}.png`) and `m34_switch_time.csv`. Verify visually that bars are colored green/blue with the 60s dashed line.

- [ ] **Step 3: Verify CSV output**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe -c "import pandas as pd; df=pd.read_csv('output/m34_switch_time.csv'); print(df.head(10)); print(f'\nShape: {df.shape}'); print(f'Domains: {sorted(df.domain.unique())}')" `

Expected: CSV with columns participant_id, domain, switch_rate. One row per participant per domain.

- [ ] **Step 4: Commit**

```bash
git add scripts/m34_switch_time.py
git commit -m "feat(scripts): add M34 time-based switch rate"
```

---

### Task 4: m35_switch_topic.py

**Files:**
- Create: `scripts/m35_switch_topic.py`
- Read: `data/topic_model.json`, `data/bertopic_model.json`
- Output: `output/m35_switch_{bertopic,lda}_{domain}.png` (8 images), `output/m35_switch_{bertopic,lda}.csv` (2 CSVs)

- [ ] **Step 1: Create m35_switch_topic.py**

```python
#!/usr/bin/env python3
"""
M35: Switch Rate - Topic-Based (BERTopic + LDA)
================================================
Per page: assign dominant topic from topic model.
Transition = topic_id differs between consecutive pages.
Switch rate = transitions / (N-1).
Runs for both BERTopic and LDA models.
Output: per-domain grid plots + CSV with switch rates.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'
BERTOPIC_PATH = DATA_DIR / 'bertopic_model.json'
LDA_PATH = DATA_DIR / 'topic_model.json'

BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'
BORDER_COLOR = '#30363d'
MUTED_COLOR = '#8b949e'

TOPIC_PALETTE = [
    '#4FC3F7', '#81C784', '#FFB74D', '#F06292', '#CE93D8',
    '#80DEEA', '#FFCC80', '#A5D6A7', '#EF9A9A', '#B0BEC5',
    '#FFF176', '#90CAF9', '#C5E1A5', '#FFAB91', '#80CBC4',
]


def compute_switch_rate(labels):
    if len(labels) < 2:
        return np.nan
    transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    return transitions / (len(labels) - 1)


def load_bertopic_topics():
    with open(BERTOPIC_PATH, 'r') as f:
        bm = json.load(f)
    return bm['topic_assignments']


def load_lda_topics():
    with open(LDA_PATH, 'r') as f:
        tm = json.load(f)
    lda_assignments = {}
    for slug, dist in tm['topic_distributions'].items():
        lda_assignments[slug] = int(np.argmax(dist))
    return lda_assignments


def get_page_topics(page_visits, topic_assignments):
    topics = []
    for pv in page_visits:
        title = pv['title']
        topic = topic_assignments.get(title, -1)
        topics.append(topic)
    return topics


def plot_domain(domain, participants_data, model_name, output_path):
    n = len(participants_data)
    cols = 4
    rows = max(1, int(np.ceil(n / cols)))

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.2))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f'M35: Switch Rate (Topic Change, {model_name}) - {domain}',
        fontsize=14, color=TEXT_COLOR, fontweight='bold', y=0.99,
    )

    axes_flat = np.array(axes).flatten() if n > 1 else [axes] if rows == 1 and cols == 1 else axes.flatten()

    for i, (pid, topics, sr) in enumerate(participants_data):
        ax = axes_flat[i]
        ax.set_facecolor(BG_COLOR)

        x = np.arange(1, len(topics) + 1)
        colors = [TOPIC_PALETTE[t % len(TOPIC_PALETTE)] if t >= 0 else '#555555' for t in topics]
        ax.bar(x, [1] * len(topics), color=colors, edgecolor='none', width=0.7, zorder=3)

        for j in range(1, len(topics)):
            if topics[j] != topics[j - 1]:
                ax.axvline(x=j + 0.5, color='#FF9800', linewidth=1.2, linestyle='--', alpha=0.7, zorder=2)

        ax.set_title(f'User {pid} - SR: {sr:.2f}', fontsize=10, color=TEXT_COLOR, fontweight='bold', pad=5)
        ax.set_xlabel('Page #', fontsize=8, color=MUTED_COLOR)
        ax.set_ylabel('Topic', fontsize=8, color=MUTED_COLOR)
        ax.set_xticks(x)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.tick_params(colors=MUTED_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)

        unique_topics = sorted(set(topics))
        for t in unique_topics:
            idx_list = [j + 1 for j, tt in enumerate(topics) if tt == t]
            c = TOPIC_PALETTE[t % len(TOPIC_PALETTE)] if t >= 0 else '#555555'
            label = f'T{t}' if t >= 0 else 'outlier'
            ax.plot([], [], 's', color=c, label=label, markersize=6)
        if len(unique_topics) <= 8:
            ax.legend(fontsize=6, facecolor=BG_COLOR, edgecolor=BORDER_COLOR,
                      labelcolor=LABEL_COLOR, loc='upper right', ncol=2)

    for k in range(n, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def process_model(model_name, topic_assignments, pids, pid_trials):
    rows_csv = []
    domain_data = {}

    for pid in pids:
        for tr in pid_trials[pid]:
            domain = tr['domain']
            if domain == 'practice':
                continue
            topics = get_page_topics(tr['page_visits'], topic_assignments)
            sr = compute_switch_rate(topics)
            rows_csv.append({'participant_id': pid, 'domain': domain, 'switch_rate': sr})

            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append((pid, topics, sr))

    for domain in sorted(domain_data.keys()):
        output_path = OUTPUT_DIR / f'm35_switch_{model_name}_{domain}.png'
        plot_domain(domain, domain_data[domain], model_name, output_path)

    df_out = pd.DataFrame(rows_csv)
    csv_path = OUTPUT_DIR / f'm35_switch_{model_name}.csv'
    df_out.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    print(f"\n  {model_name} switch rate stats by domain:")
    for domain in sorted(domain_data.keys()):
        rates = [r[2] for r in domain_data[domain] if not np.isnan(r[2])]
        print(f"    {domain}: mean={np.mean(rates):.3f}, std={np.std(rates):.3f}, n={len(rates)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[M35] Switch Rate - Topic-Based")

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    print(f"  {len(trials)} trials from {len(pids)} participants")

    print("\n  Loading BERTopic model...")
    bertopic_topics = load_bertopic_topics()
    print(f"  {len(bertopic_topics)} article assignments")
    process_model('bertopic', bertopic_topics, pids, pid_trials)

    print("\n  Loading LDA model...")
    lda_topics = load_lda_topics()
    print(f"  {len(lda_topics)} article assignments")
    process_model('lda', lda_topics, pids, pid_trials)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run m35_switch_topic.py**

Run: `cd scripts && /c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe m35_switch_topic.py`

Expected: Creates 8 PNG files and 2 CSVs. Each participant subplot shows colored bars per topic with dashed vertical lines at topic transitions.

- [ ] **Step 3: Verify CSV output**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe -c "import pandas as pd; b=pd.read_csv('output/m35_switch_bertopic.csv'); l=pd.read_csv('output/m35_switch_lda.csv'); print('BERTopic:'); print(b.head()); print(f'\nLDA:'); print(l.head())"`

Expected: Both CSVs with columns participant_id, domain, switch_rate.

- [ ] **Step 4: Commit**

```bash
git add scripts/m35_switch_topic.py
git commit -m "feat(scripts): add M35 topic-based switch rate (BERTopic + LDA)"
```

---

### Task 5: m36_switch_typing.py

**Files:**
- Create: `scripts/m36_switch_typing.py`
- Read: `scripts/helpers.py` (load_trials, typing detection), `scripts/m18_typing_binary.py` (page_had_typing_or_paste)
- Output: `output/m36_switch_typing_{domain}.png` (4 images), `output/m36_switch_typing.csv`

- [ ] **Step 1: Create m36_switch_typing.py**

```python
#!/usr/bin/env python3
"""
M36: Switch Rate - Typing/Pasting Binary (M18 logic)
====================================================
Per page: typing or paste = Exploit, else = Explore (same as M18).
Switch rate = transitions between states / (N-1).
Output: per-domain grid plots + CSV with switch rates.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR
from m18_typing_binary import page_had_typing_or_paste

BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'
BORDER_COLOR = '#30363d'
MUTED_COLOR = '#8b949e'
EXPLOIT_COLOR = '#2196F3'
EXPLORE_COLOR = '#4CAF50'


def compute_switch_rate(labels):
    if len(labels) < 2:
        return np.nan
    transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    return transitions / (len(labels) - 1)


def classify_pages_typing(page_visits, typing_intervals, paste_times):
    labels = []
    for pv in page_visits:
        is_exploit = page_had_typing_or_paste(pv, typing_intervals, paste_times)
        labels.append('exploit' if is_exploit else 'explore')
    return labels


def plot_domain(domain, participants_data, output_path):
    n = len(participants_data)
    cols = 4
    rows = max(1, int(np.ceil(n / cols)))

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.2))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f'M36: Switch Rate (Typing/Paste) - {domain}',
        fontsize=14, color=TEXT_COLOR, fontweight='bold', y=0.99,
    )

    axes_flat = np.array(axes).flatten() if n > 1 else [axes] if rows == 1 and cols == 1 else axes.flatten()

    for i, (pid, labels, sr) in enumerate(participants_data):
        ax = axes_flat[i]
        ax.set_facecolor(BG_COLOR)

        x = np.arange(1, len(labels) + 1)
        colors = [EXPLOIT_COLOR if l == 'exploit' else EXPLORE_COLOR for l in labels]
        ax.bar(x, [1] * len(labels), color=colors, edgecolor='none', width=0.7, zorder=3)

        ax.set_title(f'User {pid} - SR: {sr:.2f}', fontsize=10, color=TEXT_COLOR, fontweight='bold', pad=5)
        ax.set_xlabel('Page #', fontsize=8, color=MUTED_COLOR)
        ax.set_xticks(x)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.tick_params(colors=MUTED_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)

    for k in range(n, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[M36] Switch Rate - Typing/Pasting")

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    print(f"  {len(trials)} trials from {len(pids)} participants")

    rows_csv = []
    domain_data = {}

    for pid in pids:
        for tr in pid_trials[pid]:
            domain = tr['domain']
            if domain == 'practice':
                continue
            labels = classify_pages_typing(tr['page_visits'], tr['typing_intervals'], tr['paste_times'])
            sr = compute_switch_rate(labels)
            rows_csv.append({'participant_id': pid, 'domain': domain, 'switch_rate': sr})

            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append((pid, labels, sr))

    for domain in sorted(domain_data.keys()):
        output_path = OUTPUT_DIR / f'm36_switch_typing_{domain}.png'
        plot_domain(domain, domain_data[domain], output_path)

    df_out = pd.DataFrame(rows_csv)
    csv_path = OUTPUT_DIR / 'm36_switch_typing.csv'
    df_out.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    print(f"\nSwitch rate stats by domain:")
    for domain in sorted(domain_data.keys()):
        rates = [r[2] for r in domain_data[domain] if not np.isnan(r[2])]
        print(f"  {domain}: mean={np.mean(rates):.3f}, std={np.std(rates):.3f}, n={len(rates)}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run m36_switch_typing.py**

Run: `cd scripts && /c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe m36_switch_typing.py`

Expected: Creates 4 PNG files and `m36_switch_typing.csv`. Each participant subplot shows green/blue bars with uniform height.

- [ ] **Step 3: Verify CSV output**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe -c "import pandas as pd; df=pd.read_csv('output/m36_switch_typing.csv'); print(df.head(10)); print(f'\nShape: {df.shape}')"`

Expected: CSV with columns participant_id, domain, switch_rate.

- [ ] **Step 4: Commit**

```bash
git add scripts/m36_switch_typing.py
git commit -m "feat(scripts): add M36 typing-based switch rate"
```

---

### Task 6: m37_pca_switch.py

**Files:**
- Create: `scripts/m37_pca_switch.py`
- Read: `output/m34_switch_time.csv`, `output/m35_switch_bertopic.csv`, `output/m35_switch_lda.csv`, `output/m36_switch_typing.csv`
- Output: `output/m37_pca_switch_{bertopic,lda}_{domain}.png` (8 images), `output/m37_scores_{bertopic,lda}_{domain}.csv` (8 CSVs)

- [ ] **Step 1: Create m37_pca_switch.py**

```python
#!/usr/bin/env python3
"""
M37: PCA on Switch Rate Features
=================================
Loads M34/M35/M36 switch rate CSVs, runs PCA per domain (no standardization).
Produces scree plot, biplot, and loadings table per domain.
Runs twice: once with BERTopic topic rates, once with LDA.
Output: per-domain PCA plots + score CSVs.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA
from pathlib import Path

from helpers import load_trials, OUTPUT_DIR

SCRIPT_DIR = Path(__file__).parent

BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'
BORDER_COLOR = '#30363d'
MUTED_COLOR = '#8b949e'
BAR_COLOR = '#4FC3F7'
LINE_COLOR = '#FF9800'
ARROW_COLOR = '#FF9800'

FEATURE_NAMES = ['SR Time', 'SR Topic', 'SR Typing']

CONDITION_COLORS = {
    'high-creativity': '#4FC3F7',
    'low-creativity': '#F06292',
}
DEFAULT_DOT_COLOR = '#4FC3F7'


def load_switch_rates(topic_model_name):
    time_df = pd.read_csv(OUTPUT_DIR / 'm34_switch_time.csv')
    topic_df = pd.read_csv(OUTPUT_DIR / f'm35_switch_{topic_model_name}.csv')
    typing_df = pd.read_csv(OUTPUT_DIR / 'm36_switch_typing.csv')

    time_df = time_df.rename(columns={'switch_rate': 'sr_time'})
    topic_df = topic_df.rename(columns={'switch_rate': 'sr_topic'})
    typing_df = typing_df.rename(columns={'switch_rate': 'sr_typing'})

    merged = time_df.merge(topic_df, on=['participant_id', 'domain'], suffixes=('', '_topic'))
    merged = merged.merge(typing_df, on=['participant_id', 'domain'], suffixes=('', '_typing'))

    return merged


def get_participant_conditions(trials):
    conditions = {}
    for tr in trials:
        conditions[tr['pid']] = tr['condition']
    return conditions


def plot_pca_domain(domain, X, pids, conditions, pca, scores, topic_model_name, output_path):
    fig, (ax_scree, ax_biplot, ax_table) = plt.subplots(1, 3, figsize=(22, 7),
        gridspec_kw={'width_ratios': [1, 1.3, 0.7]})
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f'M37: PCA on Switch Rates ({topic_model_name}) - {domain}',
        fontsize=14, color=TEXT_COLOR, fontweight='bold', y=0.99,
    )

    # Panel 1: Scree plot
    ax_scree.set_facecolor(BG_COLOR)
    pct = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(pct)
    pc_labels = [f'PC{i+1}' for i in range(len(pct))]

    bars = ax_scree.bar(pc_labels, pct, color=BAR_COLOR, zorder=2)
    ax_scree.plot(pc_labels, cumulative, color=LINE_COLOR, marker='o', linewidth=2, zorder=3)

    for bar, val in zip(bars, pct):
        ax_scree.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f'{val:.1f}%', ha='center', va='bottom', color=TEXT_COLOR, fontsize=10)

    ax_scree.set_ylabel('Variance Explained (%)', color=LABEL_COLOR)
    ax_scree.set_title('Scree Plot', color=TEXT_COLOR, fontweight='bold')
    ax_scree.set_ylim(0, 110)
    ax_scree.tick_params(colors=MUTED_COLOR)
    ax_scree.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax_scree.spines.values():
        spine.set_color(BORDER_COLOR)

    # Panel 2: Biplot
    ax_biplot.set_facecolor(BG_COLOR)

    for i, pid in enumerate(pids):
        cond = conditions.get(pid, '')
        color = CONDITION_COLORS.get(cond, DEFAULT_DOT_COLOR)
        ax_biplot.scatter(scores[i, 0], scores[i, 1], color=color, s=50, alpha=0.8,
                          edgecolors='white', linewidth=0.5, zorder=3)
        ax_biplot.annotate(f'P{pid}', (scores[i, 0], scores[i, 1]),
                           fontsize=7, color=LABEL_COLOR, ha='left', va='bottom',
                           xytext=(4, 4), textcoords='offset points')

    x_range = scores[:, 0].max() - scores[:, 0].min() if len(scores) > 1 else 1
    y_range = scores[:, 1].max() - scores[:, 1].min() if len(scores) > 1 else 1
    scale = 0.4 * max(x_range, y_range)

    loadings = pca.components_
    for j, name in enumerate(FEATURE_NAMES):
        lx = loadings[0, j] * scale
        ly = loadings[1, j] * scale
        ax_biplot.annotate('', xy=(lx, ly), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2.5))
        ax_biplot.text(lx * 1.15, ly * 1.15, name, color=ARROW_COLOR, fontsize=10,
                       ha='center', va='center', fontweight='bold')

    ax_biplot.axhline(0, color=BORDER_COLOR, linewidth=0.5)
    ax_biplot.axvline(0, color=BORDER_COLOR, linewidth=0.5)
    ax_biplot.set_xlabel(f'PC1 ({pct[0]:.1f}%)', color=LABEL_COLOR)
    ax_biplot.set_ylabel(f'PC2 ({pct[1]:.1f}%)', color=LABEL_COLOR)
    ax_biplot.set_title('Biplot', color=TEXT_COLOR, fontweight='bold')
    ax_biplot.tick_params(colors=MUTED_COLOR)
    for spine in ax_biplot.spines.values():
        spine.set_color(BORDER_COLOR)

    # Panel 3: Loadings table
    ax_table.set_facecolor(BG_COLOR)
    ax_table.axis('off')
    ax_table.set_title('Loadings', color=TEXT_COLOR, fontweight='bold')

    col_labels = [f'PC{i+1}' for i in range(3)]
    row_labels = FEATURE_NAMES
    cell_text = []
    for j in range(3):
        row = [f'{loadings[i, j]:+.3f}' for i in range(min(3, len(loadings)))]
        cell_text.append(row)

    table = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for key, cell in table.get_celld().items():
        cell.set_facecolor(BG_COLOR)
        cell.set_edgecolor(BORDER_COLOR)
        cell.set_text_props(color=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def run_pca_for_model(topic_model_name, conditions):
    print(f"\n  Processing {topic_model_name}...")
    merged = load_switch_rates(topic_model_name)

    domains = sorted(merged['domain'].unique())
    for domain in domains:
        dom_df = merged[merged['domain'] == domain].dropna()
        if len(dom_df) < 3:
            print(f"    {domain}: skipping, only {len(dom_df)} participants")
            continue

        pids = dom_df['participant_id'].values
        X = dom_df[['sr_time', 'sr_topic', 'sr_typing']].values

        pca = PCA(n_components=min(3, X.shape[0], X.shape[1]))
        scores = pca.fit_transform(X)

        print(f"\n    {domain} ({topic_model_name}):")
        print(f"      N={len(pids)}")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"      PC{i+1}: {var*100:.1f}%")
        print(f"      Loadings:")
        for i, comp in enumerate(pca.components_):
            parts = ', '.join(f'{FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(len(FEATURE_NAMES)))
            print(f"        PC{i+1}: {parts}")

        output_path = OUTPUT_DIR / f'm37_pca_switch_{topic_model_name}_{domain}.png'
        plot_pca_domain(domain, X, pids, conditions, pca, scores, topic_model_name, output_path)

        scores_df = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(scores.shape[1])])
        scores_df.insert(0, 'participant_id', pids)
        scores_df.insert(1, 'domain', domain)
        csv_path = OUTPUT_DIR / f'm37_scores_{topic_model_name}_{domain}.csv'
        scores_df.to_csv(csv_path, index=False)
        print(f'      Saved: {csv_path}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[M37] PCA on Switch Rates")

    trials = load_trials()
    conditions = get_participant_conditions(trials)

    run_pca_for_model('bertopic', conditions)
    run_pca_for_model('lda', conditions)

    print("\nDone.")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run m37_pca_switch.py**

Run: `cd scripts && /c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe m37_pca_switch.py`

Expected: Creates 8 PNGs and 8 CSVs. Each image shows 3 panels (scree, biplot, loadings table). Console prints explained variance and loadings per domain.

- [ ] **Step 3: Verify CSV scores output**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe -c "import pandas as pd; df=pd.read_csv('output/m37_scores_bertopic_art_history.csv'); print(df); print(f'\nColumns: {list(df.columns)}')"`

Expected: CSV with columns participant_id, domain, PC1, PC2, PC3. One row per participant.

- [ ] **Step 4: Commit**

```bash
git add scripts/m37_pca_switch.py
git commit -m "feat(scripts): add M37 PCA on switch rate features"
```

---

### Task 7: End-to-end verification

- [ ] **Step 1: Run full pipeline**

Run the complete pipeline in order:

```bash
cd scripts
/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe compute_bertopic.py
/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe m34_switch_time.py
/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe m35_switch_topic.py
/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe m36_switch_typing.py
/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe m37_pca_switch.py
```

- [ ] **Step 2: Verify all outputs exist**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe -c "from pathlib import Path; o=Path('output'); files=sorted(f.name for f in o.iterdir() if f.name.startswith(('m34','m35','m36','m37'))); print(f'{len(files)} files:'); [print(f'  {f}') for f in files]"`

Expected: 4 (M34) + 8 (M35) + 4 (M36) + 8 PNG (M37) + 1 (M34) + 2 (M35) + 1 (M36) + 8 (M37) CSV = 36 files total.

- [ ] **Step 3: Spot-check PCA results consistency**

Run: `/c/Users/elay9/AppData/Local/Programs/Python/Python313/python.exe -c "
import pandas as pd
for model in ['bertopic', 'lda']:
    print(f'=== {model} ===')
    for domain in ['art_history', 'ecology', 'economics', 'psychology']:
        df = pd.read_csv(f'output/m37_scores_{model}_{domain}.csv')
        print(f'  {domain}: {len(df)} participants, PC1 range=[{df.PC1.min():.3f}, {df.PC1.max():.3f}]')
"`

Expected: Each domain has ~20 participants (some may have fewer due to missing data). PC1 ranges should be reasonable (not all zeros or identical values).

# Professors Analysis Plan — Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all missing analyses from `docs/professors_plan.md` — Topic Modeling (1.א), combined measures (4.א, 4.ב), PCA on raw signals, and fix incorrect M20 mapping.

**Architecture:** Each analysis is a standalone Python script in `scripts/` following the existing pattern (load data → compute → visualize → save PNG). A shared `compute_topics.py` produces `data/topic_model.json` consumed by m26 and m29. Combined measures (m27, m28) are variations of M20, reusing `build_lsa()` and `plot_trial_grid_with_switches()`.

**Tech Stack:** Python 3.10+, numpy, scipy, scikit-learn (LDA, PCA), matplotlib, pandas

---

## Critical Fix: M20 Mapping Error

M20 classifies **pages by typing/paste (M18)** and transitions by LSA cosine distance.
The professors_plan.md incorrectly maps M20 → 4.א (Semantics + Median time).

**Correct mapping:**
- **4.א (Semantics + Median time)** — does NOT exist, needs to be built
- **4.ב (Semantics + 60s)** — does NOT exist, needs to be built
- **4.ג (Semantics + Type/Paste)** — IS M20, already exists ✅

---

## File Structure

```
scripts/
├── compute_topics.py           # NEW — Train LDA, save topic_model.json
├── m26_topic_modeling.py       # NEW — 1.א visualization (topic distance grid)
├── m27_lsa_time_median.py      # NEW — 4.א (LSA transitions + time-median pages)
├── m28_lsa_60s.py              # NEW — 4.ב (LSA transitions + 60s pages)
└── m29_pca_raw.py              # NEW — PCA on 3 raw continuous signals
data/
└── topic_model.json            # NEW — LDA output artifact
output/
├── m26_trial1.png, m26_trial2.png      # 1.א
├── m26_topic_overview.png              # topic composition
├── m27_trial1.png, m27_trial2.png      # 4.א
├── m28_trial1.png, m28_trial2.png      # 4.ב
├── m29_pca_scree.png                   # PCA scree plot
├── m29_pca_biplot.png                  # PCA biplot
└── m29_pca_scores.png                  # PC1 score per participant
```

---

### Task 1: Fix professors_plan.md Mapping

**Files:**
- Modify: `docs/professors_plan.md:83-108` (section 4 status table)

- [ ] **Step 1: Update the status table**

Replace the status table entries for section 4:

```markdown
### 4.א. סמנטיקה + Median זמן — תרשים M27

| ציר | תנאי ל-Exploit |
|-----|----------------|
| סמנטיקה (1.ב.) | מרחק cosine נמוך (מתחת ל-Median של הדומיין) |
| זמן (2.א.) | זמן בדף מעל Median האישי |

- **סטטוס:** ❌ צריך לבנות (M20 מסווג דפים לפי typing/paste, לא לפי זמן!)

### 4.ב. סמנטיקה + 60 שניות — תרשים M28

| ציר | תנאי ל-Exploit |
|-----|----------------|
| סמנטיקה (1.ב.) | מרחק cosine נמוך |
| זמן (2.ב.) | זמן בדף מעל 60 שניות |

- **סטטוס:** ❌ צריך לבנות

### 4.ג. סמנטיקה + Type/Paste — תרשים M20

| ציר | תנאי ל-Exploit |
|-----|----------------|
| סמנטיקה (1.ב.) | מרחק cosine נמוך |
| Type/Paste (3.ב.) | יש פעילות כתיבה/הדבקה בדף |

- **סטטוס:** ✅ M20 קיים — זה בדיוק מה ש-M20 עושה!
```

And fix the summary table:

```markdown
| 4.א | סמנטיקה + Median זמן | M27 | ❌ צריך לבנות |
| 4.ב | סמנטיקה + 60s | M28 | ❌ צריך לבנות |
| 4.ג | סמנטיקה + Type/Paste | M20 | ✅ קיים |
```

- [ ] **Step 2: Commit**

```bash
git add docs/professors_plan.md
git commit -m "fix(docs): correct M20 mapping — M20 is 4.ג (type/paste), not 4.א (time)"
```

---

### Task 2: Topic Model Infrastructure

**Files:**
- Create: `scripts/compute_topics.py`
- Output: `data/topic_model.json`

- [ ] **Step 1: Create compute_topics.py**

```python
#!/usr/bin/env python3
"""
Compute Topic Model — LDA on Wikipedia article corpus.
======================================================
Trains LDA on wiki_texts.json, saves topic distributions
and pairwise Jensen-Shannon distances to data/topic_model.json.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

DATA_DIR = Path(__file__).parent / '..' / 'data'
WIKI_PATH = DATA_DIR / 'wiki_texts.json'
OUTPUT_PATH = DATA_DIR / 'topic_model.json'

N_TOPICS = 10
RANDOM_STATE = 42


def jsd(p, q):
    """Jensen-Shannon divergence between two probability distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m + 1e-12))
    kl_qm = np.sum(q * np.log2(q / m + 1e-12))
    return np.sqrt(0.5 * (kl_pm + kl_qm))


def main():
    with open(WIKI_PATH, 'r', encoding='utf-8') as f:
        wiki_texts = json.load(f)

    slugs = sorted(wiki_texts.keys())
    docs = [wiki_texts[s] for s in slugs]
    print(f'Loaded {len(slugs)} articles')

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english',
        max_features=5000,
    )
    dtm = vectorizer.fit_transform(docs)
    print(f'Vocabulary size: {len(vectorizer.get_feature_names_out())}')

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        max_iter=50,
        learning_method='batch',
    )
    doc_topics = lda.fit_transform(dtm)
    print(f'LDA trained: {N_TOPICS} topics, perplexity={lda.perplexity(dtm):.1f}')

    feature_names = vectorizer.get_feature_names_out()
    topic_words = {}
    for i in range(N_TOPICS):
        top_idx = lda.components_[i].argsort()[-10:][::-1]
        topic_words[str(i)] = [feature_names[j] for j in top_idx]
        print(f'  Topic {i}: {", ".join(topic_words[str(i)][:5])}')

    topic_distributions = {}
    for i, slug in enumerate(slugs):
        topic_distributions[slug] = doc_topics[i].tolist()

    distances = {}
    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            key = f'{slugs[i]}|||{slugs[j]}'
            distances[key] = float(jsd(doc_topics[i], doc_topics[j]))

    dists_arr = np.array(list(distances.values()))
    result = {
        'n_topics': N_TOPICS,
        'slugs': slugs,
        'topic_words': topic_words,
        'topic_distributions': topic_distributions,
        'distances': distances,
        'stats': {
            'mean': float(dists_arr.mean()),
            'median': float(np.median(dists_arr)),
            'std': float(dists_arr.std()),
            'min': float(dists_arr.min()),
            'max': float(dists_arr.max()),
            'n_pairs': len(distances),
        },
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {OUTPUT_PATH}')
    print(f'Distance stats: mean={result["stats"]["mean"]:.4f}, '
          f'median={result["stats"]["median"]:.4f}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run and verify output**

Run: `cd scripts && python compute_topics.py`

Expected:
- Prints topic words for each topic
- Creates `data/topic_model.json`
- Distance stats printed

```bash
ls -la ../data/topic_model.json
python -c "import json; d=json.load(open('../data/topic_model.json')); print(f'Topics: {d[\"n_topics\"]}, Articles: {len(d[\"slugs\"])}, Pairs: {d[\"stats\"][\"n_pairs\"]}')"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/compute_topics.py data/topic_model.json
git commit -m "feat(1a): add topic model infrastructure — LDA on wiki corpus"
```

---

### Task 3: Topic Modeling Visualization (1.א = M26)

**Files:**
- Create: `scripts/m26_topic_modeling.py`
- Output: `output/m26_topic_overview.png`, `output/m26_trial1.png`, `output/m26_trial2.png`

- [ ] **Step 1: Create m26_topic_modeling.py**

```python
#!/usr/bin/env python3
"""
M26: Explore/Exploit — Topic Modeling (1.א)
============================================
Transitions: Exploit if JSD topic distance <= domain median.
Pages: Exploit if typing/paste on page (M18 logic, consistent with M20).
Uses topic_model.json distances instead of LSA cosine distance.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from m18_typing_binary import page_had_typing_or_paste
from m20_cross_subject_median import count_switches, plot_trial_grid_with_switches
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

DATA_DIR = Path(__file__).parent / '..' / 'data'
TOPIC_PATH = DATA_DIR / 'topic_model.json'


def load_topic_model():
    with open(TOPIC_PATH, 'r', encoding='utf-8') as f:
        tm = json.load(f)
    slugs = tm['slugs']
    slug_idx = {s: i for i, s in enumerate(slugs)}
    return tm, slugs, slug_idx


def topic_dist(tm, slug_a, slug_b):
    """Get pre-computed JSD between two articles."""
    key1 = f'{slug_a}|||{slug_b}'
    key2 = f'{slug_b}|||{slug_a}'
    if key1 in tm['distances']:
        return tm['distances'][key1]
    if key2 in tm['distances']:
        return tm['distances'][key2]
    return np.nan


def compute_domain_medians_topic(pid_trials, tm):
    """Median JSD per domain across all subjects."""
    domain_dists = defaultdict(list)

    for pid, trials in pid_trials.items():
        for tr in trials:
            pvs = tr['page_visits']
            domain = tr['domain']
            for i in range(1, len(pvs)):
                d = topic_dist(tm, pvs[i - 1]['title'], pvs[i]['title'])
                if not np.isnan(d):
                    domain_dists[domain].append(d)

    domain_medians = {}
    for domain, dists in domain_dists.items():
        domain_medians[domain] = np.median(dists)
        print(f'  {domain}: n={len(dists)}, median={domain_medians[domain]:.4f}')

    return domain_medians


def build_sequences_topic(pids, pid_trials, tm, domain_medians):
    pid_data = {}

    for pid in pids:
        pid_data[pid] = []
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            if len(pvs) < 2:
                continue

            domain = tr['domain']
            dist_threshold = domain_medians[domain]

            trans_dists = []
            for i in range(1, len(pvs)):
                trans_dists.append(topic_dist(tm, pvs[i - 1]['title'], pvs[i]['title']))

            points = []
            for i in range(len(pvs)):
                is_exploit = page_had_typing_or_paste(
                    pvs[i], tr['typing_intervals'], tr['paste_times'])
                points.append({
                    'x': i + 1,
                    'y': 0.5 if is_exploit else -0.5,
                    'type': 'page',
                })

                if i < len(pvs) - 1:
                    d = trans_dists[i]
                    is_exploit_d = d <= dist_threshold if not np.isnan(d) else False
                    points.append({
                        'x': i + 1.5,
                        'y': 0.5 if is_exploit_d else -0.5,
                        'type': 'transition',
                        'raw': d,
                        'threshold': dist_threshold,
                    })

            pid_data[pid].append({
                'trial': tr['trial'],
                'condition': tr['condition'],
                'points': points,
                'dist_threshold': dist_threshold,
                'domain': domain,
            })

    return pid_data


def plot_topic_overview(tm, pid_trials, pids):
    """Overview: topic distribution per domain + distance histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('M26: Topic Modeling Overview',
                 fontsize=14, color='#e6edf3', fontweight='bold')

    # Left: topic distance distribution per domain
    ax = axes[0]
    ax.set_facecolor('#0d1117')
    domain_dists = defaultdict(list)
    for pid in pids:
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            for i in range(1, len(pvs)):
                d = topic_dist(tm, pvs[i - 1]['title'], pvs[i]['title'])
                if not np.isnan(d):
                    domain_dists[tr['domain']].append(d)

    colors = ['#4FC3F7', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']
    for ci, (domain, dists) in enumerate(sorted(domain_dists.items())):
        ax.hist(dists, bins=20, alpha=0.6, label=domain,
                color=colors[ci % len(colors)], edgecolor='white', linewidth=0.5)
    ax.set_xlabel('JSD Topic Distance', color='#c9d1d9')
    ax.set_ylabel('Count', color='#c9d1d9')
    ax.set_title('Topic Distance Distribution per Domain', color='#e6edf3')
    ax.legend(fontsize=9)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    # Right: top words per topic
    ax = axes[1]
    ax.set_facecolor('#0d1117')
    ax.axis('off')
    n_topics = tm['n_topics']
    for i in range(n_topics):
        words = ', '.join(tm['topic_words'][str(i)][:6])
        y = 1.0 - (i + 0.5) / n_topics
        ax.text(0.05, y, f'Topic {i}:', fontsize=10, color='#4FC3F7',
                fontweight='bold', transform=ax.transAxes, va='center')
        ax.text(0.2, y, words, fontsize=9, color='#c9d1d9',
                transform=ax.transAxes, va='center')
    ax.set_title('LDA Topics — Top Words', color='#e6edf3')

    plt.tight_layout()
    outpath = OUTPUT_DIR / 'm26_topic_overview.png'
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


def main():
    tm, slugs, slug_idx = load_topic_model()
    n_topics = tm['n_topics']

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    print(f'\nTopic model: {n_topics} topics, {len(slugs)} articles')
    print('\nCross-subject median JSD per domain:')
    domain_medians = compute_domain_medians_topic(pid_trials, tm)

    plot_topic_overview(tm, pid_trials, pids)

    pid_data = build_sequences_topic(pids, pid_trials, tm, domain_medians)

    # Reuse M20 grid plot — just change title context
    plot_trial_grid_with_switches(
        pid_data, pids, 0, 100.0, n_topics,
        'm26_trial1.png')
    plot_trial_grid_with_switches(
        pid_data, pids, 1, 100.0, n_topics,
        'm26_trial2.png')

    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M26 {t_name} (Topic Modeling {n_topics} topics) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run and verify output**

Run: `cd scripts && python m26_topic_modeling.py`

Expected:
- Prints domain medians, exploit rates
- Creates `output/m26_topic_overview.png`, `output/m26_trial1.png`, `output/m26_trial2.png`

```bash
ls -la ../output/m26_*.png
```

- [ ] **Step 3: Commit**

```bash
git add scripts/m26_topic_modeling.py output/m26_*.png
git commit -m "feat(1a): add topic modeling visualization — M26 with JSD distances"
```

---

### Task 4: Combined LSA + Time Median (4.א = M27)

**Files:**
- Create: `scripts/m27_lsa_time_median.py`
- Output: `output/m27_trial1.png`, `output/m27_trial2.png`

This is a variation of M20 where **pages** are classified by time median (like M15)
instead of typing/paste (like M18).

- [ ] **Step 1: Create m27_lsa_time_median.py**

```python
#!/usr/bin/env python3
"""
M27: Explore/Exploit — LSA Transitions + Time-Median Pages (4.א)
================================================================
Pages: Exploit if page duration >= per-trial median (M15 logic).
Transitions: Exploit if cosine distance <= cross-subject domain median (M20 logic).
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')

from m13_combined_binary import cos_dist
from m16_combined_lsa_median import build_lsa
from m20_cross_subject_median import (
    compute_domain_medians, count_switches, plot_trial_grid_with_switches,
)
from helpers import load_trials, get_pids_and_trials


def build_sequences_time_median(pids, pid_trials, slug_idx, pc, domain_medians):
    pid_data = {}

    for pid in pids:
        pid_data[pid] = []
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            if len(pvs) < 2:
                continue

            domain = tr['domain']
            dist_threshold = domain_medians[domain]

            page_times = [pv['duration'] for pv in pvs]
            time_median = np.median(page_times)

            trans_dists = []
            for i in range(1, len(pvs)):
                fi = slug_idx.get(pvs[i - 1]['title'])
                ti = slug_idx.get(pvs[i]['title'])
                if fi is not None and ti is not None:
                    trans_dists.append(cos_dist(pc[fi], pc[ti]))
                else:
                    trans_dists.append(np.nan)

            points = []
            for i in range(len(pvs)):
                is_exploit = page_times[i] >= time_median
                points.append({
                    'x': i + 1,
                    'y': 0.5 if is_exploit else -0.5,
                    'type': 'page',
                    'raw': page_times[i],
                    'threshold': time_median,
                })

                if i < len(pvs) - 1:
                    d = trans_dists[i]
                    is_exploit_d = d <= dist_threshold if not np.isnan(d) else False
                    points.append({
                        'x': i + 1.5,
                        'y': 0.5 if is_exploit_d else -0.5,
                        'type': 'transition',
                        'raw': d,
                        'threshold': dist_threshold,
                    })

            pid_data[pid].append({
                'trial': tr['trial'],
                'condition': tr['condition'],
                'points': points,
                'time_median': time_median,
                'dist_threshold': dist_threshold,
                'domain': domain,
            })

    return pid_data


def main():
    slugs, slug_idx, lsa, var_explained, n_components = build_lsa()

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    print('\nCross-subject median cosine distance per domain:')
    domain_medians = compute_domain_medians(pid_trials, slug_idx, lsa)

    pid_data = build_sequences_time_median(
        pids, pid_trials, slug_idx, lsa, domain_medians)

    plot_trial_grid_with_switches(
        pid_data, pids, 0, var_explained, n_components, 'm27_trial1.png')
    plot_trial_grid_with_switches(
        pid_data, pids, 1, var_explained, n_components, 'm27_trial2.png')

    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M27 {t_name} (LSA {n_components}D, time-median pages) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run and verify output**

Run: `cd scripts && python m27_lsa_time_median.py`

Expected:
- Prints domain medians, exploit rates
- Creates `output/m27_trial1.png`, `output/m27_trial2.png`

```bash
ls -la ../output/m27_*.png
```

- [ ] **Step 3: Commit**

```bash
git add scripts/m27_lsa_time_median.py output/m27_*.png
git commit -m "feat(4a): add LSA + time-median combined measure — M27"
```

---

### Task 5: Combined LSA + 60s Threshold (4.ב = M28)

**Files:**
- Create: `scripts/m28_lsa_60s.py`
- Output: `output/m28_trial1.png`, `output/m28_trial2.png`

- [ ] **Step 1: Create m28_lsa_60s.py**

```python
#!/usr/bin/env python3
"""
M28: Explore/Exploit — LSA Transitions + 60s Pages (4.ב)
========================================================
Pages: Exploit if page duration >= 60 seconds (M2 logic).
Transitions: Exploit if cosine distance <= cross-subject domain median (M20 logic).
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')

from m13_combined_binary import cos_dist
from m16_combined_lsa_median import build_lsa
from m20_cross_subject_median import (
    compute_domain_medians, count_switches, plot_trial_grid_with_switches,
)
from helpers import load_trials, get_pids_and_trials

TIME_THRESHOLD = 60


def build_sequences_60s(pids, pid_trials, slug_idx, pc, domain_medians):
    pid_data = {}

    for pid in pids:
        pid_data[pid] = []
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            if len(pvs) < 2:
                continue

            domain = tr['domain']
            dist_threshold = domain_medians[domain]

            trans_dists = []
            for i in range(1, len(pvs)):
                fi = slug_idx.get(pvs[i - 1]['title'])
                ti = slug_idx.get(pvs[i]['title'])
                if fi is not None and ti is not None:
                    trans_dists.append(cos_dist(pc[fi], pc[ti]))
                else:
                    trans_dists.append(np.nan)

            points = []
            for i in range(len(pvs)):
                is_exploit = pvs[i]['duration'] >= TIME_THRESHOLD
                points.append({
                    'x': i + 1,
                    'y': 0.5 if is_exploit else -0.5,
                    'type': 'page',
                    'raw': pvs[i]['duration'],
                    'threshold': TIME_THRESHOLD,
                })

                if i < len(pvs) - 1:
                    d = trans_dists[i]
                    is_exploit_d = d <= dist_threshold if not np.isnan(d) else False
                    points.append({
                        'x': i + 1.5,
                        'y': 0.5 if is_exploit_d else -0.5,
                        'type': 'transition',
                        'raw': d,
                        'threshold': dist_threshold,
                    })

            pid_data[pid].append({
                'trial': tr['trial'],
                'condition': tr['condition'],
                'points': points,
                'dist_threshold': dist_threshold,
                'domain': domain,
            })

    return pid_data


def main():
    slugs, slug_idx, lsa, var_explained, n_components = build_lsa()

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    print('\nCross-subject median cosine distance per domain:')
    domain_medians = compute_domain_medians(pid_trials, slug_idx, lsa)

    pid_data = build_sequences_60s(
        pids, pid_trials, slug_idx, lsa, domain_medians)

    plot_trial_grid_with_switches(
        pid_data, pids, 0, var_explained, n_components, 'm28_trial1.png')
    plot_trial_grid_with_switches(
        pid_data, pids, 1, var_explained, n_components, 'm28_trial2.png')

    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M28 {t_name} (LSA {n_components}D, 60s pages) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run and verify output**

Run: `cd scripts && python m28_lsa_60s.py`

Expected:
- Prints domain medians, exploit rates
- Creates `output/m28_trial1.png`, `output/m28_trial2.png`

```bash
ls -la ../output/m28_*.png
```

- [ ] **Step 3: Commit**

```bash
git add scripts/m28_lsa_60s.py output/m28_*.png
git commit -m "feat(4b): add LSA + 60s combined measure — M28"
```

---

### Task 6: PCA on Raw Continuous Signals (M29)

**Files:**
- Create: `scripts/m29_pca_raw.py`
- Output: `output/m29_pca_scree.png`, `output/m29_pca_biplot.png`, `output/m29_pca_scores.png`

- [ ] **Step 1: Create m29_pca_raw.py**

```python
#!/usr/bin/env python3
"""
M29: PCA on Raw Continuous Signals
===================================
Runs PCA on 3 continuous features per page visit:
  1. Time on page (seconds)
  2. Topic distance from previous page (JSD)
  3. Writing amount (typing duration + paste events)

Tests whether Explore/Exploit is a single dimension.
Output: scree plot, biplot, PC1 scores per participant.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

DATA_DIR = Path(__file__).parent / '..' / 'data'
TOPIC_PATH = DATA_DIR / 'topic_model.json'

FEATURE_NAMES = ['Time on page (s)', 'Topic distance (JSD)', 'Writing amount (s)']
PASTE_WEIGHT = 5.0


def load_topic_distances():
    with open(TOPIC_PATH, 'r', encoding='utf-8') as f:
        tm = json.load(f)
    return tm


def get_topic_dist(tm, slug_a, slug_b):
    key1 = f'{slug_a}|||{slug_b}'
    key2 = f'{slug_b}|||{slug_a}'
    if key1 in tm['distances']:
        return tm['distances'][key1]
    if key2 in tm['distances']:
        return tm['distances'][key2]
    return np.nan


def compute_writing_amount(pv, typing_intervals, paste_times):
    """Continuous writing measure: typing duration + weighted paste events."""
    typing_dur = 0.0
    for bs, be in typing_intervals:
        start = max(bs, pv['start'])
        end = min(be, pv['end'])
        if end > start:
            typing_dur += end - start

    n_pastes = sum(1 for pt in paste_times if pv['start'] <= pt <= pv['end'])

    return typing_dur + n_pastes * PASTE_WEIGHT


def build_feature_matrix(pids, pid_trials, tm):
    """Build (N, 3) feature matrix: time, topic_distance, writing_amount."""
    rows = []
    metadata = []

    for pid in pids:
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            for i in range(len(pvs)):
                time_on_page = pvs[i]['duration']

                if i > 0:
                    topic_d = get_topic_dist(tm, pvs[i - 1]['title'], pvs[i]['title'])
                else:
                    topic_d = np.nan

                writing = compute_writing_amount(
                    pvs[i], tr['typing_intervals'], tr['paste_times'])

                if not np.isnan(topic_d):
                    rows.append([time_on_page, topic_d, writing])
                    metadata.append({
                        'pid': pid,
                        'trial': tr['trial'],
                        'domain': tr['domain'],
                        'page_idx': i,
                        'title': pvs[i]['title'],
                    })

    return np.array(rows), metadata


def plot_scree(pca, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    var = pca.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var)
    x = np.arange(1, len(var) + 1)

    ax.bar(x, var, color='#4FC3F7', edgecolor='white', linewidth=0.5, label='Individual')
    ax.plot(x, cumvar, 'o-', color='#FF9800', linewidth=2, markersize=8, label='Cumulative')

    for i, (v, cv) in enumerate(zip(var, cumvar)):
        ax.text(x[i], v + 1.5, f'{v:.1f}%', ha='center', fontsize=11,
                color='#e6edf3', fontweight='bold')
        ax.text(x[i] + 0.15, cv + 2, f'{cv:.1f}%', ha='left', fontsize=9,
                color='#FF9800')

    ax.set_xlabel('Principal Component', color='#c9d1d9', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', color='#c9d1d9', fontsize=12)
    ax.set_title('M29: PCA Scree Plot — Raw Continuous Signals',
                 color='#e6edf3', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i}' for i in x])
    ax.tick_params(colors='#8b949e')
    ax.legend(fontsize=10, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


def plot_biplot(pca, X_scaled, metadata, outpath):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    scores = pca.transform(X_scaled)

    unique_pids = sorted(set(m['pid'] for m in metadata))
    cmap = plt.cm.tab20
    pid_colors = {p: cmap(i / len(unique_pids)) for i, p in enumerate(unique_pids)}

    for i, m in enumerate(metadata):
        ax.scatter(scores[i, 0], scores[i, 1], c=[pid_colors[m['pid']]],
                   s=20, alpha=0.5, zorder=2)

    loadings = pca.components_.T
    var = pca.explained_variance_ratio_
    scale = max(np.abs(scores[:, :2]).max(), 1) * 0.8

    for i, name in enumerate(FEATURE_NAMES):
        ax.annotate(
            '', xy=(loadings[i, 0] * scale, loadings[i, 1] * scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2.5))
        ax.text(loadings[i, 0] * scale * 1.12, loadings[i, 1] * scale * 1.12,
                name, fontsize=11, color='#FF9800', fontweight='bold',
                ha='center', va='center')

    ax.axhline(0, color='#30363d', linewidth=0.5)
    ax.axvline(0, color='#30363d', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', color='#c9d1d9', fontsize=12)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', color='#c9d1d9', fontsize=12)
    ax.set_title('M29: PCA Biplot — Feature Loadings & Page Visits',
                 color='#e6edf3', fontsize=14, fontweight='bold')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


def plot_pc1_scores(pca, X_scaled, metadata, pids, outpath):
    """Per-participant distribution of PC1 scores."""
    scores = pca.transform(X_scaled)
    pc1 = scores[:, 0]

    pid_scores = {p: [] for p in pids}
    for i, m in enumerate(metadata):
        pid_scores[m['pid']].append(pc1[i])

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    positions = []
    labels = []
    for pi, pid in enumerate(pids):
        sc = pid_scores[pid]
        if not sc:
            continue
        bp = ax.boxplot([sc], positions=[pi], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='#4FC3F7', alpha=0.6),
                        medianprops=dict(color='#FF9800', linewidth=2),
                        whiskerprops=dict(color='#8b949e'),
                        capprops=dict(color='#8b949e'),
                        flierprops=dict(markeredgecolor='#8b949e', markersize=4))
        positions.append(pi)
        labels.append(f'P{pid}')

    ax.axhline(0, color='#FF9800', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, color='#c9d1d9')
    ax.set_ylabel('PC1 Score (Exploit ← → Explore)', color='#c9d1d9', fontsize=12)
    ax.set_title('M29: PC1 Score Distribution per Participant',
                 color='#e6edf3', fontsize=14, fontweight='bold')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


def main():
    tm = load_topic_distances()
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    print('Building feature matrix...')
    X, metadata = build_feature_matrix(pids, pid_trials, tm)
    print(f'  {X.shape[0]} page visits × {X.shape[1]} features')

    print('\nRaw feature stats:')
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        print(f'  {name}: mean={col.mean():.2f}, median={np.median(col):.2f}, '
              f'std={col.std():.2f}')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    pca.fit(X_scaled)

    print('\nPCA Results:')
    print(f'  Explained variance: {pca.explained_variance_ratio_ * 100}')
    print(f'  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%')
    print(f'  Cumulative: {np.cumsum(pca.explained_variance_ratio_)*100}')

    print('\nLoadings (feature weights per PC):')
    for i, name in enumerate(FEATURE_NAMES):
        loads = [pca.components_[j, i] for j in range(3)]
        print(f'  {name}: PC1={loads[0]:+.3f}, PC2={loads[1]:+.3f}, PC3={loads[2]:+.3f}')

    plot_scree(pca, OUTPUT_DIR / 'm29_pca_scree.png')
    plot_biplot(pca, X_scaled, metadata, OUTPUT_DIR / 'm29_pca_biplot.png')
    plot_pc1_scores(pca, X_scaled, metadata, pids, OUTPUT_DIR / 'm29_pca_scores.png')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run and verify output**

Run: `cd scripts && python m29_pca_raw.py`

Expected:
- Prints feature stats, PCA variance explained, loadings
- Creates 3 PNGs in output/

```bash
ls -la ../output/m29_*.png
```

- [ ] **Step 3: Commit**

```bash
git add scripts/m29_pca_raw.py output/m29_*.png
git commit -m "feat(pca): add PCA on raw continuous signals — M29 scree/biplot/scores"
```

---

### Task 7: Update professors_plan.md Status Table

**Files:**
- Modify: `docs/professors_plan.md`

- [ ] **Step 1: Update status table to reflect all implementations**

```markdown
## סיכום סטטוס

| # | ניתוח | תרשים | סטטוס |
|---|-------|-------|-------|
| PCA | PCA על 3 אותות גולמיים | M29 | ✅ קיים |
| 1.א | Topic Modeling | M26 | ✅ קיים |
| 1.ב | LSA סמנטי | M16 | ✅ קיים |
| 2.א | Median פר נבדק (פר trial) | M15 | ✅ קיים |
| 2.ב | סף 60 שניות | M2 | ✅ קיים |
| 3.א | Type vs Paste | M4 | ✅ קיים |
| 3.ב | בינארי Type/Paste | M18 | ✅ קיים |
| 4.א | סמנטיקה + Median זמן | M27 | ✅ קיים |
| 4.ב | סמנטיקה + 60s | M28 | ✅ קיים |
| 4.ג | סמנטיקה + Type/Paste | M20 | ✅ קיים |
```

- [ ] **Step 2: Commit**

```bash
git add docs/professors_plan.md
git commit -m "docs: update status table — all analyses implemented"
```

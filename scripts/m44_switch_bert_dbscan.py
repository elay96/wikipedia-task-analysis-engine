#!/usr/bin/env python3
"""
M44: Switch Rate - BERT Embeddings + DBSCAN (eps=0.5, 7 clusters)
=================================================================
Per page: assign cluster from BERT DBSCAN model (M42).
Transition = cluster_id differs between consecutive pages.
Switch rate = transitions / (N-1).
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
MODEL_PATH = DATA_DIR / 'cleaned' / 'bertopic_dbscan.json'

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


def get_page_topics(page_visits, topic_assignments):
    topics = []
    for pv in page_visits:
        topic = topic_assignments.get(pv['title'], -1)
        topics.append(topic)
    return topics


def plot_domain(domain, participants_data, output_path):
    n = len(participants_data)
    cols = 4
    rows = max(1, int(np.ceil(n / cols)))

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.2))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f'M44: Switch Rate (BERT DBSCAN, eps=0.5) - {domain}',
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
        ax.set_xticks(x)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.tick_params(colors=MUTED_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)

        unique_topics = sorted(set(topics))
        for t in unique_topics:
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[M44] Switch Rate - BERT DBSCAN (eps=0.5)")

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    print(f"  {len(trials)} trials from {len(pids)} participants")

    with open(MODEL_PATH, 'r') as f:
        model = json.load(f)
    topic_assignments = model['topic_assignments']
    print(f"  {len(topic_assignments)} articles, {model['n_clusters']} clusters")

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
        plot_domain(domain, domain_data[domain], OUTPUT_DIR / f'm44_switch_bert_dbscan_{domain}.png')

    df_out = pd.DataFrame(rows_csv)
    csv_path = OUTPUT_DIR / 'm44_switch_bert_dbscan.csv'
    df_out.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    print(f"\nSwitch rate stats by domain:")
    for domain in sorted(domain_data.keys()):
        rates = [r[2] for r in domain_data[domain] if not np.isnan(r[2])]
        print(f"  {domain}: mean={np.mean(rates):.3f}, std={np.std(rates):.3f}, n={len(rates)}")


if __name__ == '__main__':
    main()

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

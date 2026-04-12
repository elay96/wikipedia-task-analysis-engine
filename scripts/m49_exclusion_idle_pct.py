#!/usr/bin/env python3
"""
M49: Exclusion Criterion - Idle % After Last Meaningful Event
==============================================================
For each participant-question pair, computes the percentage of task
duration that elapsed after the last meaningful event (page visit,
search, or writing activity). Proposed threshold: 40%.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'
OUTPUT_DIR = SCRIPT_DIR / '..' / 'output'

THRESHOLD = 40.0

MEANINGFUL_ACTIONS = [
    'article_open', 'search', 'link_click', 'back_navigation', 'paste',
]
SNAPSHOT_ACTIONS = ['answer_snapshot', 'answer_snapshot_cursor_leave']


def compute_idle_pct(df):
    """Compute idle % after last meaningful event per participant-question."""
    df = df.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    real = df[df['Domain'].isin(['economics', 'psychology'])]

    results = []
    for (uid, domain), g in real.groupby(['ID', 'Domain']):
        g = g.sort_values('Time')

        task_starts = g[g['Action'] == 'task_start']
        task_ends = g[g['Action'] == 'task_end']
        if len(task_starts) == 0 or len(task_ends) == 0:
            continue

        t_start = task_starts['Time'].iloc[0]
        t_end = task_ends['Time'].iloc[0]
        total_sec = (t_end - t_start).total_seconds()
        if total_sec <= 0:
            continue

        meaningful = g[g['Action'].isin(MEANINGFUL_ACTIONS)]

        snapshots = g[g['Action'].isin(SNAPSHOT_ACTIONS)].copy()
        snapshots['prev_len'] = snapshots['AnswerLength'].shift(1)
        writing_snapshots = snapshots[
            snapshots['AnswerLength'] != snapshots['prev_len']
        ]

        all_meaningful = pd.concat(
            [meaningful, writing_snapshots]
        ).sort_values('Time')
        all_meaningful = all_meaningful[
            (all_meaningful['Time'] >= t_start)
            & (all_meaningful['Time'] <= t_end)
        ]

        if len(all_meaningful) == 0:
            idle_pct = 100.0
        else:
            last_event = all_meaningful['Time'].iloc[-1]
            idle_after = (t_end - last_event).total_seconds()
            idle_pct = (idle_after / total_sec) * 100

        results.append({
            'ID': uid,
            'Domain': domain,
            'total_min': total_sec / 60,
            'idle_pct': idle_pct,
        })

    return pd.DataFrame(results)


def plot_idle_pct(res, output_path):
    """Horizontal bar chart of idle % with threshold line and stats."""
    res = res.sort_values('idle_pct', ascending=True).reset_index(drop=True)

    colors = []
    for _, r in res.iterrows():
        if r['idle_pct'] >= THRESHOLD:
            colors.append('#e74c3c')
        elif r['idle_pct'] >= 30:
            colors.append('#f39c12')
        else:
            colors.append('#2ecc71')

    labels = [
        f"P{int(r['ID'])} ({r['Domain'][:4]})" for _, r in res.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.barh(
        range(len(res)), res['idle_pct'],
        color=colors, edgecolor='white', linewidth=0.5, height=0.7,
    )

    ax.axvline(x=THRESHOLD, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(
        THRESHOLD + 1, len(res) - 1,
        f'Proposed threshold ({THRESHOLD:.0f}%)',
        color='#e74c3c', fontsize=10, fontweight='bold', va='top',
    )

    mean_val = res['idle_pct'].mean()
    median_val = res['idle_pct'].median()

    ax.axvline(x=mean_val, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.axvline(x=median_val, color='#9b59b6', linestyle=':', linewidth=1.5, alpha=0.8)

    ax.text(
        mean_val + 0.8, -0.8,
        f'Mean = {mean_val:.1f}%',
        color='#3498db', fontsize=9, fontweight='bold',
    )
    ax.text(
        median_val + 0.8, -1.6,
        f'Median = {median_val:.1f}%',
        color='#9b59b6', fontsize=9, fontweight='bold',
    )

    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(
        '% of task duration idle after last meaningful event',
        fontsize=11, fontweight='bold',
    )
    ax.set_title(
        'Idle Time After Last Meaningful Event - Exclusion Criterion Analysis\n'
        f'(Pilot data, N={len(res)} participant-question pairs)',
        fontsize=13, fontweight='bold', pad=15,
    )

    ax.set_xlim(0, 80)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='Below threshold'),
        mpatches.Patch(facecolor='#f39c12', label='Borderline (30-40%)'),
        mpatches.Patch(facecolor='#e74c3c', label=f'Excluded (>= {THRESHOLD:.0f}%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    for i, (_, r) in enumerate(res.iterrows()):
        if r['idle_pct'] >= THRESHOLD:
            ax.text(
                r['idle_pct'] + 1, i, f"{r['idle_pct']:.1f}%",
                va='center', fontsize=9, fontweight='bold', color='#e74c3c',
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    df = pd.read_csv(DATA_DIR / 'Game.csv')
    res = compute_idle_pct(df)

    res_sorted = res.sort_values('idle_pct', ascending=False)
    print('=== Idle % after last meaningful event ===')
    print(f"{'ID':>4s}  {'Domain':>12s}  {'Total(min)':>10s}  {'Idle%':>6s}")
    print('-' * 40)
    for _, r in res_sorted.iterrows():
        flag = ' ***' if r['idle_pct'] >= THRESHOLD else ''
        print(
            f"{int(r['ID']):>4d}  {r['Domain']:>12s}"
            f"  {r['total_min']:>10.1f}  {r['idle_pct']:>5.1f}%{flag}"
        )

    print()
    print(f"Mean idle%:  {res['idle_pct'].mean():.1f}%")
    print(f"Median idle%: {res['idle_pct'].median():.1f}%")
    excluded = len(res[res['idle_pct'] >= THRESHOLD])
    print(f"Excluded (>= {THRESHOLD:.0f}%): {excluded} / {len(res)}")

    res.to_csv(OUTPUT_DIR / 'm49_idle_pct.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'm49_idle_pct.csv'}")

    plot_idle_pct(res, OUTPUT_DIR / 'm49_idle_pct.png')


if __name__ == '__main__':
    main()

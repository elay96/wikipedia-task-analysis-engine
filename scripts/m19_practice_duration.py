#!/usr/bin/env python3
"""
M19: Practice Round — Task Duration Distribution
=================================================
Output: m19_practice_duration.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

DATA_PATH = Path(__file__).parent / '..' / 'data' / 'cleaned' / 'Game.csv'
OUTPUT_DIR = Path(__file__).parent / '..' / 'output'


def main():
    df = pd.read_csv(DATA_PATH)
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    practice = df[df['IsPractice'] == 1]

    durations = []
    pids_list = []
    for pid in sorted(practice['ID'].unique()):
        p = practice[practice['ID'] == pid].sort_values('Time')
        starts = p[p['Action'] == 'task_start']['Time']
        ends = p[p['Action'] == 'task_end']['Time']
        if len(starts) > 0 and len(ends) > 0:
            d = (ends.iloc[0] - starts.iloc[0]).total_seconds()
            durations.append(d)
            pids_list.append(pid)

    vals = np.array(durations)
    mins = vals / 60

    mean_v = np.mean(vals)
    median_v = np.median(vals)
    std_v = np.std(vals, ddof=1)
    se_v = std_v / np.sqrt(len(vals))
    skew_v = sp_stats.skew(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    iqr_v = q3 - q1

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    TEXT_COLOR = '#1a1a2e'
    LABEL_COLOR = '#333333'
    GRID_COLOR = '#e0e0e0'
    SPINE_COLOR = '#cccccc'
    BG_COLOR = '#fafafa'

    BLUE = '#1976D2'
    RED = '#D32F2F'
    ORANGE = '#F57C00'

    # --- Top left: histogram ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG_COLOR)
    bins = np.arange(0, max(vals) + 60, 60)
    ax1.hist(mins, bins=bins / 60, color=BLUE, edgecolor='white', alpha=0.8, zorder=3)
    ax1.axvline(mean_v / 60, color=RED, linewidth=2, linestyle='--',
                label=f'Mean: {mean_v/60:.1f} min', zorder=5)
    ax1.axvline(median_v / 60, color=ORANGE, linewidth=2, linestyle='-',
                label=f'Median: {median_v/60:.1f} min', zorder=5)
    ax1.axvspan((mean_v - std_v) / 60, (mean_v + std_v) / 60,
                alpha=0.08, color=RED, zorder=2, label=f'SD: {std_v/60:.1f} min')
    ax1.set_xlabel('Duration (minutes)', fontsize=11, color=LABEL_COLOR)
    ax1.set_ylabel('Count', fontsize=11, color=LABEL_COLOR)
    ax1.set_title('Distribution of Task Duration', fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax1.legend(fontsize=9, facecolor='white', edgecolor=SPINE_COLOR, labelcolor=LABEL_COLOR)
    ax1.tick_params(colors=LABEL_COLOR)
    for spine in ax1.spines.values():
        spine.set_color(SPINE_COLOR)
    ax1.grid(axis='y', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # --- Top right: individual bars sorted ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG_COLOR)
    sort_idx = np.argsort(vals)
    sorted_pids = [pids_list[i] for i in sort_idx]
    sorted_vals = vals[sort_idx] / 60
    colors = [RED if v * 60 > mean_v + std_v else
              ORANGE if v * 60 > mean_v else BLUE for v in sorted_vals]
    ax2.barh(range(len(sorted_pids)), sorted_vals, color=colors,
             edgecolor='white', height=0.7, zorder=3)
    ax2.axvline(mean_v / 60, color=RED, linewidth=1.5, linestyle='--', alpha=0.8)
    ax2.axvline(median_v / 60, color=ORANGE, linewidth=1.5, linestyle='-', alpha=0.8)
    ax2.set_yticks(range(len(sorted_pids)))
    ax2.set_yticklabels([f'P{p}' for p in sorted_pids], fontsize=8, color=LABEL_COLOR)
    ax2.set_xlabel('Duration (minutes)', fontsize=11, color=LABEL_COLOR)
    ax2.set_title('Per Participant (sorted)', fontsize=13, color=TEXT_COLOR, fontweight='bold')
    for i, v in enumerate(sorted_vals):
        ax2.text(v + 0.2, i, f'{v:.1f}m', va='center', fontsize=7, color='#555555')
    ax2.tick_params(colors=LABEL_COLOR)
    for spine in ax2.spines.values():
        spine.set_color(SPINE_COLOR)
    ax2.grid(axis='x', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # --- Bottom left: box + strip plot ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG_COLOR)
    ax3.boxplot(mins, vert=False, widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=BLUE, alpha=0.2, edgecolor=BLUE),
                medianprops=dict(color=ORANGE, linewidth=2),
                whiskerprops=dict(color='#666666'),
                capprops=dict(color='#666666'),
                flierprops=dict(marker='o', markerfacecolor=RED,
                                markersize=8, markeredgecolor='white'))
    np.random.seed(42)
    jitter = np.random.normal(1, 0.06, len(mins))
    ax3.scatter(mins, jitter, color=BLUE, alpha=0.7, s=40,
                edgecolors='white', linewidths=0.5, zorder=5)
    ax3.set_xlabel('Duration (minutes)', fontsize=11, color=LABEL_COLOR)
    ax3.set_title('Box Plot + Individual Points', fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax3.set_yticks([])
    ax3.tick_params(colors=LABEL_COLOR)
    for spine in ax3.spines.values():
        spine.set_color(SPINE_COLOR)
    ax3.grid(axis='x', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # --- Bottom right: summary statistics table ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('white')
    ax4.axis('off')

    stats_data = [
        ['N', f'{len(vals)}'],
        ['Mean', f'{mean_v:.1f}s  ({mean_v/60:.1f} min)'],
        ['Median', f'{median_v:.1f}s  ({median_v/60:.1f} min)'],
        ['SD', f'{std_v:.1f}s  ({std_v/60:.1f} min)'],
        ['SE', f'{se_v:.1f}s  ({se_v/60:.1f} min)'],
        ['Min', f'{vals.min():.1f}s  ({vals.min()/60:.1f} min)'],
        ['Max', f'{vals.max():.1f}s  ({vals.max()/60:.1f} min)'],
        ['Q1 (25%)', f'{q1:.1f}s  ({q1/60:.1f} min)'],
        ['Q3 (75%)', f'{q3:.1f}s  ({q3/60:.1f} min)'],
        ['IQR', f'{iqr_v:.1f}s  ({iqr_v/60:.1f} min)'],
        ['Skewness', f'{skew_v:+.2f}'],
    ]

    table = ax4.table(cellText=stats_data, colLabels=['Statistic', 'Value'],
                      loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE_COLOR)
        if row == 0:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(color=TEXT_COLOR, fontweight='bold')
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color=LABEL_COLOR)

    ax4.set_title('Descriptive Statistics', fontsize=13, color=TEXT_COLOR,
                  fontweight='bold', pad=20)

    fig.suptitle('M19: Practice Round — Task Duration',
                 fontsize=16, color=TEXT_COLOR, fontweight='bold', y=0.98)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm19_practice_duration.png'
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
M21: Explore/Exploit Tendency Distribution — All Trials, All Users
======================================================
Output: m21_tendency_dist.png
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

from m13_combined_binary import cos_dist
from m16_combined_lsa_median import build_lsa
from m18_typing_binary import page_had_typing_or_paste
from m20_cross_subject_median import compute_domain_medians, build_sequences, count_switches
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR


def main():
    slugs, slug_idx, lsa, var_explained, n_components = build_lsa()
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    domain_medians = compute_domain_medians(pid_trials, slug_idx, lsa)
    pid_data = build_sequences(pids, pid_trials, slug_idx, lsa, domain_medians)

    # Collect SR per trial per user
    sr_values = []
    labels = []
    for pid in pids:
        for td in pid_data[pid]:
            pts = td['points']
            if len(pts) < 2:
                continue
            sw = count_switches(pts)
            sr = sw / (len(pts) - 1) * 100
            sr_values.append(sr)
            labels.append(f'P{pid}-T{td["trial"]}')

    vals = np.array(sr_values)

    mean_v = np.mean(vals)
    median_v = np.median(vals)
    std_v = np.std(vals, ddof=1)
    se_v = std_v / np.sqrt(len(vals))
    skew_v = sp_stats.skew(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    iqr_v = q3 - q1

    TEXT_COLOR = '#1a1a2e'
    LABEL_COLOR = '#333333'
    GRID_COLOR = '#e0e0e0'
    SPINE_COLOR = '#cccccc'
    BG_COLOR = '#fafafa'
    BLUE = '#1976D2'
    RED = '#D32F2F'
    ORANGE = '#F57C00'

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # --- Top left: histogram ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG_COLOR)
    bins = np.arange(0, 105, 5)
    ax1.hist(vals, bins=bins, color=BLUE, edgecolor='white', alpha=0.8, zorder=3)
    ax1.axvline(mean_v, color=RED, linewidth=2, linestyle='--',
                label=f'Mean: {mean_v:.1f}%', zorder=5)
    ax1.axvline(median_v, color=ORANGE, linewidth=2, linestyle='-',
                label=f'Median: {median_v:.1f}%', zorder=5)
    ax1.axvspan(mean_v - std_v, mean_v + std_v,
                alpha=0.08, color=RED, zorder=2, label=f'SD: {std_v:.1f}%')
    ax1.set_xlabel('Explore/Exploit Tendency (%)', fontsize=11, color=LABEL_COLOR)
    ax1.set_ylabel('Count', fontsize=11, color=LABEL_COLOR)
    ax1.set_title('Distribution of Explore/Exploit Tendency', fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax1.legend(fontsize=9, facecolor='white', edgecolor=SPINE_COLOR, labelcolor=LABEL_COLOR)
    ax1.tick_params(colors=LABEL_COLOR)
    for spine in ax1.spines.values():
        spine.set_color(SPINE_COLOR)
    ax1.grid(axis='y', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # --- Top right: individual bars sorted ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG_COLOR)
    sort_idx = np.argsort(vals)
    sorted_labels = [labels[i] for i in sort_idx]
    sorted_vals = vals[sort_idx]
    colors = [RED if v > mean_v + std_v else
              ORANGE if v > mean_v else BLUE for v in sorted_vals]
    ax2.barh(range(len(sorted_labels)), sorted_vals, color=colors,
             edgecolor='white', height=0.7, zorder=3)
    ax2.axvline(mean_v, color=RED, linewidth=1.5, linestyle='--', alpha=0.8)
    ax2.axvline(median_v, color=ORANGE, linewidth=1.5, linestyle='-', alpha=0.8)
    ax2.set_yticks(range(len(sorted_labels)))
    ax2.set_yticklabels(sorted_labels, fontsize=7, color=LABEL_COLOR)
    ax2.set_xlabel('Explore/Exploit Tendency (%)', fontsize=11, color=LABEL_COLOR)
    ax2.set_title('Per Trial (sorted)', fontsize=13, color=TEXT_COLOR, fontweight='bold')
    for i, v in enumerate(sorted_vals):
        ax2.text(v + 0.8, i, f'{v:.0f}%', va='center', fontsize=6, color='#555555')
    ax2.tick_params(colors=LABEL_COLOR)
    for spine in ax2.spines.values():
        spine.set_color(SPINE_COLOR)
    ax2.grid(axis='x', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # --- Bottom left: box + strip plot ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG_COLOR)
    ax3.boxplot(vals, vert=False, widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=BLUE, alpha=0.2, edgecolor=BLUE),
                medianprops=dict(color=ORANGE, linewidth=2),
                whiskerprops=dict(color='#666666'),
                capprops=dict(color='#666666'),
                flierprops=dict(marker='o', markerfacecolor=RED,
                                markersize=8, markeredgecolor='white'))
    np.random.seed(42)
    jitter = np.random.normal(1, 0.06, len(vals))
    ax3.scatter(vals, jitter, color=BLUE, alpha=0.7, s=40,
                edgecolors='white', linewidths=0.5, zorder=5)
    ax3.set_xlabel('Explore/Exploit Tendency (%)', fontsize=11, color=LABEL_COLOR)
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
        ['N (trials)', f'{len(vals)}'],
        ['Mean', f'{mean_v:.1f}%'],
        ['Median', f'{median_v:.1f}%'],
        ['SD', f'{std_v:.1f}%'],
        ['SE', f'{se_v:.1f}%'],
        ['Min', f'{vals.min():.1f}%'],
        ['Max', f'{vals.max():.1f}%'],
        ['Q1 (25%)', f'{q1:.1f}%'],
        ['Q3 (75%)', f'{q3:.1f}%'],
        ['IQR', f'{iqr_v:.1f}%'],
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

    fig.suptitle('M21: Explore/Exploit Tendency Distribution (M20 — all trials)',
                 fontsize=16, color=TEXT_COLOR, fontweight='bold', y=0.98)

    outpath = OUTPUT_DIR / 'm21_tendency_dist.png'
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()

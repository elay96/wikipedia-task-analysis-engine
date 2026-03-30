#!/usr/bin/env python3
"""
M15 Distribution Diagnostics — Trial 1 only.
Histogram + KDE with mean/median lines for page durations and transition distances.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from m13_combined_binary import build_pca, cos_dist
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

N_COMPONENTS = 71


def collect_trial1_data(pids, pid_trials, slug_idx, pc):
    page_times = []
    trans_dists = []

    for pid in pids:
        trials = pid_trials[pid]
        if not trials:
            continue
        tr = trials[0]
        pvs = tr['page_visits']
        if len(pvs) < 2:
            continue

        page_times.extend(pv['duration'] for pv in pvs)

        for i in range(1, len(pvs)):
            fi = slug_idx.get(pvs[i - 1]['title'])
            ti = slug_idx.get(pvs[i]['title'])
            if fi is not None and ti is not None:
                trans_dists.append(cos_dist(pc[fi], pc[ti]))

    return np.array(page_times), np.array(trans_dists)


def plot_distribution(ax, data, title, xlabel):
    mean_val = np.mean(data)
    median_val = np.median(data)
    skew_val = sp_stats.skew(data)

    ax.hist(data, bins=20, density=True, color='#4FC3F7', alpha=0.55,
            edgecolor='#1a1a2e', linewidth=0.8, zorder=2)

    # KDE
    xs = np.linspace(data.min() - data.std() * 0.3, data.max() + data.std() * 0.3, 300)
    kde = sp_stats.gaussian_kde(data)
    ax.plot(xs, kde(xs), color='#e6edf3', linewidth=2, zorder=3)

    # Mean line
    ax.axvline(mean_val, color='#FF5252', linewidth=2.5, linestyle='--', zorder=4,
               label=f'Mean = {mean_val:.2f}')
    # Median line
    ax.axvline(median_val, color='#69F0AE', linewidth=2.5, linestyle='-', zorder=4,
               label=f'Median = {median_val:.2f}')

    # Annotation box
    gap = abs(mean_val - median_val)
    box_text = f'Skewness: {skew_val:+.2f}\nn = {len(data)}\nGap: {gap:.2f}'
    ax.text(0.97, 0.95, box_text, transform=ax.transAxes,
            fontsize=10, color='#e6edf3', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', edgecolor='#30363d'))

    ax.set_title(title, fontsize=13, color='#e6edf3', fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=11, color='#c9d1d9')
    ax.set_ylabel('Density', fontsize=11, color='#c9d1d9')
    ax.legend(fontsize=11, loc='upper left', facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#e6edf3')

    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.grid(axis='y', color='#21262d', linewidth=0.5)


def main():
    slugs, slug_idx, pc, var_explained = build_pca(N_COMPONENTS)
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    page_times, trans_dists = collect_trial1_data(pids, pid_trials, slug_idx, pc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'M15 Distribution Diagnostics — Trial 1\n'
        'Red dashed = Mean  |  Green solid = Median',
        fontsize=14, color='#e6edf3', fontweight='bold', y=1.02)

    plot_distribution(ax1, page_times,
                      'Page Durations', 'Duration (seconds)')
    plot_distribution(ax2, trans_dists,
                      'Transition Distances (PCA 71D)', 'Cosine Distance')

    plt.tight_layout()
    outpath = OUTPUT_DIR / 'm15_diagnostics.png'
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()

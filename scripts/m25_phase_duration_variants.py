#!/usr/bin/env python3
"""
M25: Switch Rate — Phase Duration Variants  *** P15 excluded ***
=================================================================
Compares 6 metric variants for measuring switching behavior:
  1. median(explore) + median(exploit) / 2  [original M24]
  2. median(explore) only
  3. median(exploit) only
  4. mean(explore) + mean(exploit) / 2
  5. mean(explore) only
  6. mean(exploit) only

Output: m25_phase_duration_variants.png
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
from m24_phase_duration import compute_run_lengths
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

EXCLUDED_PIDS = {15}

TEXT_COLOR = '#1a1a2e'
LABEL_COLOR = '#333333'
GRID_COLOR = '#e0e0e0'
SPINE_COLOR = '#cccccc'
BG_COLOR = '#fafafa'
BLUE = '#1976D2'
RED = '#D32F2F'
ORANGE = '#F57C00'
GREEN = '#388E3C'
PURPLE = '#7B1FA2'
TEAL = '#00897B'

VARIANT_COLORS = [BLUE, GREEN, TEAL, RED, ORANGE, PURPLE]
VARIANT_NAMES = [
    'median(expl+expt)/2',
    'median(explore)',
    'median(exploit)',
    'mean(expl+expt)/2',
    'mean(explore)',
    'mean(exploit)',
]
VARIANT_SHORT = ['Med Avg', 'Med Expl', 'Med Expt', 'Mean Avg', 'Mean Expl', 'Mean Expt']


def compute_variants(explore_runs, exploit_runs):
    """Compute all 6 metric variants. Returns list of 6 values (or NaN)."""
    med_explore = np.median(explore_runs) if explore_runs else np.nan
    med_exploit = np.median(exploit_runs) if exploit_runs else np.nan
    mean_explore = np.mean(explore_runs) if explore_runs else np.nan
    mean_exploit = np.mean(exploit_runs) if exploit_runs else np.nan

    medians = [v for v in [med_explore, med_exploit] if not np.isnan(v)]
    means = [v for v in [mean_explore, mean_exploit] if not np.isnan(v)]

    return [
        np.mean(medians) if medians else np.nan,
        med_explore,
        med_exploit,
        np.mean(means) if means else np.nan,
        mean_explore,
        mean_exploit,
    ]


def _style_ax(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


def main():
    print("[M25] Phase Duration Variants (P15 excluded)")

    slugs, slug_idx, lsa, var_explained, n_components = build_lsa()
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    pids = [p for p in pids if p not in EXCLUDED_PIDS]
    pid_trials = {p: t for p, t in pid_trials.items() if p not in EXCLUDED_PIDS}
    print(f'Excluded PIDs: {EXCLUDED_PIDS}')
    print(f'Remaining participants: {len(pids)}')

    print('\nCross-subject median cosine distance per domain:')
    domain_medians = compute_domain_medians(pid_trials, slug_idx, lsa)
    pid_data = build_sequences(pids, pid_trials, slug_idx, lsa, domain_medians)

    # Collect per-trial data for all 6 variants
    trial_records = []
    for pid in pids:
        for td in pid_data[pid]:
            pts = td['points']
            if len(pts) < 2:
                continue
            explore_runs, exploit_runs = compute_run_lengths(pts)
            variants = compute_variants(explore_runs, exploit_runs)
            if all(np.isnan(v) for v in variants):
                continue

            sw = count_switches(pts)
            tendency = sw / (len(pts) - 1) * 100

            trial_records.append({
                'pid': pid,
                'trial': td['trial'],
                'label': f'P{pid}-T{td["trial"]}',
                'variants': variants,
                'tendency': tendency,
            })

    n_trials = len(trial_records)
    variant_arrays = []
    for vi in range(6):
        arr = np.array([r['variants'][vi] for r in trial_records])
        variant_arrays.append(arr)

    tendencies = np.array([r['tendency'] for r in trial_records])

    # Valid data per variant
    valid_data = []
    for vi in range(6):
        mask = ~np.isnan(variant_arrays[vi])
        valid_data.append(variant_arrays[vi][mask])

    # --- Figure: 1 row, 2 cols ---
    fig, (ax_box, ax_stats) = plt.subplots(1, 2, figsize=(18, 8),
                                            gridspec_kw={'width_ratios': [1, 1.3]})
    fig.patch.set_facecolor('white')

    # Panel A: Box plots
    _style_ax(ax_box)
    bp = ax_box.boxplot(valid_data, vert=True, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='#666666'),
                        capprops=dict(color='#666666'),
                        flierprops=dict(marker='o', markersize=5,
                                        markeredgecolor='white'))
    for patch, color in zip(bp['boxes'], VARIANT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_box.set_xticklabels(VARIANT_SHORT, fontsize=9, rotation=30, ha='right')
    ax_box.set_ylabel('Phase Duration (steps)', fontsize=11, color=LABEL_COLOR)
    ax_box.set_title('A. All 6 Variants — Box Plots',
                     fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax_box.grid(axis='y', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # Panel B: Stats table
    ax_stats.set_facecolor('white')
    ax_stats.axis('off')

    header = ['Variant', 'N', 'Mean', 'SD', 'Median', 'Min', 'Max', 'Range',
              'r(tend)', 'p']
    table_data = []
    for vi in range(6):
        d = valid_data[vi]
        if len(d) == 0:
            table_data.append([VARIANT_SHORT[vi]] + ['-'] * 9)
            continue
        mask = ~np.isnan(variant_arrays[vi])
        r_val, p_val = sp_stats.pearsonr(tendencies[mask], variant_arrays[vi][mask]) \
            if mask.sum() > 2 else (np.nan, np.nan)
        table_data.append([
            VARIANT_SHORT[vi],
            f'{len(d)}',
            f'{np.mean(d):.2f}',
            f'{np.std(d, ddof=1):.2f}',
            f'{np.median(d):.2f}',
            f'{np.min(d):.2f}',
            f'{np.max(d):.2f}',
            f'{np.max(d) - np.min(d):.2f}',
            f'{r_val:+.3f}',
            f'{p_val:.3f}',
        ])

    table = ax_stats.table(cellText=table_data, colLabels=header,
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE_COLOR)
        if row == 0:
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(color=TEXT_COLOR, fontweight='bold')
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color=LABEL_COLOR)
    ax_stats.set_title('B. Descriptive Statistics + Tendency Correlation',
                       fontsize=13, color=TEXT_COLOR, fontweight='bold', pad=20)

    fig.suptitle(
        'M25: Switch Rate — Phase Duration Variants  *** P15 excluded ***\n'
        'Comparing median vs. mean, explore-only vs. exploit-only vs. combined',
        fontsize=14, color=TEXT_COLOR, fontweight='bold', y=1.02)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm25_phase_duration_variants.png'
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {outpath}')

    # Print summary
    print(f'\n{"="*60}')
    print(f'M25 Summary — 6 Variants (P15 excluded)')
    print(f'{"="*60}')
    print(f'Trials: {n_trials}, Participants: {len(pids)}')
    print(f'\n{"Variant":<25} {"Mean":>7} {"SD":>7} {"Median":>7} {"Range":>10}')
    print(f'{"-"*56}')
    for vi in range(6):
        d = valid_data[vi]
        if len(d) == 0:
            continue
        rng = f'{np.min(d):.1f}-{np.max(d):.1f}'
        print(f'{VARIANT_NAMES[vi]:<25} {np.mean(d):>7.2f} {np.std(d, ddof=1):>7.2f} '
              f'{np.median(d):>7.2f} {rng:>10}')


if __name__ == '__main__':
    main()

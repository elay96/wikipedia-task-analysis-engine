#!/usr/bin/env python3
"""
M24: Switch Rate — Mean Phase Duration (Explore/Exploit Run Lengths)
====================================================================
Measures how quickly participants switch between explore and exploit
phases, based on the length of consecutive runs in each phase.

Two participants can have the same explore/exploit tendency (50/50)
but very different switching patterns:
  - Fast switcher: short runs, alternates rapidly
  - Slow switcher: long runs, stays in each phase

Metric: average of median explore-run-length and median exploit-run-length
(in number of sequence steps from M20).

Output: m24_phase_duration.png
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
GRAY = '#757575'


def compute_run_lengths(points):
    """Compute lengths of consecutive same-phase runs.

    Returns (explore_runs, exploit_runs) as lists of run lengths.
    """
    if not points:
        return [], []

    explore_runs = []
    exploit_runs = []
    current_phase = points[0]['y']
    current_len = 1

    for p in points[1:]:
        if p['y'] == current_phase:
            current_len += 1
        else:
            if current_phase > 0:
                exploit_runs.append(current_len)
            else:
                explore_runs.append(current_len)
            current_phase = p['y']
            current_len = 1

    if current_phase > 0:
        exploit_runs.append(current_len)
    else:
        explore_runs.append(current_len)

    return explore_runs, exploit_runs


def compute_mean_phase_duration(explore_runs, exploit_runs):
    """Average of median explore-run-length and median exploit-run-length."""
    medians = []
    if explore_runs:
        medians.append(np.median(explore_runs))
    if exploit_runs:
        medians.append(np.median(exploit_runs))
    if not medians:
        return np.nan
    return np.mean(medians)


def _style_ax(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


def main():
    print("[M24] Phase Duration (Switch Rate)")

    slugs, slug_idx, lsa, var_explained, n_components = build_lsa()
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    print('\nCross-subject median cosine distance per domain:')
    domain_medians = compute_domain_medians(pid_trials, slug_idx, lsa)
    pid_data = build_sequences(pids, pid_trials, slug_idx, lsa, domain_medians)

    # Collect per-trial phase durations
    trial_records = []
    for pid in pids:
        for td in pid_data[pid]:
            pts = td['points']
            if len(pts) < 2:
                continue
            explore_runs, exploit_runs = compute_run_lengths(pts)
            mpd = compute_mean_phase_duration(explore_runs, exploit_runs)
            if np.isnan(mpd):
                continue

            sw = count_switches(pts)
            tendency = sw / (len(pts) - 1) * 100

            trial_records.append({
                'pid': pid,
                'trial': td['trial'],
                'label': f'P{pid}-T{td["trial"]}',
                'phase_duration': mpd,
                'explore_runs': explore_runs,
                'exploit_runs': exploit_runs,
                'tendency': tendency,
                'n_points': len(pts),
                'median_explore': np.median(explore_runs) if explore_runs else np.nan,
                'median_exploit': np.median(exploit_runs) if exploit_runs else np.nan,
            })

    vals = np.array([r['phase_duration'] for r in trial_records])
    labels = [r['label'] for r in trial_records]
    tendencies = np.array([r['tendency'] for r in trial_records])

    # Per-participant means
    pid_mpd = {}
    for r in trial_records:
        pid_mpd.setdefault(r['pid'], []).append(r['phase_duration'])
    pid_mean_mpd = {p: np.mean(v) for p, v in pid_mpd.items()}

    mean_v = np.mean(vals)
    median_v = np.median(vals)
    std_v = np.std(vals, ddof=1)
    se_v = std_v / np.sqrt(len(vals))
    skew_v = sp_stats.skew(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    iqr_v = q3 - q1

    # --- Figure ---
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A: Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1)
    max_val = max(vals)
    bins = np.arange(0, max_val + 1, 0.5)
    ax1.hist(vals, bins=bins, color=BLUE, edgecolor='white', alpha=0.8, zorder=3)
    ax1.axvline(mean_v, color=RED, linewidth=2, linestyle='--',
                label=f'Mean: {mean_v:.2f}', zorder=5)
    ax1.axvline(median_v, color=ORANGE, linewidth=2, linestyle='-',
                label=f'Median: {median_v:.2f}', zorder=5)
    ax1.axvspan(mean_v - std_v, mean_v + std_v,
                alpha=0.08, color=RED, zorder=2, label=f'SD: {std_v:.2f}')
    ax1.set_xlabel('Mean Phase Duration (steps)', fontsize=11, color=LABEL_COLOR)
    ax1.set_ylabel('Count', fontsize=11, color=LABEL_COLOR)
    ax1.set_title('A. Distribution of Phase Duration',
                  fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax1.legend(fontsize=9, facecolor='white', edgecolor=SPINE_COLOR, labelcolor=LABEL_COLOR)
    ax1.grid(axis='y', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # Panel B: Per-trial sorted bars
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2)
    sort_idx = np.argsort(vals)
    sorted_labels = [labels[i] for i in sort_idx]
    sorted_vals = vals[sort_idx]
    colors_bar = [RED if v > mean_v + std_v else
                  ORANGE if v > mean_v else BLUE for v in sorted_vals]
    ax2.barh(range(len(sorted_labels)), sorted_vals, color=colors_bar,
             edgecolor='white', height=0.7, zorder=3)
    ax2.axvline(mean_v, color=RED, linewidth=1.5, linestyle='--', alpha=0.8)
    ax2.axvline(median_v, color=ORANGE, linewidth=1.5, linestyle='-', alpha=0.8)
    ax2.set_yticks(range(len(sorted_labels)))
    ax2.set_yticklabels(sorted_labels, fontsize=7, color=LABEL_COLOR)
    ax2.set_xlabel('Mean Phase Duration (steps)', fontsize=11, color=LABEL_COLOR)
    ax2.set_title('B. Per Trial (sorted)',
                  fontsize=13, color=TEXT_COLOR, fontweight='bold')
    for i, v in enumerate(sorted_vals):
        ax2.text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=6, color='#555555')
    ax2.grid(axis='x', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # Panel C: Phase duration vs tendency (scatter)
    ax3 = fig.add_subplot(gs[0, 2])
    _style_ax(ax3)
    ax3.scatter(tendencies, vals, color=PURPLE, s=60, edgecolors='white',
                linewidths=0.8, alpha=0.8, zorder=5)
    for i, r in enumerate(trial_records):
        ax3.annotate(r['label'], (tendencies[i], vals[i]),
                     fontsize=6, color='#555555', xytext=(4, 4),
                     textcoords='offset points')

    r_corr, p_corr = sp_stats.pearsonr(tendencies, vals)
    rho_corr, p_rho = sp_stats.spearmanr(tendencies, vals)
    slope, intercept, _, _, _ = sp_stats.linregress(tendencies, vals)
    x_line = np.linspace(min(tendencies), max(tendencies), 100)
    ax3.plot(x_line, slope * x_line + intercept, color=RED,
             linewidth=2, linestyle='--', zorder=4)
    ax3.text(0.97, 0.97,
             f'r = {r_corr:.3f}, p = {p_corr:.3f}\nρ = {rho_corr:.3f}, p = {p_rho:.3f}',
             transform=ax3.transAxes, fontsize=9, va='top', ha='right',
             color=LABEL_COLOR,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor=SPINE_COLOR, alpha=0.9))
    ax3.set_xlabel('Explore/Exploit Tendency (%)', fontsize=11, color=LABEL_COLOR)
    ax3.set_ylabel('Mean Phase Duration (steps)', fontsize=11, color=LABEL_COLOR)
    ax3.set_title('C. Phase Duration vs. Tendency',
                  fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax3.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)

    # Panel D: Box + strip
    ax4 = fig.add_subplot(gs[1, 0])
    _style_ax(ax4)
    ax4.boxplot(vals, vert=False, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor=BLUE, alpha=0.2, edgecolor=BLUE),
                medianprops=dict(color=ORANGE, linewidth=2),
                whiskerprops=dict(color='#666666'),
                capprops=dict(color='#666666'),
                flierprops=dict(marker='o', markerfacecolor=RED,
                                markersize=8, markeredgecolor='white'))
    np.random.seed(42)
    jitter = np.random.normal(1, 0.06, len(vals))
    ax4.scatter(vals, jitter, color=BLUE, alpha=0.7, s=40,
                edgecolors='white', linewidths=0.5, zorder=5)
    ax4.set_xlabel('Mean Phase Duration (steps)', fontsize=11, color=LABEL_COLOR)
    ax4.set_title('D. Box Plot + Individual Points',
                  fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax4.set_yticks([])
    ax4.grid(axis='x', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # Panel E: Per-participant bar (mean across trials)
    ax5 = fig.add_subplot(gs[1, 1])
    _style_ax(ax5)
    sorted_pids = sorted(pid_mean_mpd.keys(), key=lambda p: pid_mean_mpd[p])
    sorted_pid_vals = [pid_mean_mpd[p] for p in sorted_pids]
    grand_mean = np.mean(sorted_pid_vals)
    colors_pid = [GREEN if v <= grand_mean else ORANGE for v in sorted_pid_vals]
    ax5.barh(range(len(sorted_pids)), sorted_pid_vals, color=colors_pid,
             edgecolor='white', height=0.7, zorder=3)
    ax5.axvline(grand_mean, color=RED, linewidth=1.5, linestyle='--',
                label=f'Mean: {grand_mean:.2f}')
    ax5.set_yticks(range(len(sorted_pids)))
    ax5.set_yticklabels([f'P{p}' for p in sorted_pids], fontsize=8, color=LABEL_COLOR)
    ax5.set_xlabel('Mean Phase Duration (steps)', fontsize=11, color=LABEL_COLOR)
    ax5.set_title('E. Per Participant (mean across trials)',
                  fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax5.legend(fontsize=9, facecolor='white', edgecolor=SPINE_COLOR, labelcolor=LABEL_COLOR,
               loc='lower right')
    for i, v in enumerate(sorted_pid_vals):
        ax5.text(v + 0.08, i, f'{v:.1f}', va='center', fontsize=7, color='#555555')
    ax5.grid(axis='x', color=GRID_COLOR, linewidth=0.5, zorder=0)

    # Panel F: Stats table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor('white')
    ax6.axis('off')

    stats_data = [
        ['N (trials)', f'{len(vals)}'],
        ['N (participants)', f'{len(pid_mean_mpd)}'],
        ['Mean', f'{mean_v:.2f} steps'],
        ['Median', f'{median_v:.2f} steps'],
        ['SD', f'{std_v:.2f}'],
        ['SE', f'{se_v:.2f}'],
        ['Min', f'{vals.min():.2f}'],
        ['Max', f'{vals.max():.2f}'],
        ['Q1 (25%)', f'{q1:.2f}'],
        ['Q3 (75%)', f'{q3:.2f}'],
        ['IQR', f'{iqr_v:.2f}'],
        ['Skewness', f'{skew_v:+.2f}'],
        ['', ''],
        ['vs. Tendency r', f'{r_corr:+.3f} (p={p_corr:.3f})'],
        ['vs. Tendency ρ', f'{rho_corr:+.3f} (p={p_rho:.3f})'],
    ]

    table = ax6.table(cellText=stats_data, colLabels=['Statistic', 'Value'],
                      loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE_COLOR)
        if row == 0:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(color=TEXT_COLOR, fontweight='bold')
        elif row == 13:
            cell.set_facecolor('white')
            cell.set_edgecolor('white')
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color=LABEL_COLOR)

    ax6.set_title('F. Descriptive Statistics',
                  fontsize=13, color=TEXT_COLOR, fontweight='bold', pad=20)

    fig.suptitle(
        'M24: Switch Rate — Mean Phase Duration\n'
        'avg(median explore-run-length, median exploit-run-length) per trial',
        fontsize=15, color=TEXT_COLOR, fontweight='bold', y=1.01)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm24_phase_duration.png'
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {outpath}')

    # Print summary
    print(f'\n=== M24 Summary ===')
    print(f'Trials: {len(vals)}, Participants: {len(pid_mean_mpd)}')
    print(f'Mean phase duration: {mean_v:.2f} ± {std_v:.2f} steps')
    print(f'Range: {vals.min():.2f} — {vals.max():.2f}')
    print(f'Correlation with tendency: r={r_corr:.3f}, p={p_corr:.3f}')


if __name__ == '__main__':
    main()

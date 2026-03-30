#!/usr/bin/env python3
"""
M22: Explore/Exploit Tendency — Trial Comparison & Individual Differences
=========================================================================
Panel A: Slope plot (T1 → T2 per participant)
Panel B: Paired violin + box (T1 vs T2)
Panel C: Between-subject tendency distribution (mean of both trials)
Panel D: Summary statistics table + significance tests
Output: m22_tendency_analysis.png
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from m13_combined_binary import cos_dist
from m16_combined_lsa_median import build_lsa
from m18_typing_binary import page_had_typing_or_paste
from m20_cross_subject_median import compute_domain_medians, build_sequences, count_switches
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

# --- Style constants (light academic theme) ---
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


def collect_switch_rates(pids, pid_data):
    """Compute SR per participant per trial, return paired + per-pid data."""
    pid_sr = {}  # {pid: {trial_idx: sr}}
    for pid in pids:
        pid_sr[pid] = {}
        for t_idx, td in enumerate(pid_data[pid]):
            pts = td['points']
            if len(pts) < 2:
                continue
            sw = count_switches(pts)
            pid_sr[pid][t_idx] = sw / (len(pts) - 1) * 100

    # Paired data (participants with both trials)
    paired_pids = [p for p in pids if 0 in pid_sr[p] and 1 in pid_sr[p]]
    t1 = np.array([pid_sr[p][0] for p in paired_pids])
    t2 = np.array([pid_sr[p][1] for p in paired_pids])

    # Per-participant mean (across trials)
    pid_means = []
    for pid in pids:
        vals = list(pid_sr[pid].values())
        if vals:
            pid_means.append(np.mean(vals))
    pid_means = np.array(pid_means)

    return pid_sr, paired_pids, t1, t2, pid_means


def plot_slope(ax, paired_pids, t1, t2):
    """Panel A: Slope plot T1 → T2."""
    ax.set_facecolor(BG_COLOR)

    for i, pid in enumerate(paired_pids):
        delta = t2[i] - t1[i]
        color = GREEN if delta < -5 else (RED if delta > 5 else GRAY)
        alpha = 0.7 if abs(delta) > 5 else 0.35
        ax.plot([0, 1], [t1[i], t2[i]], 'o-', color=color,
                linewidth=1.8, markersize=6, alpha=alpha,
                markeredgecolor='white', markeredgewidth=0.5, zorder=3)
        # Label outliers
        if abs(delta) > 30:
            ax.annotate(f'P{pid}', (1.03, t2[i]), fontsize=7,
                        color=color, va='center')

    # Group means
    ax.plot([0, 1], [np.mean(t1), np.mean(t2)], 's-', color='black',
            linewidth=3, markersize=10, zorder=5, label='Group mean')
    ax.errorbar([0], [np.mean(t1)], yerr=[sp_stats.sem(t1)],
                color='black', capsize=5, capthick=2, zorder=6)
    ax.errorbar([1], [np.mean(t2)], yerr=[sp_stats.sem(t2)],
                color='black', capsize=5, capthick=2, zorder=6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Trial 1', 'Trial 2'], fontsize=12, color=TEXT_COLOR)
    ax.set_xlim(-0.2, 1.35)
    ax.set_ylim(-5, 105)
    ax.set_ylabel('Explore/Exploit Tendency (%)', fontsize=11, color=LABEL_COLOR)
    ax.set_title('A. Individual Trajectories (T1 → T2)',
                 fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax.axhline(50, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left', facecolor='white', edgecolor=SPINE_COLOR)
    ax.tick_params(colors=LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.5, zorder=0)


def plot_paired_violin(ax, t1, t2):
    """Panel B: Violin + box + paired dots."""
    ax.set_facecolor(BG_COLOR)

    parts = ax.violinplot([t1, t2], positions=[0, 1], showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(BLUE)
        pc.set_alpha(0.15)

    bp = ax.boxplot([t1, t2], positions=[0, 1], widths=0.25,
                    patch_artist=True, showfliers=False)
    for box in bp['boxes']:
        box.set_facecolor(BLUE)
        box.set_alpha(0.3)
        box.set_edgecolor(BLUE)
    for median in bp['medians']:
        median.set_color(ORANGE)
        median.set_linewidth(2.5)
    for whisker in bp['whiskers']:
        whisker.set_color(GRAY)
    for cap in bp['caps']:
        cap.set_color(GRAY)

    # Paired lines
    for i in range(len(t1)):
        ax.plot([0.15, 0.85], [t1[i], t2[i]], '-', color=GRAY,
                alpha=0.25, linewidth=0.8, zorder=2)
    ax.scatter(np.full(len(t1), 0.15), t1, color=BLUE, s=30,
               edgecolors='white', linewidths=0.5, alpha=0.8, zorder=4)
    ax.scatter(np.full(len(t2), 0.85), t2, color=BLUE, s=30,
               edgecolors='white', linewidths=0.5, alpha=0.8, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Trial 1', 'Trial 2'], fontsize=12, color=TEXT_COLOR)
    ax.set_ylim(-5, 105)
    ax.set_ylabel('Explore/Exploit Tendency (%)', fontsize=11, color=LABEL_COLOR)
    ax.set_title('B. Distribution Comparison',
                 fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax.axhline(50, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.5)
    ax.tick_params(colors=LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.5, zorder=0)


def plot_between_subject(ax, pid_means):
    """Panel C: Between-subject SR distribution."""
    ax.set_facecolor(BG_COLOR)

    bins = np.arange(0, 105, 10)
    ax.hist(pid_means, bins=bins, color=BLUE, edgecolor='white',
            alpha=0.7, zorder=3)

    mean_v = np.mean(pid_means)
    sd_v = np.std(pid_means, ddof=1)
    ax.axvline(mean_v, color=RED, linewidth=2, linestyle='--',
               label=f'Mean: {mean_v:.1f}%', zorder=5)
    ax.axvspan(mean_v - sd_v, mean_v + sd_v, alpha=0.08, color=RED,
               zorder=2, label=f'±1 SD: {sd_v:.1f}%')

    # Individual strip
    np.random.seed(42)
    y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5
    jitter = np.random.uniform(-0.3, -0.1, len(pid_means))
    ax.scatter(pid_means, jitter, color=PURPLE, s=40, alpha=0.7,
               edgecolors='white', linewidths=0.5, zorder=5, clip_on=False)

    ax.set_xlabel('Mean Tendency per Participant (%)', fontsize=11, color=LABEL_COLOR)
    ax.set_ylabel('Count', fontsize=11, color=LABEL_COLOR)
    ax.set_title('C. Between-Subject Variability',
                 fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax.legend(fontsize=9, facecolor='white', edgecolor=SPINE_COLOR, labelcolor=LABEL_COLOR)
    ax.tick_params(colors=LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.5, zorder=0)


def plot_stats_table(ax, paired_pids, t1, t2, pid_means):
    """Panel D: Summary statistics + test results."""
    ax.set_facecolor('white')
    ax.axis('off')

    diff = t2 - t1
    n_paired = len(paired_pids)

    # Tests
    wilcox_stat, wilcox_p = sp_stats.wilcoxon(t1, t2) if n_paired >= 5 else (np.nan, np.nan)
    t_stat, t_p = sp_stats.ttest_rel(t1, t2) if n_paired >= 5 else (np.nan, np.nan)

    # Effect size (Cohen's d for paired)
    d_z = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

    stats_data = [
        ['', 'Trial 1', 'Trial 2', 'Between-Subj'],
        ['N', f'{n_paired}', f'{n_paired}', f'{len(pid_means)}'],
        ['Mean', f'{np.mean(t1):.1f}%', f'{np.mean(t2):.1f}%', f'{np.mean(pid_means):.1f}%'],
        ['Median', f'{np.median(t1):.1f}%', f'{np.median(t2):.1f}%', f'{np.median(pid_means):.1f}%'],
        ['SD', f'{np.std(t1, ddof=1):.1f}%', f'{np.std(t2, ddof=1):.1f}%', f'{np.std(pid_means, ddof=1):.1f}%'],
        ['', '', '', ''],
        ['T1 vs T2', 'Statistic', 'p-value', 'Effect'],
        ['Wilcoxon', f'W = {wilcox_stat:.0f}', f'p = {wilcox_p:.3f}', ''],
        ['Paired t', f't = {t_stat:.2f}', f'p = {t_p:.3f}', f"Cohen's dz = {d_z:.2f}"],
        ['Mean Δ', f'{np.mean(diff):+.1f}%', f'SD = {np.std(diff, ddof=1):.1f}%', ''],
    ]

    table = ax.table(cellText=stats_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE_COLOR)
        if row in (0, 6):
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(color=TEXT_COLOR, fontweight='bold')
        elif row == 5:
            cell.set_facecolor('white')
            cell.set_edgecolor('white')
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color=LABEL_COLOR)

    ax.set_title('D. Summary & Significance Tests',
                 fontsize=13, color=TEXT_COLOR, fontweight='bold', pad=20)


def main():
    slugs, slug_idx, lsa, var_explained, n_components = build_lsa()
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    print('Computing cross-subject domain medians...')
    domain_medians = compute_domain_medians(pid_trials, slug_idx, lsa)
    pid_data = build_sequences(pids, pid_trials, slug_idx, lsa, domain_medians)

    pid_sr, paired_pids, t1, t2, pid_means = collect_switch_rates(pids, pid_data)

    # --- Build figure ---
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_slope(ax_a, paired_pids, t1, t2)
    plot_paired_violin(ax_b, t1, t2)
    plot_between_subject(ax_c, pid_means)
    plot_stats_table(ax_d, paired_pids, t1, t2, pid_means)

    sig_label = 'n.s.' if sp_stats.wilcoxon(t1, t2).pvalue > 0.05 else 'sig.'
    fig.suptitle(
        f'M22: Explore/Exploit Tendency — Trial Comparison & Individual Differences\n'
        f'T1 vs T2: {sig_label} (Wilcoxon p = {sp_stats.wilcoxon(t1, t2).pvalue:.3f}) · '
        f'Between-subject SD = {np.std(pid_means, ddof=1):.1f}%',
        fontsize=14, color=TEXT_COLOR, fontweight='bold', y=1.01)

    outpath = OUTPUT_DIR / 'm22_tendency_analysis.png'
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {outpath}')

    # Print summary
    print(f'\n=== M22 Summary ===')
    print(f'Paired N = {len(paired_pids)}')
    print(f'Trial 1: M={np.mean(t1):.1f}%, Trial 2: M={np.mean(t2):.1f}%')
    print(f'Wilcoxon p = {sp_stats.wilcoxon(t1, t2).pvalue:.3f}')
    print(f'Between-subject SD = {np.std(pid_means, ddof=1):.1f}%')


if __name__ == '__main__':
    main()

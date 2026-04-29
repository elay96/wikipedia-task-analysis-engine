#!/usr/bin/env python3
"""
M55: Same as M52, with two corrections
======================================
M55 reproduces M52 1:1 (same pipeline, same plots, same outputs) and
changes only two things:

  1. PDF slide 1 is replaced by a single unified Exclusion Audit
     diagram. Top half: funnel (125 -> 104 -> 101) with the deltas
     (-21, -3) called out in bold red next to each arrow. Bottom
     half: the 101 eligible participants split into the two tasks
     (art_history / psychology), showing how many of the 101 lost
     each specific question.

  2. The "Switch Counts by Condition" descriptive table on the final
     stats slide gains a Standard Error column
     (SE = SD / sqrt(n), n computed per condition after dropping NaN).

Everything else - the boxplots, the correlation grid, the PCA pages,
the visual condition comparison, the t-tests - is identical to M52.
No PNG composite is produced; PDF only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats

from helpers import load_trials, OUTPUT_DIR
from m52_final_composite_dv import (
    build_question_data, apply_exclusions, run_pca,
    plot_exclusion_summary, plot_counts_distribution,
    plot_correlation_grid,
    _plot_condition_comparison_page, _welch_df,
    FEATURE_NAMES,
    BG_COLOR, TEXT_COLOR, LABEL_COLOR, MUTED_COLOR, BORDER_COLOR,
    GRID_COLOR, BAR_COLOR, LINE_COLOR,
    EXCLUDED_COLOR, KEPT_COLOR,
    CONDITION_COLORS,
    THRESHOLD_S, IDLE_THRESHOLD_PCT, MIN_PAGE_VISITS, OUTLIER_SD,
)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

COUNT_COLS = ['count_time', 'count_topic', 'count_typing']
COUNT_LABELS = ['Time', 'Topic (LDA)', 'Typing/Paste']

# Set 1 question texts shown to participants. Q1 = psychology, Q2 =
# art_history per the order the user provided.
QUESTION_PSYCHOLOGY = (
    'What brain mechanisms help people make scientific discoveries?'
)
QUESTION_ART_HISTORY = (
    'How do ancient works of art play a role in contemporary society?'
)


# ---------------------------------------------------------------------------
# Slide 1 (unified): top = funnel 125 -> 104 -> 101 (with -21 / -3 deltas
# in red), bottom = the 101 eligible participants split into per-task
# columns showing how many of them lost each specific question.
# ---------------------------------------------------------------------------

def _build_per_task_sets(question_df, exclusion_summary):
    """Return (art_excl_pids, psy_excl_pids, outlier_pids).

    A participant is in art_excl_pids if their art_history trial was
    dropped by rule 1 (<3 pages) or rule 2 (idle >= 50%). Same for
    psychology. Note that a participant can be in only one of the sets
    and still survive into the final sample (their other trial provides
    the per-participant average).
    """
    art = question_df[question_df['domain'] == 'art_history']
    psy = question_df[question_df['domain'] == 'psychology']
    art_excl = set(int(p) for p in art.loc[
        art['excluded_pages'] | art['excluded_idle'], 'participant_id'
    ])
    psy_excl = set(int(p) for p in psy.loc[
        psy['excluded_pages'] | psy['excluded_idle'], 'participant_id'
    ])
    outlier_pids = set(int(p) for p in exclusion_summary['outlier_pids'])
    return art_excl, psy_excl, outlier_pids


def _draw_box(ax, x, y, w, h, fill, border, title, big, sub,
              title_color=None, big_fontsize=22, title_fontsize=11,
              sub_fontsize=8.5, lw=2):
    """Rounded rectangle with title row, big number row, optional sub-line."""
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.005,rounding_size=0.018',
        facecolor=fill, edgecolor=border, linewidth=lw,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(box)
    cx = x + w / 2
    ax.text(cx, y + h * 0.84, title, transform=ax.transAxes,
            ha='center', va='center', fontsize=title_fontsize,
            fontweight='bold', color=title_color or TEXT_COLOR)
    ax.text(cx, y + h * 0.55, big, transform=ax.transAxes,
            ha='center', va='center', fontsize=big_fontsize,
            fontweight='bold', color=title_color or TEXT_COLOR)
    if sub:
        ax.text(cx, y + h * 0.18, sub, transform=ax.transAxes,
                ha='center', va='center', fontsize=sub_fontsize,
                color=MUTED_COLOR, linespacing=1.4)


def _draw_arrow(ax, x_top, y_top, x_bot, y_bot, color=MUTED_COLOR, lw=2):
    ax.annotate(
        '', xy=(x_bot, y_bot), xytext=(x_top, y_top),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color=color, lw=lw),
    )


def _draw_delta(ax, x, y, delta_text, rule_text):
    """Big red '-N' badge + small rule description, anchored at (x, y)."""
    ax.text(x, y, delta_text, transform=ax.transAxes,
            ha='left', va='center', fontsize=20, fontweight='bold',
            color='#C62828')
    ax.text(x + 0.07, y, rule_text, transform=ax.transAxes,
            ha='left', va='center', fontsize=9.5, color=LABEL_COLOR,
            linespacing=1.35)


def plot_unified_audit_slide(pdf, question_df, exclusion_summary,
                             final_pids, title='M55: Exclusion Audit'):
    """Single-page audit. Top: funnel 125 -> 104 -> 101 with red deltas.
    Bottom: the 101 eligible participants split per task (ART / PSY).
    """
    n_part_total = exclusion_summary['n_participants_before']
    n_fully = len(exclusion_summary['fully_excluded_pids'])
    n_outliers = exclusion_summary['n_outliers']
    n_final = exclusion_summary['n_final']
    n_after_q = n_part_total - n_fully

    # Per-task figures restricted to the FINAL eligible sample.
    # A participant is in art_lost / psy_lost if their corresponding
    # question failed rules 1 or 2; these are the only ones excluded
    # within the 101 (the "both" group is already gone, by definition).
    art_excl, psy_excl, _ = _build_per_task_sets(question_df, exclusion_summary)
    final_set = set(int(p) for p in final_pids)
    art_lost = len(art_excl & final_set)
    psy_lost = len(psy_excl & final_set)
    art_valid = n_final - art_lost
    psy_valid = n_final - psy_lost
    both_clean = n_final - len((art_excl | psy_excl) & final_set)
    lost_one = n_final - both_clean

    # Slightly bigger canvas (12 x 9) than the rest of the deck so the
    # diagram has more breathing room. PDF can mix page sizes; downstream
    # readers handle this fine.
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor(BG_COLOR)

    # ===== TITLE block =====
    # Title only - the per-question text appears inside the bottom
    # task boxes, no need to repeat it up here.
    fig.text(0.5, 0.965, title,
             ha='center', va='top', fontsize=20, fontweight='bold',
             color=TEXT_COLOR)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ===== TOP HALF: funnel 125 -> 104 -> 101 =====
    # Funnel pushed up into the space freed by removing the subtitle,
    # leaving more vertical room for the per-task split below.
    box_w = 0.28
    box_h = 0.11
    box_x = (1 - box_w) / 2
    cx = box_x + box_w / 2

    y_started = 0.81
    y_after_q = 0.64
    y_eligible = 0.47

    _draw_box(
        ax, box_x, y_started, box_w, box_h,
        fill='#E3F2FD', border='#1976D2',
        title='Started', big=f'{n_part_total}',
        sub=f'participants  -  {exclusion_summary["n_total_questions"]} questions',
        title_color='#0D47A1', big_fontsize=26,
    )

    _draw_arrow(ax, cx, y_started, cx, y_after_q + box_h)
    # -21 with a friendlier description: spell out what "rules 1+2" mean
    # in plain English so the reader doesn't have to remember the spec.
    _draw_delta(
        ax, cx + box_w / 2 + 0.025,
        (y_started + y_after_q + box_h) / 2,
        f'-{n_fully}',
        'both questions disqualified\n'
        f'(<{MIN_PAGE_VISITS} pages, or no meaningful\n'
        f'activity for {IDLE_THRESHOLD_PCT:.0f}%+ of the time)',
    )

    _draw_box(
        ax, box_x, y_after_q, box_w, box_h,
        fill='#FFF3E0', border='#E65100',
        title='After question-level rules', big=f'{n_after_q}',
        sub='participants with at least one valid question',
        title_color='#BF360C', big_fontsize=26,
    )

    _draw_arrow(ax, cx, y_after_q, cx, y_eligible + box_h)
    _draw_delta(
        ax, cx + box_w / 2 + 0.025,
        (y_after_q + y_eligible + box_h) / 2,
        f'-{n_outliers}',
        '3 SD outliers\n'
        f'(switch count > {OUTLIER_SD} SD\n'
        'from the group mean)',
    )

    _draw_box(
        ax, box_x, y_eligible, box_w, box_h,
        fill='#E8F5E9', border='#2E7D32',
        title='Eligible (final sample)', big=f'N = {n_final}',
        sub='used in every chart and statistic that follows',
        title_color='#1B5E20', big_fontsize=26, lw=2.5,
    )

    # ===== DIVIDER + SECTION HEADER =====
    # Single header line, with a clear vertical gap before the anchor
    # pill so the two never touch.
    y_divider = 0.41
    ax.plot([0.06, 0.94], [y_divider, y_divider],
            color=BORDER_COLOR, linewidth=0.6,
            transform=ax.transAxes, zorder=1)
    ax.text(0.5, y_divider - 0.022,
            f'Question-Level Exclusion within the {n_final} Eligible Participants',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=14, fontweight='bold', color=TEXT_COLOR)

    # ===== BOTTOM HALF: 101 anchor -> two task boxes =====
    # Anchor pill sits well below the header (gap of ~0.04 axis units)
    # so the section title can never cross into it; arrows fan diagonally
    # to the per-question boxes.
    y_anchor = 0.305
    anchor_w = 0.09
    anchor_h = 0.055
    anchor_x = (1 - anchor_w) / 2
    from matplotlib.patches import FancyBboxPatch
    pill = FancyBboxPatch(
        (anchor_x, y_anchor), anchor_w, anchor_h,
        boxstyle='round,pad=0.005,rounding_size=0.022',
        facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(pill)
    ax.text(anchor_x + anchor_w / 2, y_anchor + anchor_h / 2,
            f'{n_final}', transform=ax.transAxes,
            ha='center', va='center', fontsize=20, fontweight='bold',
            color='#1B5E20')

    # Branch geometry. Q1 (psychology) goes left, Q2 (art_history) goes
    # right. Task boxes pushed down so the diagonal arrows have room to
    # breathe (gap of ~0.07 axis units between anchor pill and task tops).
    import textwrap
    psy_color = '#C62828'
    art_color = '#1976D2'
    task_box_w = 0.32
    task_box_h = 0.22
    y_task = 0.015
    x_psy = 0.05
    x_art = 1 - x_psy - task_box_w
    cx_psy = x_psy + task_box_w / 2
    cx_art = x_art + task_box_w / 2

    # Arrows fan from the bottom corners of the anchor pill to the top
    # centres of the task boxes. Using the corners (not the midpoint)
    # gives the V-shape a wider, less cramped opening.
    _draw_arrow(ax, anchor_x + anchor_w * 0.10, y_anchor,
                cx_psy, y_task + task_box_h, color=psy_color, lw=2)
    _draw_arrow(ax, anchor_x + anchor_w * 0.90, y_anchor,
                cx_art, y_task + task_box_h, color=art_color, lw=2)

    # Helper to draw a task box with the question text inside.
    def _draw_task_box(x, fill, border, color, q_label, q_subject,
                       q_text, valid):
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch(
            (x, y_task), task_box_w, task_box_h,
            boxstyle='round,pad=0.005,rounding_size=0.022',
            facecolor=fill, edgecolor=border, linewidth=2,
            transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(box)
        cx_box = x + task_box_w / 2

        # Header row: "Question 1  -  psychology"
        ax.text(cx_box, y_task + task_box_h - 0.022,
                f'{q_label}  -  {q_subject}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold', color=color)

        # Wrap the question to ~38 chars so it always fits in 2 lines
        # within the box, then render it as a single multiline text.
        wrapped = textwrap.fill(q_text, width=38)
        ax.text(cx_box, y_task + task_box_h - 0.052,
                f'"{wrapped}"',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9.5, color=TEXT_COLOR, style='italic',
                linespacing=1.45)

        # Big "valid / total" line
        ax.text(cx_box, y_task + 0.062,
                f'{valid} / {n_final}',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=24, fontweight='bold', color=color)
        ax.text(cx_box, y_task + 0.024,
                'valid questions',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9.5, color=MUTED_COLOR)

    _draw_task_box(
        x=x_psy, fill='#FFEBEE', border=psy_color, color=psy_color,
        q_label='Question 1', q_subject='psychology',
        q_text=QUESTION_PSYCHOLOGY, valid=psy_valid,
    )
    _draw_task_box(
        x=x_art, fill='#E3F2FD', border=art_color, color=art_color,
        q_label='Question 2', q_subject='art_history',
        q_text=QUESTION_ART_HISTORY, valid=art_valid,
    )

    # Red "-N" badges sit on the OUTER side of each diagonal arrow (away
    # from the page centre) at the arrow midpoint, with the "lost this Q"
    # caption directly underneath. Both stay clear of the pill above and
    # the task box below.
    arrow_mid_y = (y_anchor + y_task + task_box_h) / 2

    # Midpoint x of each arrow: between the pill exit and the task box centre
    psy_arrow_mid_x = (anchor_x + anchor_w * 0.10 + cx_psy) / 2
    art_arrow_mid_x = (anchor_x + anchor_w * 0.90 + cx_art) / 2

    # Push the badge slightly outward from the arrow so it doesn't overlap
    psy_badge_x = psy_arrow_mid_x - 0.045
    art_badge_x = art_arrow_mid_x + 0.045

    ax.text(psy_badge_x, arrow_mid_y + 0.008, f'-{psy_lost}',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=20, fontweight='bold', color='#C62828')
    ax.text(psy_badge_x, arrow_mid_y + 0.003, 'lost this Q',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color=MUTED_COLOR, style='italic')

    ax.text(art_badge_x, arrow_mid_y + 0.008, f'-{art_lost}',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=20, fontweight='bold', color='#C62828')
    ax.text(art_badge_x, arrow_mid_y + 0.003, 'lost this Q',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color=MUTED_COLOR, style='italic')

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()


# ---------------------------------------------------------------------------
# Slide 5b override: same layout as M52 but with an SE column
# ---------------------------------------------------------------------------

def plot_condition_stats_with_se(pdf, avg_df, scores, conditions):
    """Recreates M52's slide 5b with an extra SE column in the descriptives.

    Layout, t-tests, interpretation block and styling are identical to
    M52._plot_condition_comparison_page (page 5b). Only the descriptive
    table widens to include SE = SD / sqrt(n_per_condition).
    """
    pids = avg_df['participant_id'].values
    cond_labels = [conditions.get(p, '') for p in pids]
    pc1 = scores[:, 0]
    diffuse_mask = np.array([c == 'diffuse' for c in cond_labels])
    clumpy_mask = np.array([c == 'clumpy' for c in cond_labels])
    pc1_diffuse = pc1[diffuse_mask]
    pc1_clumpy = pc1[clumpy_mask]
    n_d, n_c = int(diffuse_mask.sum()), int(clumpy_mask.sum())

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        'Condition Comparison: Descriptive Statistics & t-Tests\n'
        f'N = {len(pids)} (Diffuse: {n_d}, Clumpy: {n_c})',
        fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.97,
    )

    ax = fig.add_axes([0.06, 0.05, 0.88, 0.82])
    ax.axis('off')

    test_rows = []
    all_vars = COUNT_COLS + ['PC1']
    all_labels = COUNT_LABELS + ['PC1 (composite)']
    for var, label in zip(all_vars, all_labels):
        if var == 'PC1':
            d_vals, c_vals = pc1_diffuse, pc1_clumpy
        else:
            # Drop NaN before computing n / SE - missing values must not
            # inflate the denominator of SE = SD / sqrt(n).
            d_vals = avg_df.loc[diffuse_mask, var].dropna().values
            c_vals = avg_df.loc[clumpy_mask, var].dropna().values

        d_n, c_n = len(d_vals), len(c_vals)
        d_mean = float(np.mean(d_vals)) if d_n else float('nan')
        c_mean = float(np.mean(c_vals)) if c_n else float('nan')
        d_sd = float(np.std(d_vals, ddof=1)) if d_n > 1 else 0.0
        c_sd = float(np.std(c_vals, ddof=1)) if c_n > 1 else 0.0
        d_se = d_sd / np.sqrt(d_n) if d_n else float('nan')
        c_se = c_sd / np.sqrt(c_n) if c_n else float('nan')

        t_stat, p_val = sp_stats.ttest_ind(d_vals, c_vals, equal_var=False)
        df_welch = _welch_df(d_vals, c_vals)
        pooled_sd = np.sqrt(
            ((d_n - 1) * d_sd ** 2 + (c_n - 1) * c_sd ** 2) / (d_n + c_n - 2)
        ) if (d_sd + c_sd) > 0 else 0
        cohens_d = (c_mean - d_mean) / pooled_sd if pooled_sd > 0 else 0

        test_rows.append({
            'label': label, 'd_n': d_n, 'c_n': c_n,
            'd_mean': d_mean, 'd_sd': d_sd, 'd_se': d_se,
            'c_mean': c_mean, 'c_sd': c_sd, 'c_se': c_se,
            't': t_stat, 'df': df_welch, 'p': p_val, 'd_cohen': cohens_d,
        })

    lines = []
    lines.append('DESCRIPTIVE STATISTICS  (M / SD / SE per condition)')
    lines.append('=' * 88)
    diffuse_hdr = f'Diffuse (n={n_d})'
    clumpy_hdr = f'Clumpy (n={n_c})'
    lines.append(f'{"Variable":<18} {diffuse_hdr:>32}  {clumpy_hdr:>32}')
    lines.append('-' * 88)
    for r in test_rows:
        d_str = f'M={r["d_mean"]:5.2f}  SD={r["d_sd"]:4.2f}  SE={r["d_se"]:4.2f}'
        c_str = f'M={r["c_mean"]:5.2f}  SD={r["c_sd"]:4.2f}  SE={r["c_se"]:4.2f}'
        lines.append(f'{r["label"]:<18} {d_str:>32}  {c_str:>32}')
    lines.append('')
    lines.append('  SE = SD / sqrt(n); n is post-exclusion non-NaN count.')
    lines.append('')
    lines.append('')
    lines.append('INDEPENDENT-SAMPLES t-TESTS (Welch)')
    lines.append('=' * 88)
    lines.append(
        f'{"Variable":<18} {"t":>7} {"df":>6} {"p":>9} {"Cohen d":>9}  {"Sig.":>6}'
    )
    lines.append('-' * 88)
    for r in test_rows:
        sig = '***' if r['p'] < .001 else '**' if r['p'] < .01 else '*' if r['p'] < .05 else 'n.s.'
        lines.append(
            f'{r["label"]:<18} {r["t"]:>7.2f} {r["df"]:>6.1f} '
            f'{r["p"]:>9.4f} {r["d_cohen"]:>+9.2f}  {sig:>6}'
        )
    lines.append('')
    lines.append('  * p < .05   ** p < .01   *** p < .001')
    lines.append('')
    lines.append('')
    lines.append('INTERPRETATION')
    lines.append('=' * 88)

    pc1_row = test_rows[-1]
    diff = pc1_row['c_mean'] - pc1_row['d_mean']
    direction = 'more' if diff > 0 else 'fewer'
    lines.append(f'  PC1 difference (Clumpy - Diffuse): {diff:+.2f}')
    lines.append(f'  Clumpy condition shows {direction} explore-exploit switches')
    lines.append(f'  Effect size (Cohen\'s d): {pc1_row["d_cohen"]:+.2f}')
    if pc1_row['p'] < .05:
        lines.append(f'  Result is statistically significant (p = {pc1_row["p"]:.4f})')
    else:
        lines.append(f'  Result is NOT statistically significant (p = {pc1_row["p"]:.4f})')
        lines.append(f'  Low statistical power due to small sample (N = {len(pids)})')

    max_d_row = max(test_rows[:-1], key=lambda r: abs(r['d_cohen']))
    lines.append('')
    lines.append(f'  Largest sub-variable effect: {max_d_row["label"]}')
    lines.append(f'    d = {max_d_row["d_cohen"]:+.2f}, p = {max_d_row["p"]:.4f}')

    lines.append('')
    lines.append('-' * 88)
    lines.append('  Note: Typing metric uses answer_snapshot bursts + paste events.')

    text = '\n'.join(lines)
    ax.text(
        0.02, 0.98, text, transform=ax.transAxes,
        fontsize=9, family='monospace', va='top', color=TEXT_COLOR,
        linespacing=1.45,
    )

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()

    return test_rows


# ---------------------------------------------------------------------------
# Slide 5a (visuals only, no stats text) - reused verbatim from M52 by
# splitting _plot_condition_comparison_page into "page 5a" and writing our
# own page 5b. M52's helper writes BOTH 5a and 5b; we want 5a but not 5b.
# Easiest path: copy 5a here (short), call our SE-aware 5b after it.
# ---------------------------------------------------------------------------

def _plot_condition_visual_page(pdf, avg_df, scores, conditions):
    """Page 5a only: the four-panel visual comparison from M52, no stats text."""
    pids = avg_df['participant_id'].values
    cond_labels = [conditions.get(p, '') for p in pids]
    pc1 = scores[:, 0]
    diffuse_mask = np.array([c == 'diffuse' for c in cond_labels])
    clumpy_mask = np.array([c == 'clumpy' for c in cond_labels])
    pc1_diffuse = pc1[diffuse_mask]
    pc1_clumpy = pc1[clumpy_mask]
    n_d, n_c = int(diffuse_mask.sum()), int(clumpy_mask.sum())
    mean_d, mean_c = np.mean(pc1_diffuse), np.mean(pc1_clumpy)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        'Condition Comparison: Clumpy vs Diffuse\n'
        f'N = {len(pids)} (Diffuse: {n_d}, Clumpy: {n_c})',
        fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                          top=0.88, bottom=0.07, left=0.09, right=0.95)

    # Top-left: PC1 strip plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG_COLOR)
    jitter_d = np.random.default_rng(42).uniform(-0.06, 0.06, size=len(pc1_diffuse))
    jitter_c = np.random.default_rng(7).uniform(-0.06, 0.06, size=len(pc1_clumpy))
    ax1.scatter(np.zeros(len(pc1_diffuse)) + jitter_d, pc1_diffuse,
                color=CONDITION_COLORS['diffuse'], s=70, alpha=0.85,
                edgecolors='#333', linewidth=0.6, zorder=3)
    ax1.scatter(np.ones(len(pc1_clumpy)) + jitter_c, pc1_clumpy,
                color=CONDITION_COLORS['clumpy'], s=70, alpha=0.85,
                edgecolors='#333', linewidth=0.6, zorder=3)

    pids_d = pids[diffuse_mask]
    pids_c = pids[clumpy_mask]
    for i, (pid, val) in enumerate(zip(pids_d, pc1_diffuse)):
        ax1.annotate(f'P{pid}', (jitter_d[i], val), fontsize=7,
                     color=LABEL_COLOR, xytext=(6, 0), textcoords='offset points')
    for i, (pid, val) in enumerate(zip(pids_c, pc1_clumpy)):
        ax1.annotate(f'P{pid}', (1 + jitter_c[i], val), fontsize=7,
                     color=LABEL_COLOR, xytext=(6, 0), textcoords='offset points')

    bar_w = 0.3
    ax1.hlines(mean_d, -bar_w, bar_w, color=CONDITION_COLORS['diffuse'],
               linewidth=2.5, zorder=4)
    ax1.hlines(mean_c, 1 - bar_w, 1 + bar_w, color=CONDITION_COLORS['clumpy'],
               linewidth=2.5, zorder=4)
    ax1.text(-bar_w - 0.05, mean_d, f'M={mean_d:.2f}', ha='right', va='center',
             fontsize=9, color=CONDITION_COLORS['diffuse'], fontweight='bold')
    ax1.text(1 + bar_w + 0.05, mean_c, f'M={mean_c:.2f}', ha='left', va='center',
             fontsize=9, color=CONDITION_COLORS['clumpy'], fontweight='bold')

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Diffuse', 'Clumpy'], fontsize=11, fontweight='bold')
    ax1.set_ylabel('PC1 Score (Composite Index)', color=LABEL_COLOR, fontweight='bold')
    ax1.set_title('PC1: Composite Explore-Exploit\nby Condition', color=TEXT_COLOR,
                  fontweight='bold', fontsize=12)
    ax1.set_xlim(-0.5, 1.5)
    ax1.axhline(0, color=BORDER_COLOR, linewidth=0.5, linestyle=':')
    ax1.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax1.spines.values():
        spine.set_color(BORDER_COLOR)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Top-right: grouped bars (M +/- SD, same as M52 page 5a)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG_COLOR)
    means_d_list = [avg_df.loc[diffuse_mask, c].mean() for c in COUNT_COLS]
    means_c_list = [avg_df.loc[clumpy_mask, c].mean() for c in COUNT_COLS]
    # SE = SD / sqrt(n_non_nan) per condition per variable. .std() and
    # .count() both skip NaN, so the denominator matches the numerator's
    # sample.
    ses_d_list = [
        avg_df.loc[diffuse_mask, c].std() / np.sqrt(avg_df.loc[diffuse_mask, c].count())
        for c in COUNT_COLS
    ]
    ses_c_list = [
        avg_df.loc[clumpy_mask, c].std() / np.sqrt(avg_df.loc[clumpy_mask, c].count())
        for c in COUNT_COLS
    ]

    x = np.arange(len(COUNT_COLS))
    w = 0.32
    bars_d = ax2.bar(x - w / 2, means_d_list, w, yerr=ses_d_list, capsize=4,
                     color=CONDITION_COLORS['diffuse'], edgecolor='white',
                     linewidth=0.5, label='Diffuse', alpha=0.85, zorder=2)
    bars_c = ax2.bar(x + w / 2, means_c_list, w, yerr=ses_c_list, capsize=4,
                     color=CONDITION_COLORS['clumpy'], edgecolor='white',
                     linewidth=0.5, label='Clumpy', alpha=0.85, zorder=2)
    for bar, val in zip(bars_d, means_d_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                 color=CONDITION_COLORS['diffuse'], fontweight='bold')
    for bar, val in zip(bars_c, means_c_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                 color=CONDITION_COLORS['clumpy'], fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(COUNT_LABELS, fontsize=10)
    ax2.set_ylabel('Mean Switch Count', color=LABEL_COLOR, fontweight='bold')
    ax2.set_title('Raw Switch Counts\nby Condition (Mean +/- SE)', color=TEXT_COLOR,
                  fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax2.spines.values():
        spine.set_color(BORDER_COLOR)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Bottom-left: parallel coordinates per participant
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG_COLOR)
    for pid in pids:
        cond = conditions.get(pid, '')
        color = CONDITION_COLORS.get(cond, '#1976D2')
        vals = [avg_df.loc[avg_df['participant_id'] == pid, c].values[0] for c in COUNT_COLS]
        alpha = 0.7 if cond == 'diffuse' else 0.9
        lw = 1.5 if cond == 'diffuse' else 2.5
        ax3.plot(range(3), vals, marker='o', color=color, alpha=alpha,
                 linewidth=lw, markersize=5, zorder=3)
        ax3.annotate(f'P{pid}', (2, vals[2]), fontsize=7, color=color,
                     xytext=(5, 0), textcoords='offset points')
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(COUNT_LABELS, fontsize=10)
    ax3.set_ylabel('Switch Count', color=LABEL_COLOR, fontweight='bold')
    ax3.set_title('Individual Profiles\nby Switch Type', color=TEXT_COLOR,
                  fontweight='bold', fontsize=12)
    ax3.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax3.spines.values():
        spine.set_color(BORDER_COLOR)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Bottom-right: legend box
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CONDITION_COLORS['diffuse'],
               markersize=10, label=f'Diffuse (n={n_d})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CONDITION_COLORS['clumpy'],
               markersize=10, label=f'Clumpy (n={n_c})'),
    ]
    ax4.legend(handles=legend_elements, loc='center', fontsize=13,
               framealpha=0.9, edgecolor=BORDER_COLOR, title='Conditions',
               title_fontsize=14)

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()


# ---------------------------------------------------------------------------
# Full PDF report - mirrors M52 exactly except for slide 1 and slide 5b.
# ---------------------------------------------------------------------------

def create_pdf_report_m55(avg_df, pca, scores, pct, conditions, question_df,
                          exclusion_summary, output_path):
    with PdfPages(output_path) as pdf:
        # --- Slide 1 (unified): funnel + per-task split ---
        plot_unified_audit_slide(
            pdf, question_df, exclusion_summary,
            final_pids=avg_df['participant_id'].values,
        )

        # --- Slide 2: Descriptive Statistics (identical to M52) ---
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.patch.set_facecolor(BG_COLOR)
        fig.suptitle('Descriptive Statistics: Raw Switch Counts',
                     fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98)
        plot_counts_distribution(axes[0], avg_df)
        plot_correlation_grid(axes[1], avg_df)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, facecolor=BG_COLOR)
        plt.close()

        # --- Slide 3: PCA Results (identical to M52) ---
        _plot_pca_results_page(pdf, pca, pct)

        # --- Slide 4: Per-participant PC1 scores (identical to M52) ---
        _plot_per_participant_scores(pdf, avg_df, scores, conditions)

        # --- Slide 5a: Condition visual comparison (identical to M52) ---
        _plot_condition_visual_page(pdf, avg_df, scores, conditions)

        # --- Slide 5b (replaced): condition descriptives WITH SE column ---
        plot_condition_stats_with_se(pdf, avg_df, scores, conditions)

    print(f'Saved: {output_path}')


def _plot_pca_results_page(pdf, pca, pct):
    """M52 page 3 (PCA scree + loadings text), copied verbatim."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('PCA Results: Dimensionality Reduction on Raw Switch Counts',
                 fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98)

    gs = fig.add_gridspec(1, 2, wspace=0.05,
                          top=0.88, bottom=0.08, left=0.08, right=0.95,
                          width_ratios=[1, 1.2])

    ax_scree = fig.add_subplot(gs[0, 0])
    cumulative = np.cumsum(pct)
    pc_labels = [f'PC{i + 1}' for i in range(len(pct))]
    ax_scree.set_facecolor(BG_COLOR)
    bars = ax_scree.bar(pc_labels, pct, color=BAR_COLOR, zorder=2,
                        edgecolor=BORDER_COLOR)
    ax_scree.plot(pc_labels, cumulative, color=LINE_COLOR, marker='o',
                  linewidth=2, zorder=3)
    for bar, val in zip(bars, pct):
        ax_scree.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                      f'{val:.1f}%', ha='center', va='center', color='white',
                      fontsize=11, fontweight='bold', zorder=4)
    for x, cum_val in zip(pc_labels, cumulative):
        ax_scree.text(x, cum_val + 4, f'{cum_val:.1f}%', ha='center',
                      va='bottom', color=LINE_COLOR, fontsize=9, fontweight='bold')
    ax_scree.set_ylabel('Variance Explained (%)', color=LABEL_COLOR, fontweight='bold')
    ax_scree.set_title('Variance Explained by Each\nPrincipal Component',
                       color=TEXT_COLOR, fontweight='bold', fontsize=13)
    ax_scree.set_ylim(0, 120)
    ax_scree.tick_params(colors=MUTED_COLOR)
    ax_scree.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax_scree.spines.values():
        spine.set_color(BORDER_COLOR)

    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis('off')
    loadings = pca.components_

    explanation = 'PCA Loadings (variable weights per component):\n\n'
    for i in range(3):
        explanation += f'  PC{i + 1} ({pct[i]:.1f}% variance):\n'
        load_order = np.argsort(-np.abs(loadings[i]))
        for j in load_order:
            val = loadings[i, j]
            bar_len = int(abs(val) * 20)
            bar_char = '+' if val > 0 else '-'
            bar = bar_char * bar_len
            explanation += f'    {FEATURE_NAMES[j]:14s}  {val:+.3f}  {bar}\n'
        explanation += '\n'

    explanation += 'Interpretation:\n\n'
    for i in range(3):
        dominant_idx = np.argmax(np.abs(loadings[i]))
        dominant_name = FEATURE_NAMES[dominant_idx]
        dominant_val = loadings[i, dominant_idx]
        direction = 'more' if dominant_val > 0 else 'fewer'
        var_short = dominant_name.split()[-1].lower()
        explanation += (
            f'  PC{i + 1}: Primarily reflects {var_short} switching\n'
            f'       (loading = {dominant_val:+.3f})\n'
            f'       Higher score = {direction} {var_short} transitions\n\n'
        )
    ax_text.text(0.02, 0.98, explanation, transform=ax_text.transAxes,
                 fontsize=10, family='monospace', va='top', color=TEXT_COLOR,
                 linespacing=1.4)

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()


def _plot_per_participant_scores(pdf, avg_df, scores, conditions):
    """M52 page 4 (PC1 bars per participant), copied verbatim."""
    pids = avg_df['participant_id'].values
    n_participants = len(pids)
    fig_h = max(6, 0.4 * n_participants + 2)
    fig, ax = plt.subplots(figsize=(11, min(fig_h, 14)))
    fig.patch.set_facecolor(BG_COLOR)

    pc1 = scores[:, 0]
    idx = np.argsort(pc1)
    sorted_pids = pids[idx]
    sorted_pc1 = pc1[idx]
    colors = [CONDITION_COLORS.get(conditions.get(p, ''), '#1976D2')
              for p in sorted_pids]

    bar_height = 0.65 if n_participants <= 30 else 0.5
    ax.barh(range(n_participants), sorted_pc1, color=colors,
            edgecolor='white', linewidth=0.5, height=bar_height)
    for i, val in enumerate(sorted_pc1):
        ha = 'left' if val >= 0 else 'right'
        offset = 0.08 if val >= 0 else -0.08
        ax.text(val + offset, i, f'{val:.1f}', ha=ha, va='center',
                fontsize=8, color=MUTED_COLOR)

    ax.set_yticks(range(n_participants))
    fs = 9 if n_participants <= 25 else 7
    ax.set_yticklabels([f'P{p}' for p in sorted_pids], fontsize=fs)
    ax.set_xlabel('PC1 Score (Composite Explore-Exploit Index)',
                  color=LABEL_COLOR, fontweight='bold', fontsize=11)
    ax.axvline(0, color=BORDER_COLOR, linewidth=1)

    mean_pc1 = np.mean(pc1)
    ax.axvline(mean_pc1, color=EXCLUDED_COLOR, linestyle='--', linewidth=1.5)
    ax.text(mean_pc1 + 0.1, -1.2, f'Mean = {mean_pc1:.2f}',
            color=EXCLUDED_COLOR, fontsize=10, fontweight='bold', va='top')

    ax.set_title('Individual PC1 Scores: Composite Explore-Exploit Index',
                 color=TEXT_COLOR, fontweight='bold', fontsize=14, pad=15)
    ax.set_facecolor(BG_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.tick_params(colors=MUTED_COLOR)
    ax.set_ylim(-1.5, n_participants - 0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M55] Same as M52 with corrected slide 1 + SE column')
    print('=' * 60)

    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    conditions = {tr['pid']: tr['condition'] for tr in trials}
    all_real_pids = sorted(set(
        tr['pid'] for tr in trials if tr['domain'] != 'practice'
    ))
    print(f'  Real-trial participants in raw data: {len(all_real_pids)}')

    print('\n--- Building question-level data (recomputed from Game.csv) ---')
    question_df = build_question_data(trials)
    print(f'  Total question rows: {len(question_df)}')

    print('\n--- Applying M52 exclusion criteria ---')
    avg_df, exclusion_summary = apply_exclusions(question_df)

    print('\n--- Per-task exclusion breakdown ---')
    art_excl, psy_excl, outlier_pids = _build_per_task_sets(
        question_df, exclusion_summary
    )
    print(f'  art_history excluded: {len(art_excl)}')
    print(f'  psychology  excluded: {len(psy_excl)}')
    print(f'  art only:  {len(art_excl - psy_excl)}'
          f'  | psy only: {len(psy_excl - art_excl)}'
          f'  | both: {len(art_excl & psy_excl)}')

    # Sanity check: every PID excluded by rules 1+2 (= "both") plus every
    # 3SD outlier must be absent from avg_df. Otherwise the downstream
    # charts would silently include them.
    excluded_pids = (art_excl & psy_excl) | outlier_pids
    leak = excluded_pids & set(int(p) for p in avg_df['participant_id'])
    assert not leak, f'Excluded PIDs leaked into final sample: {sorted(leak)}'
    expected_final = len(all_real_pids) - len(excluded_pids)
    assert len(avg_df) == expected_final, (
        f'Final N mismatch: avg_df has {len(avg_df)}, '
        f'expected {expected_final} ({len(all_real_pids)} - {len(excluded_pids)})'
    )
    print('  Verified: excluded PIDs do not appear in any chart or table.')

    print('\n--- PCA on final sample ---')
    pca, scores, pct = run_pca(avg_df)
    for i, v in enumerate(pct):
        print(f'  PC{i + 1}: {v:.1f}%')

    print('\n--- Saving outputs ---')

    # Final-sample CSV (matches M52 format)
    final_csv = OUTPUT_DIR / 'm55_final_composite_dv.csv'
    pd.DataFrame({
        'participant_id': avg_df['participant_id'].values,
        'condition': [conditions.get(p, '') for p in avg_df['participant_id']],
        'count_time': avg_df['count_time'].values,
        'count_topic': avg_df['count_topic'].values,
        'count_typing': avg_df['count_typing'].values,
        'PC1': scores[:, 0],
        'PC2': scores[:, 1],
        'PC3': scores[:, 2],
    }).to_csv(final_csv, index=False)
    print(f'Saved: {final_csv}')

    # PDF report (no PNG composite this run)
    pdf_path = OUTPUT_DIR / 'm55_final_composite_dv.pdf'
    create_pdf_report_m55(
        avg_df, pca, scores, pct, conditions, question_df,
        exclusion_summary, pdf_path,
    )

    print('\nDone.')


if __name__ == '__main__':
    main()

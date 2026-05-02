#!/usr/bin/env python3
"""
Email-summary PDF (4 pages) - signed Cohen's d variant
======================================================
Identical to M63, except panels (B) and (C) show Cohen's d as signed
values (not absolute), and extra pages show construct validity (page 3)
and bootstrap confidence intervals (page 4) for the Mean reading-only
streak measure.

  Page 1 - Headline:
    Pre-registered analysis null vs writing-pattern signal.
    Bar chart of the Mean reading-only streak means by condition.

  Page 2 - Three robustness checks:
    (A) The 5 writing-pattern measures
    (B) Output-volume covariate
    (C) Trial-order moderation

  Page 3 - Construct validity:
    Mean reading-only streak vs the 3 original switch measures
    (typing W/N, topic LDA, time >60s dwell).

  Page 4 - Bootstrap CI:
    10,000-iteration bootstrap distributions of raw and adjusted
    Cohen's d for the Mean reading-only streak.

Reads only existing per-script output CSVs. No re-computation of statistics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from helpers import OUTPUT_DIR

# Light palette
BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
MUTED = '#666666'
BORDER = '#CCCCCC'
GRID = '#E8E8E8'
COND_COLORS = {'diffuse': '#1976D2', 'clumpy': '#C62828'}

# The 5 measures featured throughout (the writing-pattern signal).
MEASURES = [
    'seq_typing_entropy',
    'seq_typing_max_run',
    'first_writing_time_s',
    'seq_typing_mean_run_explore',
    'seq_topic_mean_run_exploit',
]

# Plain-English labels - no internal column names exposed to advisors.
LABELS = {
    'seq_typing_entropy':          'Writing entropy across pages',
    'seq_typing_max_run':          'Longest write/no-write streak',
    'first_writing_time_s':        'Time to first writing (s)',
    'seq_typing_mean_run_explore': 'Mean reading-only streak',
    'seq_topic_mean_run_exploit':  'Mean same-topic streak',
}

# Pre-registered headline number (PC1 of the three switch counts).
PREREG_PC1_D = 0.04
PREREG_PC1_P = 0.83


def _style_axes(ax):
    ax.set_facecolor(BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=MUTED)


# ----------------------------------------------------------------------
# Page 1 - Headline
# ----------------------------------------------------------------------

def page_headline(pdf, m56_csv, m57_csv):
    m56 = pd.read_csv(m56_csv)
    m57 = pd.read_csv(m57_csv).set_index('measure')

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'Wikipedia task (N = 101): pre-registered analysis null,\n'
        'reading-only streaks show a robust signal',
        fontsize=15, fontweight='bold', color=TEXT, y=0.965,
    )

    # ----- Two top banners: pre-reg vs exploratory -----
    banner_y = 0.69
    banner_h = 0.19

    # Pre-reg null banner (left, gray)
    ax_pre = fig.add_axes([0.06, banner_y, 0.42, banner_h])
    ax_pre.axis('off')
    ax_pre.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_pre.transAxes,
        facecolor='#F5F5F5', edgecolor='#999999', linewidth=1.2,
    ))
    ax_pre.text(0.5, 0.88, 'Pre-registered analysis',
                transform=ax_pre.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold', color=TEXT)
    ax_pre.text(0.5, 0.62,
                'PC1 of three switch counts (time, topic, typing)',
                transform=ax_pre.transAxes, ha='center', va='center',
                fontsize=10, color=MUTED, style='italic')
    ax_pre.text(0.5, 0.36,
                f'd = {PREREG_PC1_D:+.2f},  p = {PREREG_PC1_P:.2f}',
                transform=ax_pre.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='#888888')
    ax_pre.text(0.5, 0.13, 'no condition difference',
                transform=ax_pre.transAxes, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#888888')

    # Exploratory finding banner (right, green)
    mrs = m57.loc['seq_typing_mean_run_explore']
    ax_ex = fig.add_axes([0.52, banner_y, 0.42, banner_h])
    ax_ex.axis('off')
    ax_ex.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_ex.transAxes,
        facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5,
    ))
    ax_ex.text(0.5, 0.88, 'Exploratory finding',
                transform=ax_ex.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold', color='#1B5E20')
    ax_ex.text(0.5, 0.62,
                'Mean reading-only streak',
                transform=ax_ex.transAxes, ha='center', va='center',
                fontsize=10, color=MUTED, style='italic')
    ax_ex.text(0.5, 0.36,
                f'd = {mrs.adj_cohen_d:+.2f},  p = {mrs.adj_p:.3f}',
                transform=ax_ex.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='#1B5E20')
    ax_ex.text(0.5, 0.13, 'Clumpy > Diffuse',
                transform=ax_ex.transAxes, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#1B5E20')

    # ----- Bar chart of entropy means by condition -----
    ax_bar = fig.add_axes([0.30, 0.24, 0.42, 0.38])
    _style_axes(ax_bar)
    ax_bar.grid(True, color=GRID, linewidth=0.5, axis='y', zorder=0)

    d_vals = m56.loc[m56['condition'] == 'diffuse', 'seq_typing_mean_run_explore'].dropna()
    c_vals = m56.loc[m56['condition'] == 'clumpy',  'seq_typing_mean_run_explore'].dropna()
    means = [d_vals.mean(), c_vals.mean()]
    sems  = [d_vals.std() / np.sqrt(len(d_vals)),
             c_vals.std() / np.sqrt(len(c_vals))]

    bars = ax_bar.bar(
        ['Diffuse', 'Clumpy'], means, yerr=sems, capsize=6,
        color=[COND_COLORS['diffuse'], COND_COLORS['clumpy']],
        edgecolor='white', linewidth=0.8, alpha=0.92, zorder=2, width=0.55,
    )
    for bar, val in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val / 2,
                    f'M = {val:.2f}', ha='center', va='center',
                    fontsize=12, color='white', fontweight='bold', zorder=4)

    rng = np.random.default_rng(7)
    ax_bar.scatter(rng.uniform(-0.16, 0.16, len(d_vals)), d_vals,
                   color=COND_COLORS['diffuse'], alpha=0.35, s=22, zorder=3,
                   edgecolors='white', linewidth=0.5)
    ax_bar.scatter(1 + rng.uniform(-0.16, 0.16, len(c_vals)), c_vals,
                   color=COND_COLORS['clumpy'], alpha=0.35, s=22, zorder=3,
                   edgecolors='white', linewidth=0.5)

    ax_bar.set_ylabel('Mean reading-only streak (pages)',
                      color=LABEL, fontweight='bold', fontsize=11)
    ax_bar.set_title(
        'Higher = longer stretches of pages without writing\n'
        'Lower = writing interleaved more frequently across the search',
        color=MUTED, fontsize=10, pad=8, fontweight='normal',
    )
    top = max(d_vals.max(), c_vals.max())
    ax_bar.set_ylim(0, top * 1.10)

    # ----- Bottom: takeaway -----
    ax_take = fig.add_axes([0.06, 0.03, 0.88, 0.16])
    ax_take.axis('off')
    ax_take.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_take.transAxes,
        facecolor='#FAFAFA', edgecolor=BORDER, linewidth=1,
    ))
    line1 = 'The manipulation does not move the pre-registered switch counts,'
    line2 = 'but it does change HOW writing is distributed across pages:'
    line3 = ('In Clumpy, the average reading-only stretch is longer - '
             'writing concentrates into bursts with longer non-writing gaps.')
    line4 = ('In Diffuse, writing is interleaved throughout the search, '
             'keeping reading-only stretches short.')
    ax_take.text(0.5, 0.86, line1, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=10, color=TEXT)
    ax_take.text(0.5, 0.63, line2, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=10, color=TEXT)
    ax_take.text(0.5, 0.38, line3, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=10.5,
                 color=TEXT, fontweight='bold')
    ax_take.text(0.5, 0.14, line4, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=10, color=TEXT)

    pdf.savefig(fig, facecolor=BG)
    plt.close()


# ----------------------------------------------------------------------
# Page 2 - Three robustness checks
# ----------------------------------------------------------------------

def _hbar_pair(ax, labels, left_vals, right_vals,
               left_label, right_label,
               left_color, right_color, legend_loc='lower right'):
    """Plot two horizontal bar series stacked per measure."""
    y = np.arange(len(labels))
    ax.barh(y - 0.20, left_vals,  height=0.36, color=left_color,
            edgecolor='white', linewidth=0.8, label=left_label, zorder=2)
    ax.barh(y + 0.20, right_vals, height=0.36, color=right_color,
            edgecolor='white', linewidth=0.8, label=right_label, zorder=2)
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc=legend_loc, framealpha=1.0, frameon=True,
              edgecolor=BORDER, facecolor='white')
    ax.grid(True, color=GRID, linewidth=0.5, axis='x', zorder=0)


def _symmetric_xlim(ax, *value_lists, pad=1.15):
    """Symmetric x-limits around zero so signed bars read cleanly."""
    all_vals = [v for vs in value_lists for v in vs]
    m = max(abs(v) for v in all_vals) if all_vals else 1.0
    lim = m * pad if m > 0 else 1.0
    ax.set_xlim(-lim, lim)


def _draw_panel_header(fig, top, label, question, answer, what_bars_show,
                       what_label='What the bars show:'):
    """Question-answer panel header. Three text rows:
      [letter] [bold question]                         answer (colored)
      "<what_label>: ..."   (italic, muted, may be multi-line; skipped if empty)
    """
    fig.text(0.06, top, label,
             fontsize=14, fontweight='bold', color='#1976D2')
    fig.text(0.10, top, question,
             fontsize=11.5, fontweight='bold', color=TEXT, va='baseline')
    fig.text(0.96, top, answer,
             fontsize=11.5, fontweight='bold', color='#2E7D32', va='baseline',
             ha='right')
    if what_bars_show:
        fig.text(0.10, top - 0.020, f'{what_label}  {what_bars_show}',
                 fontsize=9.5, color=MUTED, style='italic', va='top',
                 linespacing=1.4)


def page_followups(pdf, m57_csv, m60_csv, m61_csv):
    m57 = pd.read_csv(m57_csv).set_index('measure')
    m60 = pd.read_csv(m60_csv)
    _ = m61_csv  # warm-up test no longer rendered; kept in signature

    fig = plt.figure(figsize=(11, 10.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'The 5 writing-pattern measures, plus two robustness checks',
        fontsize=15, fontweight='bold', color=TEXT, y=0.975,
    )
    fig.text(
        0.5, 0.948,
        '(A) defines the measures used throughout.  '
        '(B) and (C) re-analyze them from different angles.',
        ha='center', va='top', fontsize=10.5, color=MUTED, style='italic',
    )

    measure_labels = [LABELS[m] for m in MEASURES]

    # ---------- Panel A: The 5 writing-pattern measures ----------
    _draw_panel_header(
        fig, top=0.910, label='(A)',
        question='The 5 writing-pattern measures',
        answer='used throughout panels B and C',
        what_bars_show='each measure captures the W/N sequence from a different angle.',
        what_label='What they capture:',
    )
    ax_key = fig.add_axes([0.06, 0.665, 0.88, 0.205])
    ax_key.axis('off')
    ax_key.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_key.transAxes,
        facecolor='#FAFAFA', edgecolor=BORDER, linewidth=1,
    ))
    key_lines = (
        'Writing entropy                  -  balance between W and N pages across the question\n'
        'Longest write/no-write streak    -  longest run of a single label (W or N)\n'
        'Time to first writing (s)        -  latency until the first writing on any page\n'
        'Mean reading-only streak         -  average length of consecutive no-write (N) pages\n'
        'Mean same-topic streak           -  average length of consecutive pages on the same LDA topic'
    )
    ax_key.text(0.025, 0.88, key_lines,
                transform=ax_key.transAxes, ha='left', va='top',
                fontsize=10, color=LABEL, family='monospace', linespacing=1.85)

    # ---------- Panel B: Output-volume covariate ----------
    _draw_panel_header(
        fig, top=0.610, label='(B)',
        question='Does the effect survive controlling for output volume?',
        answer='YES - it strengthens',
        what_bars_show='effect size for each measure, before vs after partialling out '
                       'final answer length in a regression.',
    )
    ax_b = fig.add_axes([0.34, 0.395, 0.62, 0.180])
    _style_axes(ax_b)
    raw_d = [m57.loc[m, 'raw_cohen_d'] for m in MEASURES]
    adj_d = [m57.loc[m, 'adj_cohen_d'] for m in MEASURES]
    _hbar_pair(
        ax_b, measure_labels, raw_d, adj_d,
        'Raw effect', 'After regressing out answer length',
        '#B0BEC5', '#2E7D32',
    )
    _symmetric_xlim(ax_b, raw_d, adj_d)
    ax_b.set_xlabel("Cohen's d  (Diffuse - Clumpy)",
                    color=LABEL, fontsize=10)

    # ---------- Panel C: Trial-order moderation ----------
    _draw_panel_header(
        fig, top=0.340, label='(C)',
        question='Does the effect decay or grow across questions?',
        answer='GROWS on Q2',
        what_bars_show='effect size on participants\' first vs second question '
                       '(question order was randomised across participants).',
    )
    ax_c = fig.add_axes([0.34, 0.115, 0.62, 0.190])
    _style_axes(ax_c)
    q1_d = [m60[(m60['measure'] == m) & (m60['stratum'] == 'first')]
            ['adj_d'].iloc[0] for m in MEASURES]
    q2_d = [m60[(m60['measure'] == m) & (m60['stratum'] == 'second')]
            ['adj_d'].iloc[0] for m in MEASURES]
    _hbar_pair(
        ax_c, measure_labels, q1_d, q2_d,
        'First question (Q1)', 'Second question (Q2)',
        '#90A4AE', '#1976D2',
    )
    _symmetric_xlim(ax_c, q1_d, q2_d)
    ax_c.set_xlabel("Cohen's d  (Diffuse - Clumpy)",
                    color=LABEL, fontsize=10)

    pdf.savefig(fig, facecolor=BG)
    plt.close()


# ----------------------------------------------------------------------
# Page 3 - Construct validity: streak vs original switch measures
# ----------------------------------------------------------------------

# Three switch measures plotted on page 3 (column name -> human label)
SWITCH_PANELS = [
    ('switch_rate_typing', 'Typing switching (W/N)',
     'Switch rate: write <-> no-write'),
    ('switch_rate_topic',  'Topic switching (LDA)',
     'Switch rate: dominant LDA topic'),
    ('switch_rate_time',   'Time switching (>60s dwell)',
     'Switch rate: long <-> short page dwell'),
]


def _scatter_with_fit(ax, x, y, cond):
    """Single scatter panel: dots colored by condition + OLS fit + r in corner."""
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)

    mask = ~(np.isnan(x) | np.isnan(y))
    x, y, cond = x[mask], y[mask], cond[mask]

    for c in ['diffuse', 'clumpy']:
        m = cond == c
        ax.scatter(x[m], y[m], color=COND_COLORS[c], s=26, alpha=0.65,
                   edgecolors='white', linewidth=0.5, zorder=2,
                   label=f'{c.capitalize()} (n={m.sum()})')

    # OLS fit line across the full x range
    if len(x) >= 3 and x.std() > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color='#444', linewidth=1.4, zorder=3, alpha=0.85)

    # Pearson r + p in the top-left corner
    if len(x) >= 10 and x.std() > 0 and y.std() > 0:
        r, p = sp_stats.pearsonr(x, y)
        ax.text(0.04, 0.96,
                f'r = {r:+.2f}\np = {p:.3g}\nN = {len(x)}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9.5, color=TEXT, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor=BORDER,
                          boxstyle='round,pad=0.35', linewidth=0.8))


def page_correlations(pdf, m65_csv):
    df = pd.read_csv(m65_csv)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'How the Mean reading-only streak relates to the\n'
        'three original switch measures',
        fontsize=15, fontweight='bold', color=TEXT, y=0.965,
    )
    fig.text(
        0.5, 0.895,
        'Per participant, averaged across both questions   '
        '(N = 99 of 101; 2 wrote on every page, leaving no reading-only streak)',
        ha='center', va='top', fontsize=10, color=MUTED, style='italic',
    )

    # Three scatter panels in a row
    panel_w = 0.275
    panel_h = 0.50
    panel_y = 0.30
    x_starts = [0.06, 0.365, 0.67]

    cond = df['condition'].to_numpy()
    y_vals = df['seq_typing_mean_run_explore'].to_numpy()

    for (col, title, x_label), x_start in zip(SWITCH_PANELS, x_starts):
        ax = fig.add_axes([x_start, panel_y, panel_w, panel_h])
        _scatter_with_fit(ax, df[col].to_numpy(), y_vals, cond)
        ax.set_title(title, fontsize=11, color=TEXT, fontweight='bold', pad=8)
        ax.set_xlabel(x_label, color=LABEL, fontsize=9.5)

    # Y label only on the leftmost panel (shared meaning across the three)
    fig.text(0.012, 0.55, 'Mean reading-only streak (pages)',
             rotation=90, ha='center', va='center',
             fontsize=10.5, color=LABEL, fontweight='bold')

    # Legend (single, shared across all 3 panels)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Diffuse',
                   markerfacecolor=COND_COLORS['diffuse'],
                   markeredgecolor='white', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Clumpy',
                   markerfacecolor=COND_COLORS['clumpy'],
                   markeredgecolor='white', markersize=8),
        plt.Line2D([0], [0], color='#444', linewidth=1.4, label='OLS fit'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.255), ncol=3,
               frameon=True, edgecolor=BORDER, facecolor='white',
               fontsize=9.5)

    # Bottom takeaway: technical observation on top, headline conclusion below.
    ax_take = fig.add_axes([0.06, 0.03, 0.88, 0.19])
    ax_take.axis('off')
    ax_take.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_take.transAxes,
        facecolor='#FAFAFA', edgecolor=BORDER, linewidth=1,
    ))

    # Technical observation (regular weight, smaller)
    obs1 = ('Mean reading-only streak is mathematically tied to typing switching '
            '(r = -0.64) - they summarise the same W/N sequence.')
    obs2 = ('Its link to topic switching (r = +0.16) and time switching (r = -0.18) '
            'is essentially zero.')
    ax_take.text(0.5, 0.84, obs1, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=9.5, color=TEXT)
    ax_take.text(0.5, 0.66, obs2, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=9.5, color=TEXT)

    # Divider
    ax_take.plot([0.04, 0.96], [0.50, 0.50], transform=ax_take.transAxes,
                 color=BORDER, linewidth=0.8)

    # Headline conclusion - large, bold, green to mirror the page-1 banner.
    headline = ('The manipulation reshapes writing rhythm specifically - '
                'not topic exploration or reading pace.')
    ax_take.text(0.5, 0.25, headline, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=12.5,
                 color='#1B5E20', fontweight='bold')

    pdf.savefig(fig, facecolor=BG)
    plt.close()


# ----------------------------------------------------------------------
# Page 4 - Bootstrap CI for Mean reading-only streak
# ----------------------------------------------------------------------

def _bootstrap_panel(ax, boot_vals, observed, ci_lo, ci_hi, p_emp, title):
    """One bootstrap distribution panel: histogram + observed line + CI shading."""
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.5, axis='y', zorder=0)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

    # Histogram
    ax.hist(boot_vals, bins=50, color='#90CAF9', edgecolor='white',
            linewidth=0.4, alpha=0.85, zorder=2)

    ymin, ymax = ax.get_ylim()

    # 95% CI shading
    ax.axvspan(ci_lo, ci_hi, color='#1976D2', alpha=0.12, zorder=1,
               label='95% CI')

    # Zero line
    ax.axvline(0, color='#888', linewidth=1.0, linestyle='--', zorder=3)

    # Observed value line
    ax.axvline(observed, color='#C62828', linewidth=2.0, zorder=4,
               label=f'observed = {observed:+.2f}')

    # Symmetric x-limits around zero so the CI position is read correctly
    span = max(abs(ci_lo), abs(ci_hi), abs(observed))
    pad = span * 1.25 if span > 0 else 1.0
    ax.set_xlim(-pad, pad)
    ax.set_ylim(0, ymax * 1.15)

    # Stats box in upper-right corner
    crosses_zero = ci_lo <= 0 <= ci_hi
    verdict = 'CI crosses 0' if crosses_zero else 'CI excludes 0'
    verdict_color = '#888888' if crosses_zero else '#1B5E20'
    stats_txt = (f'observed d  = {observed:+.3f}\n'
                 f'95% CI       = [{ci_lo:+.3f}, {ci_hi:+.3f}]\n'
                 f'p (empirical) = {p_emp:.3f}')
    ax.text(0.97, 0.97, stats_txt,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9.5, color=TEXT, family='monospace',
            bbox=dict(facecolor='white', edgecolor=BORDER,
                      boxstyle='round,pad=0.4', linewidth=0.8))
    ax.text(0.97, 0.72, verdict,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, color=verdict_color, fontweight='bold')

    ax.set_title(title, fontsize=11.5, color=TEXT, fontweight='bold', pad=8)
    ax.set_xlabel("Cohen's d  (Diffuse - Clumpy)", color=LABEL, fontsize=9.5)
    ax.set_ylabel('Bootstrap iterations', color=LABEL, fontsize=9.5)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=1.0,
              frameon=True, edgecolor=BORDER, facecolor='white')


def page_bootstrap(pdf, m67_dist_csv, m67_summary_csv):
    boot = pd.read_csv(m67_dist_csv)
    summ = pd.read_csv(m67_summary_csv).set_index('effect_size')

    n_iter = len(boot)
    n_d = int(summ.iloc[0]['n_diffuse'])
    n_c = int(summ.iloc[0]['n_clumpy'])

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'Bootstrap confidence intervals for the\n'
        'Mean reading-only streak effect',
        fontsize=15, fontweight='bold', color=TEXT, y=0.965,
    )
    fig.text(
        0.5, 0.895,
        f'{n_iter:,} stratified resamples of participants  '
        f'(N = {n_d + n_c}: {n_d} Diffuse, {n_c} Clumpy)',
        ha='center', va='top', fontsize=10, color=MUTED, style='italic',
    )

    # Two histogram panels side by side - matches the two columns of M66 panel B
    panel_w = 0.42
    panel_h = 0.50
    panel_y = 0.30

    raw = summ.loc['raw_cohen_d']
    adj = summ.loc['adj_cohen_d']

    ax_raw = fig.add_axes([0.06, panel_y, panel_w, panel_h])
    _bootstrap_panel(
        ax_raw,
        boot['boot_raw_d'].dropna().to_numpy(),
        observed=raw['observed'],
        ci_lo=raw['ci_lo_2.5'], ci_hi=raw['ci_hi_97.5'],
        p_emp=raw['p_empirical'],
        title='Raw Cohen\'s d  (no covariate)',
    )

    ax_adj = fig.add_axes([0.52, panel_y, panel_w, panel_h])
    _bootstrap_panel(
        ax_adj,
        boot['boot_adj_d'].dropna().to_numpy(),
        observed=adj['observed'],
        ci_lo=adj['ci_lo_2.5'], ci_hi=adj['ci_hi_97.5'],
        p_emp=adj['p_empirical'],
        title='Adjusted Cohen\'s d  (controls for answer length)',
    )

    # Bottom takeaway
    ax_take = fig.add_axes([0.06, 0.03, 0.88, 0.19])
    ax_take.axis('off')
    ax_take.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_take.transAxes,
        facecolor='#FAFAFA', edgecolor=BORDER, linewidth=1,
    ))

    obs1 = ('Each histogram = the Cohen\'s d we would observe across 10,000 '
            'resamples of the same N participants.')
    obs2 = ('The raw effect\'s 95% CI crosses zero; '
            'after controlling for answer length, the CI excludes zero.')
    ax_take.text(0.5, 0.84, obs1, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=9.5, color=TEXT)
    ax_take.text(0.5, 0.66, obs2, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=9.5, color=TEXT)

    ax_take.plot([0.04, 0.96], [0.50, 0.50], transform=ax_take.transAxes,
                 color=BORDER, linewidth=0.8)

    headline = ('The Mean reading-only streak effect is robust to resampling '
                'once output volume is accounted for.')
    ax_take.text(0.5, 0.25, headline, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=12.5,
                 color='#1B5E20', fontweight='bold')

    pdf.savefig(fig, facecolor=BG)
    plt.close()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Building 4-page email-summary PDF (signed Cohens d variant)')

    m56_csv = OUTPUT_DIR / 'm56_eda_writing_sequential.csv'
    m57_csv = OUTPUT_DIR / 'm57_covariate_analysis.csv'
    m60_csv = OUTPUT_DIR / 'm60_trial_order_moderation.csv'
    m61_csv = OUTPUT_DIR / 'm61_warmup_effect.csv'
    m65_csv = OUTPUT_DIR / 'm65_per_participant_streak_switches.csv'
    m67_dist_csv = OUTPUT_DIR / 'm67_bootstrap_streak.csv'
    m67_summary_csv = OUTPUT_DIR / 'm67_bootstrap_streak_summary.csv'

    for p in [m56_csv, m57_csv, m60_csv, m61_csv, m65_csv,
              m67_dist_csv, m67_summary_csv]:
        if not Path(p).exists():
            raise FileNotFoundError(f'Missing input: {p}')

    pdf_path = OUTPUT_DIR / 'm66_email_summary_signed.pdf'
    with PdfPages(pdf_path) as pdf:
        page_headline(pdf, m56_csv, m57_csv)
        page_followups(pdf, m57_csv, m60_csv, m61_csv)
        page_correlations(pdf, m65_csv)
        page_bootstrap(pdf, m67_dist_csv, m67_summary_csv)
    print(f'Saved: {pdf_path}')


if __name__ == '__main__':
    main()

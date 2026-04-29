#!/usr/bin/env python3
"""
Email-summary PDF (2 pages)
===========================
Two clean pages for the advisor email:

  Page 1 - Headline:
    Pre-registered analysis null vs writing-pattern entropy signal.
    Bar chart of the entropy means by condition with individual dots.

  Page 2 - Three robustness checks:
    (A) Output-volume covariate
    (B) Effect across trial order (Q1 vs Q2)
    (C) Warm-up test on Q1 (high vs low engagement)

Reads only existing per-script output CSVs. No re-computation of statistics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
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
        'writing-pattern entropy shows a robust signal',
        fontsize=15, fontweight='bold', color=TEXT, y=0.965,
    )

    # ----- Two top banners: pre-reg vs exploratory -----
    # Banners use 2-line stat blocks so the green banner's longer text
    # ("Diffuse > Clumpy") never overflows the box.
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
    ent = m57.loc['seq_typing_entropy']
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
                'Writing entropy across pages',
                transform=ax_ex.transAxes, ha='center', va='center',
                fontsize=10, color=MUTED, style='italic')
    ax_ex.text(0.5, 0.36,
                f'd = +{ent.adj_cohen_d:.2f},  p = {ent.adj_p:.3f}',
                transform=ax_ex.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='#1B5E20')
    ax_ex.text(0.5, 0.13, 'Diffuse > Clumpy',
                transform=ax_ex.transAxes, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#1B5E20')

    # ----- Bar chart of entropy means by condition -----
    ax_bar = fig.add_axes([0.30, 0.20, 0.42, 0.42])
    _style_axes(ax_bar)
    ax_bar.grid(True, color=GRID, linewidth=0.5, axis='y', zorder=0)

    d_ent = m56.loc[m56['condition'] == 'diffuse', 'seq_typing_entropy'].dropna()
    c_ent = m56.loc[m56['condition'] == 'clumpy',  'seq_typing_entropy'].dropna()
    means = [d_ent.mean(), c_ent.mean()]
    sems  = [d_ent.std() / np.sqrt(len(d_ent)),
             c_ent.std() / np.sqrt(len(c_ent))]

    bars = ax_bar.bar(
        ['Diffuse', 'Clumpy'], means, yerr=sems, capsize=6,
        color=[COND_COLORS['diffuse'], COND_COLORS['clumpy']],
        edgecolor='white', linewidth=0.8, alpha=0.92, zorder=2, width=0.55,
    )
    # Mean labels go INSIDE the bar (white, bold) - putting them above
    # the bar collides with the individual-participant scatter dots.
    for bar, val in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val / 2,
                    f'M = {val:.2f}', ha='center', va='center',
                    fontsize=12, color='white', fontweight='bold', zorder=4)

    rng = np.random.default_rng(7)
    ax_bar.scatter(rng.uniform(-0.16, 0.16, len(d_ent)), d_ent,
                   color=COND_COLORS['diffuse'], alpha=0.35, s=22, zorder=3,
                   edgecolors='white', linewidth=0.5)
    ax_bar.scatter(1 + rng.uniform(-0.16, 0.16, len(c_ent)), c_ent,
                   color=COND_COLORS['clumpy'], alpha=0.35, s=22, zorder=3,
                   edgecolors='white', linewidth=0.5)

    ax_bar.set_ylabel('Writing entropy', color=LABEL, fontweight='bold', fontsize=11)
    ax_bar.set_title(
        'Higher = writing evenly spread across pages\n'
        'Lower = writing concentrated in long streaks',
        color=MUTED, fontsize=10, pad=8, fontweight='normal',
    )
    ax_bar.set_ylim(0, 1.08)

    # ----- Bottom: takeaway -----
    # Hard-wrap the takeaway text into lines that fit comfortably inside
    # the box; matplotlib's wrap=True does not always honour the axes width.
    ax_take = fig.add_axes([0.06, 0.04, 0.88, 0.11])
    ax_take.axis('off')
    ax_take.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_take.transAxes,
        facecolor='#FAFAFA', edgecolor=BORDER, linewidth=1,
    ))
    take = (
        'The manipulation does not move the pre-registered switch counts,\n'
        'but it does change HOW writing is distributed across pages:\n'
        'Diffuse integrates writing throughout the search; '
        'Clumpy accumulates it into long streaks.'
    )
    ax_take.text(0.5, 0.5, take, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=10.5,
                 color=TEXT, linespacing=1.5)

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
    # White-background framed legend so it never disappears into a bar.
    ax.legend(fontsize=9, loc=legend_loc, framealpha=1.0, frameon=True,
              edgecolor=BORDER, facecolor='white')
    ax.grid(True, color=GRID, linewidth=0.5, axis='x', zorder=0)


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
    raw_d = [abs(m57.loc[m, 'raw_cohen_d']) for m in MEASURES]
    adj_d = [abs(m57.loc[m, 'adj_cohen_d']) for m in MEASURES]
    _hbar_pair(
        ax_b, measure_labels, raw_d, adj_d,
        'Raw effect', 'After regressing out answer length',
        '#B0BEC5', '#2E7D32',
    )
    ax_b.set_xlabel("|Cohen's d|", color=LABEL, fontsize=10)

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
    q1_d = [abs(m60[(m60['measure'] == m) & (m60['stratum'] == 'first')]
                ['adj_d'].iloc[0]) for m in MEASURES]
    q2_d = [abs(m60[(m60['measure'] == m) & (m60['stratum'] == 'second')]
                ['adj_d'].iloc[0]) for m in MEASURES]
    _hbar_pair(
        ax_c, measure_labels, q1_d, q2_d,
        'First question (Q1)', 'Second question (Q2)',
        '#90A4AE', '#1976D2',
    )
    ax_c.set_xlabel("|Cohen's d|", color=LABEL, fontsize=10)

    pdf.savefig(fig, facecolor=BG)
    plt.close()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Building 2-page email-summary PDF')

    m56_csv = OUTPUT_DIR / 'm56_eda_writing_sequential.csv'
    m57_csv = OUTPUT_DIR / 'm57_covariate_analysis.csv'
    m60_csv = OUTPUT_DIR / 'm60_trial_order_moderation.csv'
    m61_csv = OUTPUT_DIR / 'm61_warmup_effect.csv'

    for p in [m56_csv, m57_csv, m60_csv, m61_csv]:
        if not Path(p).exists():
            raise FileNotFoundError(f'Missing input: {p}')

    pdf_path = OUTPUT_DIR / 'm63_email_summary.pdf'
    with PdfPages(pdf_path) as pdf:
        page_headline(pdf, m56_csv, m57_csv)
        page_followups(pdf, m57_csv, m60_csv, m61_csv)
    print(f'Saved: {pdf_path}')


if __name__ == '__main__':
    main()

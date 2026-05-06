#!/usr/bin/env python3
"""
Email-summary PDF (1 page) - raincloud variant of M66 page 1
============================================================
Single-page raincloud (half violin + box + jittered points) for the
Mean reading-only streak by condition. Drop-in replacement for the
M66 bar-chart headline page.

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

MEASURES = [
    'seq_typing_entropy',
    'seq_typing_max_run',
    'first_writing_time_s',
    'seq_typing_mean_run_explore',
    'seq_topic_mean_run_exploit',
]

LABELS = {
    'seq_typing_entropy':          'Writing entropy across pages',
    'seq_typing_max_run':          'Longest write/no-write streak',
    'first_writing_time_s':        'Time to first writing (s)',
    'seq_typing_mean_run_explore': 'Mean reading-only streak',
    'seq_topic_mean_run_exploit':  'Mean same-topic streak',
}

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
# Raincloud helper - the key difference vs M66
# ----------------------------------------------------------------------

def _raincloud_panel(ax, groups, group_labels, group_colors,
                     ylabel, title=None, show_mean_label=True,
                     mean_label_fmt='M = {:.2f}', rng_seed=7):
    """Two-condition raincloud: half violin (right) + box + jittered scatter (left).

    Each group gets one x position. The violin density estimate is drawn on
    the right half of that position; a thin boxplot sits at the position; the
    raw points are jittered on the left half.
    """
    _style_axes(ax)
    ax.grid(True, color=GRID, linewidth=0.5, axis='y', zorder=0)

    rng = np.random.default_rng(rng_seed)
    positions = np.arange(len(groups), dtype=float)

    # ----- half violins on the right -----
    parts = ax.violinplot(
        groups, positions=positions, widths=0.85, showmeans=False,
        showmedians=False, showextrema=False,
    )
    for body, pos, color in zip(parts['bodies'], positions, group_colors):
        verts = body.get_paths()[0].vertices
        # keep only the right half
        verts[:, 0] = np.clip(verts[:, 0], pos, pos + 0.42)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.30)
        body.set_zorder(2)

    # ----- thin boxplots at the centre -----
    bp = ax.boxplot(
        groups, positions=positions, widths=0.10, patch_artist=True,
        showfliers=False, zorder=4,
        medianprops=dict(color='white', linewidth=1.6),
        whiskerprops=dict(color='#333333', linewidth=1.0),
        capprops=dict(color='#333333', linewidth=1.0),
    )
    for box, color in zip(bp['boxes'], group_colors):
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.85)

    # ----- jittered scatter on the left -----
    for vals, pos, color in zip(groups, positions, group_colors):
        jitter = rng.uniform(-0.30, -0.06, size=len(vals))
        ax.scatter(pos + jitter, vals,
                   color=color, alpha=0.55, s=22, zorder=3,
                   edgecolors='white', linewidth=0.5)

    # ----- mean markers + labels -----
    means = [np.mean(g) for g in groups]
    for pos, m, color in zip(positions, means, group_colors):
        ax.scatter([pos], [m], marker='D', s=42,
                   facecolor='white', edgecolor=color, linewidth=1.6,
                   zorder=5)
        if show_mean_label:
            ax.text(pos + 0.50, m, mean_label_fmt.format(m),
                    ha='left', va='center', fontsize=10.5,
                    color=color, fontweight='bold')

    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, fontsize=11, color=TEXT)
    ax.set_xlim(-0.6, len(groups) - 1 + 0.95)
    ax.set_ylabel(ylabel, color=LABEL, fontweight='bold', fontsize=11)
    if title is not None:
        ax.set_title(title, color=MUTED, fontsize=10, pad=8)


# ----------------------------------------------------------------------
# Page 1 - Headline (raincloud)
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

    # ----- Raincloud of Mean reading-only streak -----
    ax_rc = fig.add_axes([0.28, 0.22, 0.46, 0.40])
    d_vals = m56.loc[m56['condition'] == 'diffuse',
                     'seq_typing_mean_run_explore'].dropna().to_numpy()
    c_vals = m56.loc[m56['condition'] == 'clumpy',
                     'seq_typing_mean_run_explore'].dropna().to_numpy()
    _raincloud_panel(
        ax_rc,
        groups=[d_vals, c_vals],
        group_labels=['Diffuse', 'Clumpy'],
        group_colors=[COND_COLORS['diffuse'], COND_COLORS['clumpy']],
        ylabel='Mean reading-only streak (pages)',
        title=('Higher = longer stretches of pages without writing\n'
               'Lower = writing interleaved more frequently across the search'),
    )

    # ----- Bottom takeaway -----
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
# main
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Building 1-page raincloud headline (M69)')

    m56_csv = OUTPUT_DIR / 'm56_eda_writing_sequential.csv'
    m57_csv = OUTPUT_DIR / 'm57_covariate_analysis.csv'

    for p in [m56_csv, m57_csv]:
        if not Path(p).exists():
            raise FileNotFoundError(f'Missing input: {p}')

    pdf_path = OUTPUT_DIR / 'm69_email_summary_violin.pdf'
    with PdfPages(pdf_path) as pdf:
        page_headline(pdf, m56_csv, m57_csv)
    print(f'Saved: {pdf_path}')


if __name__ == '__main__':
    main()

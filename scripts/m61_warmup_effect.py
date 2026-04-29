#!/usr/bin/env python3
"""
M61: Does the warm-up hypothesis explain the Q2 > Q1 pattern in M60?
====================================================================
Hypothesis: Q1 is noisy because participants are still learning the Wikipedia
interface (calibrating effort, understanding what makes a "good answer").
Once calibrated, the condition signal emerges (this is what M60 showed:
stronger effect on Q2).

Test: split participants by Q1 engagement (proxy for "how quickly did they
calibrate"). If participants who explored more pages on Q1 already show the
condition effect on Q1 (before high-engagement low-engagement split),
this supports the warm-up explanation.

Engagement proxy: n_pages_Q1 (more pages = more confident interaction with
the interface).

Outputs: output/m61_warmup_effect.{csv,pdf}
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
from m56_eda_writing_sequential import build_question_df
from m57_covariate_analysis import ols_fit, cohens_d, COVARIATE
from m60_trial_order_moderation import attach_trial_position, analyze_subset

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'

GROUP_COLORS = {
    'high_engagement': '#2E7D32',  # green
    'low_engagement':  '#C62828',  # red
    'all':             '#1976D2',
}

TARGET_MEASURES = [
    'seq_typing_entropy',
    'seq_typing_max_run',
    'first_writing_time_s',
    'seq_typing_mean_run_explore',
    'seq_topic_mean_run_exploit',
]


def build_engagement_split(qdf_clean):
    """For each participant, compute Q1 engagement (n_pages on first question).

    Returns DataFrame with cols: participant_id, q1_n_pages, engagement_group.
    """
    q1 = qdf_clean[qdf_clean['trial_position'] == 1][['participant_id', 'n_pages']]
    q1 = q1.rename(columns={'n_pages': 'q1_n_pages'})
    median = q1['q1_n_pages'].median()
    q1['engagement_group'] = q1['q1_n_pages'].apply(
        lambda x: 'high_engagement' if x > median else 'low_engagement')
    return q1, median


def run_analysis_per_group(qdf_clean, q1_engagement):
    """Run M57-style analysis on Q1-only data, split by Q1 engagement."""
    # Get Q1 only
    q1_only = qdf_clean[qdf_clean['trial_position'] == 1].copy()
    q1_only = q1_only.merge(q1_engagement[['participant_id', 'engagement_group']],
                            on='participant_id', how='left')

    rows = []
    for grp in ['all', 'high_engagement', 'low_engagement']:
        if grp == 'all':
            sub = q1_only
        else:
            sub = q1_only[q1_only['engagement_group'] == grp]
        for m in TARGET_MEASURES:
            r = analyze_subset(sub, m)
            if r is None:
                continue
            r['group'] = grp
            n_d = (sub['condition'] == 'diffuse').sum()
            n_c = (sub['condition'] == 'clumpy').sum()
            r['n_total'] = n_d + n_c
            rows.append(r)
    return pd.DataFrame(rows)


def plot_forest_by_engagement(ax, results_df):
    ax.set_facecolor(BG)
    measures = TARGET_MEASURES
    n = len(measures)
    offset = {'high_engagement': -0.22, 'low_engagement': +0.22, 'all': 0.0}
    color = {'high_engagement': GROUP_COLORS['high_engagement'],
             'low_engagement': GROUP_COLORS['low_engagement'],
             'all': GROUP_COLORS['all']}
    marker = {'high_engagement': 'o', 'low_engagement': 'o', 'all': 'D'}
    size = {'high_engagement': 80, 'low_engagement': 80, 'all': 130}

    for i, m in enumerate(measures):
        for grp in ['all', 'high_engagement', 'low_engagement']:
            r = results_df[(results_df['measure'] == m) & (results_df['group'] == grp)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            val = r['adj_d']
            p = r['adj_p']
            y = i + offset[grp]
            ax.scatter(val, y, s=size[grp], color=color[grp],
                       edgecolors='#000' if grp == 'all' else color[grp],
                       linewidth=1.4, marker=marker[grp], zorder=4, alpha=0.95)
            sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            ax.annotate(f'{val:+.2f}{sig}', (val, y), xytext=(7, 0),
                        textcoords='offset points', fontsize=8,
                        color=color[grp], va='center', fontweight='bold')

    ax.axvline(0, color=BORDER, linewidth=1)
    for thresh in [-0.5, -0.2, 0.2, 0.5]:
        ax.axvline(thresh, color=GRID, linewidth=0.5, linestyle=':')

    ax.set_yticks(range(n))
    ax.set_yticklabels(measures, fontsize=9)
    ax.set_xlabel('Q1 adjusted Cohen\'s d (Diffuse - Clumpy)',
                  color=LABEL, fontweight='bold')
    ax.set_title('Warm-up test: Q1 effect by Q1 engagement (n_pages median split)',
                 color=TEXT, fontweight='bold', fontsize=13, pad=10)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.4, axis='x', zorder=0)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    ax.set_xlim(-1.2, 1.2)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=GROUP_COLORS['all'],
               markeredgecolor='#000', markersize=10, label='All Q1 (M60 baseline)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS['high_engagement'],
               markersize=8, label='High engagement Q1 (more pages)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS['low_engagement'],
               markersize=8, label='Low engagement Q1 (fewer pages)'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='lower right', framealpha=0.9)


def make_summary_table_page(results_df, median_pages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M61: Warm-up Test - Q1 Effect by Q1 Engagement',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')

    lines = []
    lines.append(f'Engagement split: median Q1 n_pages = {median_pages:.1f}')
    lines.append('  high_engagement = participants with more than median pages on Q1')
    lines.append('  low_engagement  = participants with at-or-below-median pages on Q1')
    lines.append('')
    lines.append('Hypothesis tested: if warm-up explains M60 (Q2 > Q1), then participants')
    lines.append('who engaged more with Q1 should already show the condition effect on Q1.')
    lines.append('')

    lines.append(f'{"Measure":<32} {"Group":<18} {"N":>4} '
                 f'{"Raw d":>7} {"Raw p":>7} {"Adj d":>7} {"Adj p":>7}')
    lines.append('-' * 92)
    for m in TARGET_MEASURES:
        for grp in ['all', 'high_engagement', 'low_engagement']:
            r = results_df[(results_df['measure'] == m) & (results_df['group'] == grp)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            lines.append(
                f'{m if grp == "all" else "":<32} '
                f'{grp:<18} {r["n_total"]:>4} '
                f'{r["raw_d"]:>+7.2f} {r["raw_p"]:>7.3f} '
                f'{r["adj_d"]:>+7.2f} {r["adj_p"]:>7.3f}'
            )
        lines.append('')

    lines.append('Verdict pattern:')
    lines.append('  If high-eng Q1 effect >> low-eng Q1 effect AND high-eng matches M60-Q2:')
    lines.append('    -> warm-up hypothesis SUPPORTED')
    lines.append('  If both groups show similar Q1 effect:')
    lines.append('    -> warm-up hypothesis NOT supported (effect just mounts over time)')
    lines.append('  If low-eng Q1 effect > high-eng Q1 effect:')
    lines.append('    -> contrary - effect might be triggered by under-engagement')

    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes, fontsize=8.5,
            family='monospace', va='top', color=TEXT, linespacing=1.4)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M61] Warm-up effect analysis')
    print('=' * 60)

    print('\n--- Loading and building question-level features ---')
    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    qdf = build_question_df(trials)
    qdf = attach_trial_position(qdf, trials)
    excluded_mask = qdf['excluded_pages'] | qdf['excluded_idle']
    qdf_clean = qdf[~excluded_mask].copy()
    M52_OUTLIERS = {26, 79, 134}
    qdf_clean = qdf_clean[~qdf_clean['participant_id'].isin(M52_OUTLIERS)].copy()
    print(f'  After exclusions: {len(qdf_clean)} questions')

    print('\n--- Computing engagement split (Q1 n_pages median) ---')
    q1_engagement, median_pages = build_engagement_split(qdf_clean)
    print(f'  Median Q1 n_pages = {median_pages:.1f}')
    print(f'  N participants with Q1 data: {len(q1_engagement)}')
    print('  Group sizes:')
    print(q1_engagement['engagement_group'].value_counts().to_string())

    # Sanity: are conditions balanced across engagement groups?
    cond_per_pid = qdf_clean.drop_duplicates('participant_id')[['participant_id', 'condition']]
    merged = q1_engagement.merge(cond_per_pid, on='participant_id')
    print('\n  Condition x engagement crosstab:')
    print(pd.crosstab(merged['condition'], merged['engagement_group']))

    print('\n--- Running stratified Q1 analysis ---')
    results_df = run_analysis_per_group(qdf_clean, q1_engagement)

    # Print
    print(f'\n{"Measure":<32} {"Group":<18} {"adj d":>7} {"adj p":>7}')
    print('-' * 70)
    for m in TARGET_MEASURES:
        for grp in ['all', 'high_engagement', 'low_engagement']:
            r = results_df[(results_df['measure'] == m) & (results_df['group'] == grp)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            print(f'{m if grp == "all" else "":<32} {grp:<18} '
                  f'{r["adj_d"]:>+7.2f} {r["adj_p"]:>7.3f}')
        print()

    # Verdict
    print('=' * 60)
    print('VERDICT')
    print('=' * 60)
    for m in TARGET_MEASURES:
        high = results_df[(results_df['measure'] == m) & (results_df['group'] == 'high_engagement')]
        low = results_df[(results_df['measure'] == m) & (results_df['group'] == 'low_engagement')]
        if len(high) == 0 or len(low) == 0:
            continue
        h_d = high.iloc[0]['adj_d']
        l_d = low.iloc[0]['adj_d']
        if abs(h_d) > abs(l_d) + 0.2:
            tag = 'WARMUP_SUPPORTED'
        elif abs(l_d) > abs(h_d) + 0.2:
            tag = 'CONTRARY (low-eng stronger)'
        else:
            tag = 'NEITHER (similar)'
        print(f'  {m:<32} high d={h_d:+.2f}  low d={l_d:+.2f}  {tag}')

    # Save
    csv_out = OUTPUT_DIR / 'm61_warmup_effect.csv'
    results_df.to_csv(csv_out, index=False)
    print(f'\nSaved CSV: {csv_out}')

    pdf_path = OUTPUT_DIR / 'm61_warmup_effect.pdf'
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor(BG)
        plot_forest_by_engagement(ax, results_df)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        fig = make_summary_table_page(results_df, median_pages)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    print(f'Saved PDF: {pdf_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()

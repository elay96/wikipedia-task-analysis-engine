#!/usr/bin/env python3
"""
M58: Per-question consistency check
===================================
Question: do the M57 surviving findings replicate within each of the two domains
(art_history vs psychology) separately? If yes -> robust. If only in one
domain -> the effect may depend on question content.

Method: rebuild per-question feature matrix from M56, stratify by domain,
re-run unadjusted t-test and covariate-adjusted regression for the surviving
M57 measures.

Inputs:  data/cleaned/Game.csv
Outputs: output/m58_per_question_consistency.{csv,pdf}
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
from m56_eda_writing_sequential import (
    build_question_df, MIN_PAGE_VISITS, IDLE_THRESHOLD_PCT,
)
from m57_covariate_analysis import ols_fit, cohens_d, COVARIATE

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

# Light palette
BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'

# Surviving M57 measures + a few KILLED ones for comparison
TARGET_MEASURES = [
    'seq_typing_entropy',         # main winner
    'seq_typing_max_run',         # supporting
    'first_writing_time_s',       # supporting
    'seq_typing_mean_run_explore',
    'seq_topic_mean_run_exploit',
    'pasted_chars',               # control: was killed in M57
]

DOMAIN_COLORS = {
    'art_history': '#7B1FA2',
    'psychology':  '#00897B',
    'pooled':      '#1976D2',
}


def analyze_one_subset(qdf_subset, measure):
    """Run unadjusted + adjusted analysis on a single-domain question dataframe."""
    sub = qdf_subset[['condition', measure, COVARIATE]].dropna()
    d_vals = sub.loc[sub['condition'] == 'diffuse', measure].to_numpy()
    c_vals = sub.loc[sub['condition'] == 'clumpy', measure].to_numpy()

    if len(d_vals) < 3 or len(c_vals) < 3:
        return None

    # Unadjusted
    t_raw, p_raw = sp_stats.ttest_ind(d_vals, c_vals, equal_var=False)
    d_raw = cohens_d(d_vals, c_vals)

    # Adjusted (ANCOVA: y ~ intercept + cond + covariate)
    n = len(sub)
    X = np.column_stack([
        np.ones(n),
        (sub['condition'].values == 'diffuse').astype(float),
        sub[COVARIATE].values.astype(float),
    ])
    y = sub[measure].values.astype(float)
    fit = ols_fit(X, y)
    adj_diff = fit['beta'][1]
    adj_t = fit['t'][1]
    adj_p = fit['p'][1]
    sigma = np.sqrt(fit['sigma2'])
    adj_d = adj_diff / sigma if sigma > 0 else 0.0

    return {
        'measure': measure,
        'n_d': len(d_vals), 'n_c': len(c_vals),
        'raw_d': d_raw, 'raw_p': p_raw,
        'adj_d': adj_d, 'adj_p': adj_p,
        'd_mean': d_vals.mean(), 'c_mean': c_vals.mean(),
    }


def build_results_table(qdf_clean):
    rows = []
    for domain in ['art_history', 'psychology', 'pooled']:
        if domain == 'pooled':
            sub_qdf = qdf_clean.copy()
            # For pooled: average per participant first
            agg_cols = TARGET_MEASURES + [COVARIATE]
            sub_qdf = (sub_qdf.groupby(['participant_id', 'condition'], as_index=False)
                              [agg_cols].mean())
        else:
            sub_qdf = qdf_clean[qdf_clean['domain'] == domain].copy()
        for m in TARGET_MEASURES:
            r = analyze_one_subset(sub_qdf, m)
            if r is None:
                continue
            r['domain'] = domain
            rows.append(r)
    return pd.DataFrame(rows)


def plot_forest(ax, results_df, value_col='adj_d', title=''):
    """Forest plot: one row per measure, three points per measure (one per subset)."""
    ax.set_facecolor(BG)
    measures = TARGET_MEASURES
    n_meas = len(measures)
    domains_order = ['pooled', 'art_history', 'psychology']
    domain_offset = {'pooled': 0, 'art_history': -0.22, 'psychology': +0.22}

    for i, m in enumerate(measures):
        for d in domains_order:
            r = results_df[(results_df['measure'] == m) & (results_df['domain'] == d)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            val = r[value_col]
            p = r['adj_p'] if value_col == 'adj_d' else r['raw_p']
            y = i + domain_offset[d]
            color = DOMAIN_COLORS[d]
            edge = '#000' if d == 'pooled' else color
            size = 130 if d == 'pooled' else 75
            marker = 'D' if d == 'pooled' else 'o'
            ax.scatter(val, y, s=size, color=color, edgecolors=edge, linewidth=1.4,
                       marker=marker, zorder=4, alpha=0.95)
            # Annotate p-value
            sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            label = f'{val:+.2f}{sig}'
            ax.annotate(label, (val, y), xytext=(7, 0), textcoords='offset points',
                        fontsize=8, color=color, va='center', fontweight='bold')

    ax.axvline(0, color=BORDER, linewidth=1)
    for thresh in [-0.5, -0.2, 0.2, 0.5]:
        ax.axvline(thresh, color=GRID, linewidth=0.5, linestyle=':')

    ax.set_yticks(range(n_meas))
    ax.set_yticklabels(measures, fontsize=9)
    ax.set_xlabel(f'Cohen\'s d ({value_col}, Diffuse - Clumpy)',
                  color=LABEL, fontweight='bold')
    ax.set_title(title, color=TEXT, fontweight='bold', fontsize=13, pad=10)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.4, axis='x', zorder=0)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.0)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=DOMAIN_COLORS['pooled'],
               markeredgecolor='#000', markersize=10, label='Pooled (M57 result)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DOMAIN_COLORS['art_history'],
               markersize=8, label='art_history only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DOMAIN_COLORS['psychology'],
               markersize=8, label='psychology only'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='lower right', framealpha=0.9)


def make_summary_table_page(results_df):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M58: Per-question consistency table',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')

    lines = []
    lines.append(f'{"Measure":<32} {"Domain":<13} {"N(D/C)":>8} '
                 f'{"Raw d":>7} {"Raw p":>7} {"Adj d":>7} {"Adj p":>7}')
    lines.append('-' * 96)
    for m in TARGET_MEASURES:
        for d in ['pooled', 'art_history', 'psychology']:
            r = results_df[(results_df['measure'] == m) & (results_df['domain'] == d)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            lines.append(
                f'{m if d == "pooled" else "":<32} '
                f'{d:<13} {r["n_d"]:>3}/{r["n_c"]:<4} '
                f'{r["raw_d"]:>+7.2f} {r["raw_p"]:>7.3f} '
                f'{r["adj_d"]:>+7.2f} {r["adj_p"]:>7.3f}'
            )
        lines.append('')
    lines.append('Adjusted = controls for final_answer_length')
    lines.append('Pooled = average per participant (M57 method)')
    lines.append('Domain rows = single-question analysis (1 obs per participant)')

    text = '\n'.join(lines)
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=8.5,
            family='monospace', va='top', color=TEXT, linespacing=1.4)
    return fig


def make_consistency_judgment(results_df):
    """For each measure, judge whether art_history and psychology agree."""
    judgments = []
    for m in TARGET_MEASURES:
        rows = {d: results_df[(results_df['measure'] == m) & (results_df['domain'] == d)]
                for d in ['pooled', 'art_history', 'psychology']}
        if any(len(r) == 0 for r in rows.values()):
            judgments.append({'measure': m, 'verdict': 'INSUFFICIENT_DATA'})
            continue
        ah_d = rows['art_history'].iloc[0]['adj_d']
        ps_d = rows['psychology'].iloc[0]['adj_d']
        pooled_d = rows['pooled'].iloc[0]['adj_d']

        same_sign = (ah_d * ps_d) > 0
        both_meaningful = abs(ah_d) >= 0.2 and abs(ps_d) >= 0.2
        either_significant = (rows['art_history'].iloc[0]['adj_p'] < .05 or
                              rows['psychology'].iloc[0]['adj_p'] < .05)

        if same_sign and both_meaningful and either_significant:
            verdict = 'CONSISTENT'
        elif same_sign and both_meaningful:
            verdict = 'WEAK_CONSISTENT'  # Same direction, both nonzero, neither sig in subset
        elif same_sign:
            verdict = 'ONE_WEAK'  # Same direction but one is small
        else:
            verdict = 'INCONSISTENT'  # Opposite signs

        judgments.append({
            'measure': m, 'pooled_d': pooled_d,
            'ah_d': ah_d, 'ps_d': ps_d, 'verdict': verdict,
        })
    return pd.DataFrame(judgments)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M58] Per-question consistency check')
    print('=' * 60)

    print('\n--- Loading and building question-level features ---')
    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    qdf = build_question_df(trials)
    print(f'  Total questions: {len(qdf)}')

    # Apply exclusions: pages and idle (M56 style)
    excluded_mask = qdf['excluded_pages'] | qdf['excluded_idle']
    qdf_clean = qdf[~excluded_mask].copy()
    # Drop M52 outlier participants
    M52_OUTLIER_PIDS = {26, 79, 134}
    qdf_clean = qdf_clean[~qdf_clean['participant_id'].isin(M52_OUTLIER_PIDS)].copy()
    print(f'  After exclusions: {len(qdf_clean)} questions')
    for d in ['art_history', 'psychology']:
        sub = qdf_clean[qdf_clean['domain'] == d]
        n_d = (sub['condition'] == 'diffuse').sum()
        n_c = (sub['condition'] == 'clumpy').sum()
        print(f'    {d}: N(diff)={n_d}, N(clumpy)={n_c}')

    print('\n--- Running stratified analysis ---')
    results_df = build_results_table(qdf_clean)

    # Print summary
    print(f'\n{"Measure":<32} {"Domain":<13} {"adj d":>7} {"adj p":>7}')
    print('-' * 64)
    for m in TARGET_MEASURES:
        for d in ['pooled', 'art_history', 'psychology']:
            r = results_df[(results_df['measure'] == m) & (results_df['domain'] == d)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            print(f'{m if d == "pooled" else "":<32} {d:<13} '
                  f'{r["adj_d"]:>+7.2f} {r["adj_p"]:>7.3f}')
        print()

    # Consistency judgment
    judgments = make_consistency_judgment(results_df)
    print('CONSISTENCY VERDICTS:')
    for _, j in judgments.iterrows():
        if j['verdict'] in ('INSUFFICIENT_DATA',):
            print(f'  {j["measure"]:<32} {j["verdict"]}')
        else:
            print(f'  {j["measure"]:<32} {j["verdict"]:<18}'
                  f'(pooled {j["pooled_d"]:+.2f}, '
                  f'AH {j["ah_d"]:+.2f}, PS {j["ps_d"]:+.2f})')

    # Save CSV
    csv_out = OUTPUT_DIR / 'm58_per_question_consistency.csv'
    results_df.to_csv(csv_out, index=False)
    print(f'\nSaved CSV: {csv_out}')

    # Build PDF
    pdf_path = OUTPUT_DIR / 'm58_per_question_consistency.pdf'
    with PdfPages(pdf_path) as pdf:
        # Page 1: forest plot of adjusted effects
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor(BG)
        plot_forest(ax, results_df, value_col='adj_d',
                    title='Adjusted Cohen\'s d by domain (controlling for answer length)')
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # Page 2: forest plot of raw effects
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor(BG)
        plot_forest(ax, results_df, value_col='raw_d',
                    title='Raw Cohen\'s d by domain (no covariate)')
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # Page 3: summary table
        fig = make_summary_table_page(results_df)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    print(f'Saved PDF: {pdf_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()

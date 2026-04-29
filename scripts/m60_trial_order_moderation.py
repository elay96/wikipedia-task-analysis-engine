#!/usr/bin/env python3
"""
M60: Does trial order moderate the manipulation effect?
=======================================================
Hypothesis: the SFG manipulation produces a transient state effect that may
decay between question 1 and question 2. If true, the diffuse-clumpy
difference should be larger on the FIRST question and smaller (or zero) on
the SECOND.

Design check (verified): TrialIndex 1 = first real question, TrialIndex 2 =
second. Domain order is randomized and balanced (clumpy 31/32 AH/PS first;
diffuse 31/31).

Method: stratify the M57 covariate analysis by trial position. For each
surviving M57 measure, compare:
  - First-question only (n ~ 100 questions, 1 per participant)
  - Second-question only (n ~ 100 questions)
  - Pooled (M57 baseline)

Outputs: output/m60_trial_order_moderation.{csv,pdf}
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

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

# Light palette
BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'

POSITION_COLORS = {
    1: '#1976D2',  # First question - cool blue
    2: '#E65100',  # Second question - warm orange
    'pooled': '#4A4A4A',
}

TARGET_MEASURES = [
    'seq_typing_entropy',
    'seq_typing_max_run',
    'first_writing_time_s',
    'seq_typing_mean_run_explore',
    'seq_topic_mean_run_exploit',
    'pasted_chars',
    'final_answer_length',
]


def attach_trial_position(qdf, trials):
    """Add 'trial_position' column (1 or 2) by looking up TrialIndex from raw trials."""
    pos_map = {}
    for t in trials:
        if t['domain'] == 'practice':
            continue
        pos_map[(t['pid'], t['domain'])] = t['trial']  # TrialIndex from helpers
    qdf = qdf.copy()
    qdf['trial_position'] = qdf.apply(
        lambda r: pos_map.get((r['participant_id'], r['domain']), np.nan), axis=1)
    return qdf


def analyze_subset(sub_qdf, measure):
    """Same as M57.analyze_measure but on an arbitrary question-level subset."""
    cols = list(dict.fromkeys(['condition', measure, COVARIATE]))
    sub = sub_qdf[cols].dropna()
    d_vals = sub.loc[sub['condition'] == 'diffuse', measure].to_numpy()
    c_vals = sub.loc[sub['condition'] == 'clumpy', measure].to_numpy()
    if len(d_vals) < 3 or len(c_vals) < 3:
        return None
    t_raw, p_raw = sp_stats.ttest_ind(d_vals, c_vals, equal_var=False)
    d_raw = cohens_d(d_vals, c_vals)

    # Adjusted analysis: skip when measure == COVARIATE (would be self-regression)
    if measure == COVARIATE:
        adj_d, adj_p = d_raw, p_raw  # fall back to unadjusted
    else:
        n = len(sub)
        X = np.column_stack([
            np.ones(n),
            (sub['condition'].values == 'diffuse').astype(float),
            sub[COVARIATE].values.astype(float),
        ])
        y = sub[measure].values.astype(float)
        try:
            fit = ols_fit(X, y)
            adj_diff = fit['beta'][1]
            adj_p = fit['p'][1]
            sigma = np.sqrt(fit['sigma2'])
            adj_d = adj_diff / sigma if sigma > 0 else 0.0
        except np.linalg.LinAlgError:
            adj_d, adj_p = np.nan, np.nan
    return {
        'measure': measure, 'n_d': len(d_vals), 'n_c': len(c_vals),
        'raw_d': d_raw, 'raw_p': p_raw, 'adj_d': adj_d, 'adj_p': adj_p,
        'd_mean': d_vals.mean(), 'c_mean': c_vals.mean(),
    }


def build_results(qdf_clean):
    rows = []
    sub1 = qdf_clean[qdf_clean['trial_position'] == 1]
    sub2 = qdf_clean[qdf_clean['trial_position'] == 2]
    # Dedupe column list (COVARIATE may already be in TARGET_MEASURES)
    agg_cols = list(dict.fromkeys(TARGET_MEASURES + [COVARIATE]))
    pooled_avg = (qdf_clean.groupby(['participant_id', 'condition'], as_index=False)
                          [agg_cols].mean())
    for m in TARGET_MEASURES:
        for label, dat in [('first', sub1), ('second', sub2), ('pooled', pooled_avg)]:
            r = analyze_subset(dat, m)
            if r is None:
                continue
            r['stratum'] = label
            rows.append(r)
    return pd.DataFrame(rows)


def plot_forest_by_position(ax, results_df):
    """Forest plot: rows = measures, points per stratum (first / second / pooled)."""
    ax.set_facecolor(BG)
    measures = TARGET_MEASURES
    n = len(measures)
    offset = {'first': -0.22, 'second': +0.22, 'pooled': 0.0}
    color = {'first': POSITION_COLORS[1], 'second': POSITION_COLORS[2],
             'pooled': POSITION_COLORS['pooled']}
    marker = {'first': 'o', 'second': 'o', 'pooled': 'D'}
    size = {'first': 80, 'second': 80, 'pooled': 130}

    for i, m in enumerate(measures):
        for label in ['pooled', 'first', 'second']:
            r = results_df[(results_df['measure'] == m) & (results_df['stratum'] == label)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            val = r['adj_d']
            p = r['adj_p']
            y = i + offset[label]
            ax.scatter(val, y, s=size[label], color=color[label],
                       edgecolors='#000' if label == 'pooled' else color[label],
                       linewidth=1.4, marker=marker[label], zorder=4, alpha=0.95)
            sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            ax.annotate(f'{val:+.2f}{sig}', (val, y), xytext=(7, 0),
                        textcoords='offset points', fontsize=8,
                        color=color[label], va='center', fontweight='bold')

    ax.axvline(0, color=BORDER, linewidth=1)
    for thresh in [-0.5, -0.2, 0.2, 0.5]:
        ax.axvline(thresh, color=GRID, linewidth=0.5, linestyle=':')

    ax.set_yticks(range(n))
    ax.set_yticklabels(measures, fontsize=9)
    ax.set_xlabel('Adjusted Cohen\'s d (Diffuse - Clumpy)',
                  color=LABEL, fontweight='bold')
    ax.set_title('Effect by trial position',
                 color=TEXT, fontweight='bold', fontsize=13, pad=10)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.4, axis='x', zorder=0)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.2)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=POSITION_COLORS['pooled'],
               markeredgecolor='#000', markersize=10, label='Pooled (M57 baseline)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=POSITION_COLORS[1],
               markersize=8, label='First question only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=POSITION_COLORS[2],
               markersize=8, label='Second question only'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='lower right', framealpha=0.9)


def make_summary_table_page(results_df):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M60: Trial Order Moderation Table',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')

    lines = []
    lines.append(f'{"Measure":<32} {"Stratum":<8} {"N(D/C)":>8} '
                 f'{"Raw d":>7} {"Raw p":>7} {"Adj d":>7} {"Adj p":>7} {"Decay?"}')
    lines.append('-' * 92)
    for m in TARGET_MEASURES:
        first_d = None
        second_d = None
        for label in ['pooled', 'first', 'second']:
            r = results_df[(results_df['measure'] == m) & (results_df['stratum'] == label)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            if label == 'first':
                first_d = r['adj_d']
            elif label == 'second':
                second_d = r['adj_d']
            decay = ''
            if label == 'second' and first_d is not None and second_d is not None:
                if abs(second_d) < 0.3 * abs(first_d):
                    decay = '<<< DECAYED'
                elif abs(second_d) < 0.6 * abs(first_d):
                    decay = '<<  ATTENUATED'
                elif abs(second_d) > 1.5 * abs(first_d):
                    decay = '>>  STRONGER'
                else:
                    decay = '~~  STABLE'
            lines.append(
                f'{m if label == "pooled" else "":<32} '
                f'{label:<8} {r["n_d"]:>3}/{r["n_c"]:<4} '
                f'{r["raw_d"]:>+7.2f} {r["raw_p"]:>7.3f} '
                f'{r["adj_d"]:>+7.2f} {r["adj_p"]:>7.3f} {decay}'
            )
        lines.append('')

    lines.append('Decay verdict: comparison of |adj_d|(second) vs |adj_d|(first).')
    lines.append('  <<< DECAYED       second < 30% of first  -> manipulation faded')
    lines.append('  <<  ATTENUATED    second < 60% of first  -> partial decay')
    lines.append('  ~~  STABLE        within +/- 50% of first')
    lines.append('  >>  STRONGER      second > 150% of first -> unexpected reversal')

    text = '\n'.join(lines)
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=8.5,
            family='monospace', va='top', color=TEXT, linespacing=1.4)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M60] Trial order moderation analysis')
    print('=' * 60)

    print('\n--- Loading and building question-level features ---')
    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    qdf = build_question_df(trials)
    qdf = attach_trial_position(qdf, trials)

    # Apply M56 exclusions
    excluded_mask = qdf['excluded_pages'] | qdf['excluded_idle']
    qdf_clean = qdf[~excluded_mask].copy()
    M52_OUTLIERS = {26, 79, 134}
    qdf_clean = qdf_clean[~qdf_clean['participant_id'].isin(M52_OUTLIERS)].copy()
    print(f'  After exclusions: {len(qdf_clean)} questions')
    for pos in [1, 2]:
        sub = qdf_clean[qdf_clean['trial_position'] == pos]
        n_d = (sub['condition'] == 'diffuse').sum()
        n_c = (sub['condition'] == 'clumpy').sum()
        print(f'    Position {pos}: N(diffuse)={n_d}, N(clumpy)={n_c}')

    # Order x domain check
    print('\n--- Order-domain crosstab (should be near-balanced) ---')
    print(pd.crosstab(qdf_clean['trial_position'], qdf_clean['domain']))

    print('\n--- Running stratified analysis ---')
    results_df = build_results(qdf_clean)

    # Print summary
    print(f'\n{"Measure":<32} {"Stratum":<8} {"adj d":>7} {"adj p":>7}')
    print('-' * 62)
    for m in TARGET_MEASURES:
        for label in ['pooled', 'first', 'second']:
            r = results_df[(results_df['measure'] == m) & (results_df['stratum'] == label)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            print(f'{m if label == "pooled" else "":<32} {label:<8} '
                  f'{r["adj_d"]:>+7.2f} {r["adj_p"]:>7.3f}')
        print()

    # Save CSV
    csv_out = OUTPUT_DIR / 'm60_trial_order_moderation.csv'
    results_df.to_csv(csv_out, index=False)
    print(f'Saved CSV: {csv_out}')

    # PDF
    pdf_path = OUTPUT_DIR / 'm60_trial_order_moderation.pdf'
    with PdfPages(pdf_path) as pdf:
        # Page 1: forest plot
        fig, ax = plt.subplots(figsize=(11, 6.5))
        fig.patch.set_facecolor(BG)
        plot_forest_by_position(ax, results_df)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # Page 2: summary table
        fig = make_summary_table_page(results_df)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    print(f'Saved PDF: {pdf_path}')

    # Verdict summary
    print('\n' + '=' * 60)
    print('VERDICT')
    print('=' * 60)
    for m in TARGET_MEASURES:
        first = results_df[(results_df['measure'] == m) & (results_df['stratum'] == 'first')]
        second = results_df[(results_df['measure'] == m) & (results_df['stratum'] == 'second')]
        if len(first) == 0 or len(second) == 0:
            continue
        f_d = first.iloc[0]['adj_d']
        s_d = second.iloc[0]['adj_d']
        ratio = abs(s_d) / abs(f_d) if abs(f_d) > 0 else np.nan
        if not np.isnan(ratio):
            tag = ('DECAYED' if ratio < 0.3 else
                   'ATTENUATED' if ratio < 0.6 else
                   'STABLE' if ratio < 1.5 else 'STRONGER')
            print(f'  {m:<32} first d={f_d:+.2f}  second d={s_d:+.2f}  '
                  f'(2nd/1st = {ratio:.2f})  {tag}')

    print('\nDone.')


if __name__ == '__main__':
    main()

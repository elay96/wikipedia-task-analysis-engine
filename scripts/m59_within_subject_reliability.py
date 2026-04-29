#!/usr/bin/env python3
"""
M59: Within-subject reliability of pattern measures
====================================================
Question: are the M57 surviving measures STABLE TRAITS (a participant who scores
high on art_history also scores high on psychology) or NOISY STATES (the
participant's score barely correlates between their two questions)?

This distinguishes two interpretations of M58's domain asymmetry:
  - Hypothesis A: psychology is just noisier - if so, low test-retest reliability
    in BOTH conditions, but the manipulation still pulls the means.
  - Hypothesis B: the effect lives in art_history-like questions specifically -
    if so, reliability could be similar but the effect emerges only in one domain.

Analyses:
  1. Test-retest correlation (Pearson r) per measure: art_history value vs
     psychology value for the same participant.
  2. Spearman-Brown corrected reliability for the average of two questions.
  3. ICC(2,1) estimate via variance decomposition.
  4. Visual: scatter of AH vs PS by condition.

Outputs: output/m59_within_subject_reliability.{csv,pdf}
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

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

# Light palette
BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'
COND_COLORS = {'diffuse': '#1976D2', 'clumpy': '#C62828'}

TARGET_MEASURES = [
    'seq_typing_entropy',
    'seq_typing_max_run',
    'first_writing_time_s',
    'seq_typing_mean_run_explore',
    'seq_topic_mean_run_exploit',
    'pasted_chars',
    'final_answer_length',
    'count_topic_proxy',  # placeholder removed in code
]
# Drop placeholder
TARGET_MEASURES = [m for m in TARGET_MEASURES if m != 'count_topic_proxy']


def pivot_wide(qdf, measure):
    """Wide table: one row per participant, columns 'art_history' and 'psychology'."""
    wide = qdf.pivot_table(
        index=['participant_id', 'condition'],
        columns='domain',
        values=measure,
        aggfunc='first',
    ).reset_index()
    return wide


def compute_reliability(wide):
    """Pearson r + Spearman-Brown + ICC(2,1) on the paired-question values."""
    paired = wide.dropna(subset=['art_history', 'psychology']).copy()
    n = len(paired)
    if n < 10:
        return None

    ah = paired['art_history'].to_numpy()
    ps = paired['psychology'].to_numpy()

    # Pearson r
    if ah.std() == 0 or ps.std() == 0:
        r, p_r = (np.nan, np.nan)
    else:
        r, p_r = sp_stats.pearsonr(ah, ps)

    # Spearman-Brown for the average of 2 questions
    sb = (2 * r) / (1 + r) if r > -1 and not np.isnan(r) else np.nan

    # ICC(2,1) via two-way random effects ANOVA decomposition
    # Treating the 2 "questions" as raters of the same subject.
    # ICC(2,1) = (MS_subjects - MS_error) /
    #           (MS_subjects + (k-1)*MS_error + k*(MS_judges - MS_error)/n)
    Y = np.column_stack([ah, ps])
    grand_mean = Y.mean()
    subj_means = Y.mean(axis=1)
    judge_means = Y.mean(axis=0)
    k = 2
    ss_subj = k * np.sum((subj_means - grand_mean) ** 2)
    ss_judge = n * np.sum((judge_means - grand_mean) ** 2)
    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_error = ss_total - ss_subj - ss_judge
    df_subj = n - 1
    df_judge = k - 1
    df_error = (n - 1) * (k - 1)
    ms_subj = ss_subj / df_subj if df_subj > 0 else np.nan
    ms_judge = ss_judge / df_judge if df_judge > 0 else np.nan
    ms_error = ss_error / df_error if df_error > 0 else np.nan
    if ms_subj is not None and not np.isnan(ms_error) and ms_error > 0:
        denom = (ms_subj + (k - 1) * ms_error
                 + k * (ms_judge - ms_error) / n)
        icc21 = (ms_subj - ms_error) / denom if denom > 0 else np.nan
    else:
        icc21 = np.nan

    return {
        'n_paired': n,
        'pearson_r': r, 'pearson_p': p_r,
        'spearman_brown': sb,
        'icc21': icc21,
        'ms_subjects': ms_subj, 'ms_error': ms_error,
        'between_var_pct': 100 * ms_subj / (ms_subj + ms_error)
            if (ms_subj + ms_error) > 0 else np.nan,
    }


def reliability_by_condition(wide):
    """Test-retest correlation separately within each condition."""
    out = {}
    for cond in ['diffuse', 'clumpy']:
        sub = wide[wide['condition'] == cond].dropna(subset=['art_history', 'psychology'])
        if len(sub) < 10:
            out[cond] = None
            continue
        ah = sub['art_history'].to_numpy()
        ps = sub['psychology'].to_numpy()
        if ah.std() == 0 or ps.std() == 0:
            out[cond] = {'n': len(sub), 'r': np.nan, 'p': np.nan}
            continue
        r, p = sp_stats.pearsonr(ah, ps)
        out[cond] = {'n': len(sub), 'r': r, 'p': p}
    return out


def interpret_reliability(r):
    if np.isnan(r):
        return 'INSUFFICIENT'
    if r >= 0.75:
        return 'EXCELLENT'
    if r >= 0.50:
        return 'GOOD'
    if r >= 0.30:
        return 'MODERATE'
    if r >= 0.10:
        return 'POOR'
    return 'NEGLIGIBLE'


def plot_test_retest(ax, wide, measure, rel):
    """Scatter of art_history vs psychology values, colored by condition."""
    ax.set_facecolor(BG)
    paired = wide.dropna(subset=['art_history', 'psychology'])
    for cond in ['diffuse', 'clumpy']:
        s = paired[paired['condition'] == cond]
        ax.scatter(s['art_history'], s['psychology'],
                   color=COND_COLORS[cond], s=30, alpha=0.7,
                   edgecolors='#333', linewidth=0.3,
                   label=f'{cond} (n={len(s)})', zorder=3)

    # Diagonal
    lo = min(paired['art_history'].min(), paired['psychology'].min())
    hi = max(paired['art_history'].max(), paired['psychology'].max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=BORDER, linewidth=1, linestyle=':', zorder=1, label='y = x')

    # Regression line
    x = paired['art_history'].to_numpy()
    y = paired['psychology'].to_numpy()
    if x.std() > 0 and y.std() > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(lo - pad, hi + pad, 50)
        ax.plot(xs, slope * xs + intercept, color='#FB8C00',
                linewidth=2, zorder=2, label='OLS fit')

    title = (f'{measure}\n'
             f'r = {rel["pearson_r"]:.2f} (p={rel["pearson_p"]:.3f}, n={rel["n_paired"]})  '
             f'ICC(2,1) = {rel["icc21"]:.2f}  '
             f'[{interpret_reliability(rel["pearson_r"])}]')
    ax.set_title(title, color=TEXT, fontweight='bold', fontsize=10)
    ax.set_xlabel('art_history value', color=LABEL, fontweight='bold', fontsize=9)
    ax.set_ylabel('psychology value', color=LABEL, fontweight='bold', fontsize=9)
    ax.legend(fontsize=7, framealpha=0.85, loc='best')
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.4, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def make_summary_table_page(rel_df, by_cond):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M59: Within-Subject Reliability of Pattern Measures',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')

    lines = []
    lines.append('OVERALL RELIABILITY (both conditions pooled)')
    lines.append(f'{"Measure":<32} {"n":>4} {"Pearson r":>10} {"p":>7} '
                 f'{"S-B":>6} {"ICC(2,1)":>9} {"Verdict":>14}')
    lines.append('-' * 92)
    for _, r in rel_df.iterrows():
        verdict = interpret_reliability(r['pearson_r'])
        lines.append(
            f'{r["measure"]:<32} {r["n_paired"]:>4} '
            f'{r["pearson_r"]:>+10.3f} {r["pearson_p"]:>7.3f} '
            f'{r["spearman_brown"]:>+6.2f} {r["icc21"]:>+9.2f} {verdict:>14}'
        )

    lines.append('')
    lines.append('RELIABILITY BY CONDITION (does it differ between groups?)')
    lines.append(f'{"Measure":<32} {"Diffuse r":>10} {"n":>4}  {"Clumpy r":>10} {"n":>4}')
    lines.append('-' * 72)
    for measure, conds in by_cond.items():
        d_str = (f'{conds["diffuse"]["r"]:>+10.3f} {conds["diffuse"]["n"]:>4}'
                 if conds.get('diffuse') else '       n/a    n/a')
        c_str = (f'{conds["clumpy"]["r"]:>+10.3f} {conds["clumpy"]["n"]:>4}'
                 if conds.get('clumpy') else '       n/a    n/a')
        lines.append(f'{measure:<32} {d_str}  {c_str}')

    lines.append('')
    lines.append('Interpretation guide:')
    lines.append('  EXCELLENT  r >= .75 - measure captures a stable trait')
    lines.append('  GOOD       r >= .50 - measure has substantial trait variance')
    lines.append('  MODERATE   r >= .30 - measure mixes trait and noise')
    lines.append('  POOR       r >= .10 - mostly noise, but some signal')
    lines.append('  NEGLIGIBLE r < .10  - measure is essentially state/noise')
    lines.append('')
    lines.append('Spearman-Brown = reliability of the AVERAGED 2-question score.')
    lines.append('ICC(2,1) = treats questions as random raters of the same subject.')

    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes, fontsize=8.5,
            family='monospace', va='top', color=TEXT, linespacing=1.4)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M59] Within-subject reliability check')
    print('=' * 60)

    print('\n--- Loading and building question-level features ---')
    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    qdf = build_question_df(trials)

    # Apply M56 exclusions
    excluded_mask = qdf['excluded_pages'] | qdf['excluded_idle']
    qdf_clean = qdf[~excluded_mask].copy()
    M52_OUTLIERS = {26, 79, 134}
    qdf_clean = qdf_clean[~qdf_clean['participant_id'].isin(M52_OUTLIERS)].copy()
    print(f'  After exclusions: {len(qdf_clean)} questions, '
          f'{qdf_clean["participant_id"].nunique()} participants')

    print('\n--- Computing reliability per measure ---')
    rel_rows = []
    by_cond = {}
    wide_per_measure = {}
    for m in TARGET_MEASURES:
        wide = pivot_wide(qdf_clean, m)
        wide_per_measure[m] = wide
        rel = compute_reliability(wide)
        if rel is None:
            print(f'  SKIP {m}: insufficient paired data')
            continue
        rel['measure'] = m
        rel_rows.append(rel)
        by_cond[m] = reliability_by_condition(wide)

        verdict = interpret_reliability(rel['pearson_r'])
        print(f'  {m:<32}  r={rel["pearson_r"]:+.2f} '
              f'(n={rel["n_paired"]})  ICC={rel["icc21"]:+.2f}  [{verdict}]')

    rel_df = pd.DataFrame(rel_rows)

    print('\n--- Reliability by condition ---')
    for m, conds in by_cond.items():
        d_r = conds.get('diffuse', {}).get('r', np.nan) if conds.get('diffuse') else np.nan
        c_r = conds.get('clumpy', {}).get('r', np.nan) if conds.get('clumpy') else np.nan
        diff = d_r - c_r if not (np.isnan(d_r) or np.isnan(c_r)) else np.nan
        print(f'  {m:<32}  diffuse r={d_r:+.2f}  clumpy r={c_r:+.2f}  '
              f'(diff={diff:+.2f})')

    # Save CSV
    csv_out = OUTPUT_DIR / 'm59_within_subject_reliability.csv'
    rel_df.to_csv(csv_out, index=False)
    print(f'\nSaved CSV: {csv_out}')

    # PDF
    pdf_path = OUTPUT_DIR / 'm59_within_subject_reliability.pdf'
    with PdfPages(pdf_path) as pdf:
        # Page 1: summary table
        fig = make_summary_table_page(rel_df, by_cond)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # Pages 2+: 2x2 grid of test-retest scatters
        measures_to_plot = [m for m in TARGET_MEASURES if m in by_cond]
        for i in range(0, len(measures_to_plot), 4):
            chunk = measures_to_plot[i:i + 4]
            fig, axes = plt.subplots(2, 2, figsize=(11, 9))
            fig.patch.set_facecolor(BG)
            axes = axes.ravel()
            for j, m in enumerate(chunk):
                rel = next(r for r in rel_rows if r['measure'] == m)
                plot_test_retest(axes[j], wide_per_measure[m], m, rel)
            for k in range(len(chunk), 4):
                axes[k].axis('off')
            plt.tight_layout()
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)

    print(f'Saved PDF: {pdf_path}')

    # Verdict summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for _, r in rel_df.iterrows():
        v = interpret_reliability(r['pearson_r'])
        print(f'  {r["measure"]:<32} {v}')
    print('\nDone.')


if __name__ == '__main__':
    main()

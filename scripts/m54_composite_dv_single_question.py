#!/usr/bin/env python3
"""
M54: Composite DV restricted to a single question (domain).
===========================================================
Per advisor request: re-run the M52 pipeline using only one of the two
questions, since one was rejected disproportionately. We default to the
question with fewer rejections (art_history). Pass --domain psychology to
flip.

Each participant now contributes at most ONE question, so exclusion at the
participant level is the same as exclusion at the question level: we keep
participants whose chosen question survives the page-count and idle filters,
then apply the 3-SD outlier rule on the resulting averages (which here equal
the single observation).

Outputs (suffixed with the chosen domain):
  - output/m54_<domain>_only.png
  - output/m54_<domain>_only.csv
  - output/m54_<domain>_only.pdf
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from helpers import load_trials, OUTPUT_DIR
from m52_final_composite_dv import (
    build_question_data, run_pca, FEATURE_NAMES,
    create_pdf_report,
    BG_COLOR, TEXT_COLOR, LABEL_COLOR, MUTED_COLOR, EXCLUDED_COLOR,
    KEPT_COLOR, BORDER_COLOR,
    THRESHOLD_S, IDLE_THRESHOLD_PCT, MIN_PAGE_VISITS, OUTLIER_SD,
)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

VALID_DOMAINS = ('art_history', 'psychology')


def filter_to_domain(question_df, domain):
    sub = question_df[question_df['domain'] == domain].copy()
    return sub.reset_index(drop=True)


def apply_exclusions_single(question_df, domain):
    """One row per participant; same rule as M52 but no averaging needed."""
    n_total = len(question_df)
    excluded_mask = question_df['excluded_pages'] | question_df['excluded_idle']
    clean = question_df[~excluded_mask].copy()

    excl_pages = int(question_df['excluded_pages'].sum())
    excl_idle = int(question_df['excluded_idle'].sum())
    excl_both = int((question_df['excluded_pages'] & question_df['excluded_idle']).sum())
    excl_total = int(excluded_mask.sum())

    print(f'  Domain: {domain}')
    print(f'  Questions in scope: {n_total}')
    print(f'  Excluded (< {MIN_PAGE_VISITS} pages): {excl_pages}')
    print(f'  Excluded (idle >= {IDLE_THRESHOLD_PCT:.0f}%): {excl_idle}')
    print(f'  Excluded (both): {excl_both}')
    print(f'  Remaining: {len(clean)} / {n_total}')

    count_cols = ['count_time', 'count_topic', 'count_typing']
    avg_df = clean[['participant_id'] + count_cols].dropna().reset_index(drop=True)

    outlier_mask = pd.Series(False, index=avg_df.index)
    outlier_details = []
    for col in count_cols:
        m, s = avg_df[col].mean(), avg_df[col].std()
        lo, hi = m - OUTLIER_SD * s, m + OUTLIER_SD * s
        col_mask = (avg_df[col] < lo) | (avg_df[col] > hi)
        if col_mask.any():
            for pid in avg_df.loc[col_mask, 'participant_id']:
                v = avg_df.loc[avg_df['participant_id'] == pid, col].values[0]
                outlier_details.append(
                    f'    P{pid}: {col} = {v:.2f} (mean={m:.2f}, sd={s:.2f})'
                )
        outlier_mask = outlier_mask | col_mask
    n_outliers = int(outlier_mask.sum())
    print(f'  Outliers (>{OUTLIER_SD} SD): {n_outliers}')
    for d in outlier_details:
        print(d)

    final_df = avg_df[~outlier_mask].reset_index(drop=True)
    print(f'  Final N: {len(final_df)} participants')

    all_pids = set(question_df['participant_id'].unique())
    surviving_pids = set(clean['participant_id'].unique())
    fully_excluded_pids = all_pids - surviving_pids
    outlier_pids = set(avg_df.loc[outlier_mask, 'participant_id'].values) if n_outliers else set()

    summary = {
        'n_total_questions': n_total,
        'n_participants_before': n_total,
        'excl_pages': excl_pages,
        'excl_idle': excl_idle,
        'excl_both': excl_both,
        'excl_total': excl_total,
        'n_clean_questions': len(clean),
        'n_outliers': n_outliers,
        'outlier_details': outlier_details,
        'n_final': len(final_df),
        'fully_excluded_pids': fully_excluded_pids,
        'outlier_pids': outlier_pids,
        'domain': domain,
    }
    return final_df, summary


def make_summary_panel(summary, domain):
    """Single-page text summary that mirrors the look of M52's title page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis('off')
    ax.text(0.5, 0.92, f'M54: Single-Question Composite DV ({domain})',
            transform=ax.transAxes, fontsize=22, fontweight='bold',
            ha='center', va='top', color=TEXT_COLOR)
    ax.text(0.5, 0.84,
            'PCA on raw switch counts using only one of the two task questions',
            transform=ax.transAxes, fontsize=12, ha='center', va='top',
            color=MUTED_COLOR, style='italic')
    text = (
        f'Domain restricted to:  {domain}\n\n'
        f'Exclusion criteria (mirrors M52):\n'
        f'  1. Question with < {MIN_PAGE_VISITS} page visits\n'
        f'  2. Question with idle >= {IDLE_THRESHOLD_PCT:.0f}%\n'
        f'  3. Participant > {OUTLIER_SD} SD outlier on any DV\n\n'
        f'Initial sample (this domain): {summary["n_total_questions"]}\n'
        f'  excluded by pages: {summary["excl_pages"]}\n'
        f'  excluded by idle:  {summary["excl_idle"]}\n'
        f'  excluded by both:  {summary["excl_both"]}\n'
        f'  3 SD outliers:     {summary["n_outliers"]}\n'
        f'\n'
        f'Final sample: N = {summary["n_final"]} participants'
    )
    ax.text(0.06, 0.66, text, transform=ax.transAxes, fontsize=11,
            family='monospace', va='top', color=TEXT_COLOR, linespacing=1.7)
    return fig


def render_question_df_for_pdf(question_df, domain):
    """Adapt the per-domain question_df so M52's plot_exclusion_summary works."""
    sub = question_df[question_df['domain'] == domain].copy()
    return sub.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=VALID_DOMAINS, default='art_history',
                        help='Which question domain to analyse (default: art_history)')
    args = parser.parse_args()
    domain = args.domain

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[M54] Composite DV restricted to {domain}')
    print('=' * 60)

    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    conditions = {tr['pid']: tr['condition'] for tr in trials}

    print('\n--- Building question-level data ---')
    full_question_df = build_question_data(trials)
    print(f'  Total questions across both domains: {len(full_question_df)}')

    domain_df = filter_to_domain(full_question_df, domain)
    print(f'  Questions in domain "{domain}": {len(domain_df)}')

    print('\n--- Applying exclusion criteria ---')
    avg_df, exclusion_summary = apply_exclusions_single(domain_df, domain)

    if len(avg_df) < 4:
        print('Too few participants for PCA after exclusions. Aborting.')
        return

    count_cols = ['count_time', 'count_topic', 'count_typing']
    print('\n--- Descriptive statistics (final sample) ---')
    for col in count_cols:
        v = avg_df[col]
        print(f'  {col}: mean={v.mean():.2f}, sd={v.std():.2f}, '
              f'min={v.min():.2f}, max={v.max():.2f}')

    corr = avg_df[count_cols].corr()
    print('\n--- Correlation matrix ---')
    print(corr.to_string())

    print('\n--- PCA (raw counts, no standardization) ---')
    pca, scores, pct = run_pca(avg_df)
    for i, v in enumerate(pct):
        print(f'  PC{i+1}: {v:.1f}%')
    print('  Loadings:')
    for i, comp in enumerate(pca.components_):
        parts = ', '.join(f'{FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(3))
        print(f'    PC{i+1}: {parts}')

    # Outputs
    print('\n--- Saving outputs ---')
    csv_path = OUTPUT_DIR / f'm54_{domain}_only.csv'
    pd.DataFrame({
        'participant_id': avg_df['participant_id'].values,
        'condition': [conditions.get(p, '') for p in avg_df['participant_id']],
        'count_time': avg_df['count_time'].values,
        'count_topic': avg_df['count_topic'].values,
        'count_typing': avg_df['count_typing'].values,
        'PC1': scores[:, 0],
        'PC2': scores[:, 1],
        'PC3': scores[:, 2],
    }).to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    # Reuse M52's PDF builder; pass the domain-only question_df so the exclusion
    # bar chart shows what we actually used.
    pdf_path = OUTPUT_DIR / f'm54_{domain}_only.pdf'
    pdf_question_df = render_question_df_for_pdf(full_question_df, domain)
    create_pdf_report(avg_df, pca, scores, pct, conditions, pdf_question_df,
                      exclusion_summary, pdf_path)

    # Light-weight side-by-side comparison vs the full M52 result, if available
    full_csv = OUTPUT_DIR / 'm52_final_composite_dv.csv'
    if full_csv.exists():
        full = pd.read_csv(full_csv)
        merged = full.rename(columns={'PC1': 'PC1_full'})[['participant_id', 'PC1_full']].merge(
            pd.DataFrame({'participant_id': avg_df['participant_id'].values,
                          'PC1_q2': scores[:, 0]}),
            on='participant_id', how='inner',
        )
        if len(merged) >= 4:
            r = np.corrcoef(merged['PC1_full'], merged['PC1_q2'])[0, 1]
            print(f'\nCross-check vs M52 PC1: N overlap = {len(merged)}, '
                  f'Pearson r = {r:.3f}')

    print('\nDone.')


if __name__ == '__main__':
    main()

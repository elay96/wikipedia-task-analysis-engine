#!/usr/bin/env python3
"""
M65: Correlation between mean reading-only streak and W/N switch rate
======================================================================
Question (advisor meeting prep, 2026-04-30): are seq_typing_mean_run_explore
(mean reading-only streak) and the W/N switch rate redundant, or do they
capture different aspects of the same sequence?

Two switch measures:
  - n_switches_typing: raw count of W <-> N transitions in the page sequence
  - switch_rate_typing: n_switches / (n_pages - 1), i.e. normalized by max
    possible switches.

Reports:
  - Pearson + Spearman correlation per-question (all kept questions).
  - Correlation per-participant (averaged across both questions).
  - Effect-size table for each measure (replicates m60 metric for switch rate).

Inputs:  data/cleaned/Game.csv
Outputs: output/m65_streak_vs_switchrate.{csv,pdf}
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats

from helpers import load_trials, OUTPUT_DIR
from m18_typing_binary import page_had_typing_or_paste
from m56_eda_writing_sequential import (
    build_question_df, MIN_PAGE_VISITS, IDLE_THRESHOLD_PCT,
)
from m57_covariate_analysis import cohens_d, COVARIATE, ols_fit

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'
LDA_PATH = DATA_DIR / 'cleaned' / 'topic_model.json'

# Time-based exploit threshold (matches M34)
TIME_THRESHOLD_S = 60.0

BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'
COND_COLORS = {'diffuse': '#1976D2', 'clumpy': '#C62828'}


def load_lda_topic_per_page():
    """Load LDA topic_model.json and return {page_slug: dominant_topic_id}."""
    with open(LDA_PATH, 'r') as f:
        tm = json.load(f)
    return {slug: int(np.argmax(dist))
            for slug, dist in tm['topic_distributions'].items()}


def _switch_count_and_rate(seq):
    """Return (n_switches, switch_rate) for a sequence of labels."""
    n = len(seq)
    if n < 2:
        return np.nan, np.nan
    seq = np.asarray(seq)
    n_switches = int((seq[1:] != seq[:-1]).sum())
    return n_switches, n_switches / (n - 1)


def compute_switch_features(trials):
    """Per-question switch count + normalized switch rate for all 3 sources:
       typing (W/N), topic (LDA dominant topic), time (>60s exploit/explore).
    """
    lda_topic = load_lda_topic_per_page()
    rows = []
    for tr in trials:
        if tr['domain'] == 'practice':
            continue
        pvs = tr['page_visits']
        n = len(pvs)

        # Typing W/N sequence
        typing_lbl = [page_had_typing_or_paste(pv, tr['typing_intervals'], tr['paste_times'])
                      for pv in pvs]
        typing_seq = np.array([1 if v else 0 for v in typing_lbl], dtype=int)
        n_sw_typing, sr_typing = _switch_count_and_rate(typing_seq)

        # Topic (LDA) sequence: dominant topic per page; -1 if missing.
        topic_seq = np.array([lda_topic.get(pv['title'], -1) for pv in pvs], dtype=int)
        n_sw_topic, sr_topic = _switch_count_and_rate(topic_seq)

        # Time sequence: dwell > THRESHOLD = exploit (1), else explore (0).
        time_seq = np.array([1 if pv['duration'] > TIME_THRESHOLD_S else 0 for pv in pvs],
                            dtype=int)
        n_sw_time, sr_time = _switch_count_and_rate(time_seq)

        rows.append({
            'participant_id': tr['pid'],
            'condition': tr['condition'],
            'domain': tr['domain'],
            'n_pages': n,
            'n_switches_typing': n_sw_typing,
            'switch_rate_typing': sr_typing,
            'n_switches_topic': n_sw_topic,
            'switch_rate_topic': sr_topic,
            'n_switches_time': n_sw_time,
            'switch_rate_time': sr_time,
        })
    return pd.DataFrame(rows)


def correlate_two(a, b, label_a, label_b):
    """Return Pearson and Spearman correlation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 10 or a.std() == 0 or b.std() == 0:
        return None
    r_p, p_p = sp_stats.pearsonr(a, b)
    r_s, p_s = sp_stats.spearmanr(a, b)
    return {
        'pair': f'{label_a} vs {label_b}',
        'n': len(a),
        'pearson_r': r_p, 'pearson_p': p_p,
        'spearman_r': r_s, 'spearman_p': p_s,
    }


def run_effect_test(avg_df, measure):
    """Welch t + adj_d (controlling for COVARIATE)."""
    sub = avg_df[['condition', measure, COVARIATE]].dropna()
    d_vals = sub.loc[sub['condition'] == 'diffuse', measure].to_numpy()
    c_vals = sub.loc[sub['condition'] == 'clumpy', measure].to_numpy()
    if len(d_vals) < 3 or len(c_vals) < 3:
        return None
    t_raw, p_raw = sp_stats.ttest_ind(d_vals, c_vals, equal_var=False)
    d_raw = cohens_d(d_vals, c_vals)
    n = len(sub)
    X = np.column_stack([
        np.ones(n),
        (sub['condition'].values == 'diffuse').astype(float),
        sub[COVARIATE].values.astype(float),
    ])
    y = sub[measure].values.astype(float)
    fit = ols_fit(X, y)
    sigma = np.sqrt(fit['sigma2'])
    adj_d = fit['beta'][1] / sigma if sigma > 0 else 0.0
    adj_p = fit['p'][1]
    return {
        'measure': measure, 'n_d': len(d_vals), 'n_c': len(c_vals),
        'd_mean': d_vals.mean(), 'c_mean': c_vals.mean(),
        'raw_d': d_raw, 'raw_p': p_raw, 'adj_d': adj_d, 'adj_p': adj_p,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M65] Streak vs switch-rate correlation')
    print('=' * 60)

    print('\n--- Loading data ---')
    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')

    # Build switch feature df + standard m56 question features and merge
    sw_df = compute_switch_features(trials)
    qdf = build_question_df(trials)

    merged = qdf.merge(
        sw_df[['participant_id', 'condition', 'domain',
               'n_switches_typing', 'switch_rate_typing',
               'n_switches_topic',  'switch_rate_topic',
               'n_switches_time',   'switch_rate_time']],
        on=['participant_id', 'condition', 'domain'], how='left',
    )

    # Apply M56 exclusions
    excluded = merged['excluded_pages'] | merged['excluded_idle']
    qdf_clean = merged[~excluded].copy()
    M52_OUTLIERS = {26, 79, 134}
    qdf_clean = qdf_clean[~qdf_clean['participant_id'].isin(M52_OUTLIERS)].copy()
    print(f'  Clean questions: {len(qdf_clean)}')

    # ---- Per-question correlations -----------------------------------------
    print('\n=== Per-question correlations (N = clean questions) ===')
    pairs = [
        ('seq_typing_mean_run_explore', 'n_switches_typing'),
        ('seq_typing_mean_run_explore', 'switch_rate_typing'),
        ('seq_typing_mean_run_explore', 'switch_rate_topic'),
        ('seq_typing_mean_run_explore', 'switch_rate_time'),
        ('seq_typing_entropy', 'n_switches_typing'),
        ('seq_typing_entropy', 'switch_rate_typing'),
        ('n_switches_typing', 'switch_rate_typing'),
    ]
    cor_rows = []
    for a, b in pairs:
        r = correlate_two(qdf_clean[a], qdf_clean[b], a, b)
        if r is None:
            continue
        cor_rows.append(r)
        print(f'  {a} vs {b}')
        print(f'    Pearson r={r["pearson_r"]:+.3f} (p={r["pearson_p"]:.3g}, n={r["n"]})')
        print(f'    Spearman r={r["spearman_r"]:+.3f} (p={r["spearman_p"]:.3g})')

    cor_df = pd.DataFrame(cor_rows)

    # ---- Per-participant correlations (averaged across the 2 questions) ----
    print('\n=== Per-participant correlations (averaged across questions) ===')
    measure_cols = [c for c in qdf_clean.columns
                    if c not in {'participant_id', 'condition', 'domain', 'n_pages',
                                 'idle_pct', 'excluded_pages', 'excluded_idle'}]
    avg_df = (qdf_clean.groupby(['participant_id', 'condition'], as_index=False)
                       [measure_cols].mean())
    print(f'  N participants: {len(avg_df)}')
    pp_rows = []
    for a, b in pairs:
        r = correlate_two(avg_df[a], avg_df[b], a, b)
        if r is None:
            continue
        pp_rows.append(r)
        print(f'  {a} vs {b}')
        print(f'    Pearson r={r["pearson_r"]:+.3f} (p={r["pearson_p"]:.3g}, n={r["n"]})')
        print(f'    Spearman r={r["spearman_r"]:+.3f} (p={r["spearman_p"]:.3g})')

    pp_df = pd.DataFrame(pp_rows)

    # ---- Effect on switch measures themselves -----------------------------
    print('\n=== Effect (Diffuse - Clumpy) for switch measures ===')
    eff_rows = []
    for m in ['n_switches_typing', 'switch_rate_typing',
              'seq_typing_mean_run_explore', 'seq_typing_entropy']:
        r = run_effect_test(avg_df, m)
        if r is None:
            continue
        eff_rows.append(r)
        print(f'  {m:<32}  raw d={r["raw_d"]:+.2f} (p={r["raw_p"]:.3f})  '
              f'adj d={r["adj_d"]:+.2f} (p={r["adj_p"]:.3f})')
    eff_df = pd.DataFrame(eff_rows)

    # ---- Save CSV ----------------------------------------------------------
    csv_out = OUTPUT_DIR / 'm65_streak_vs_switchrate.csv'
    with open(csv_out, 'w', encoding='utf-8') as f:
        f.write('# Per-question correlations\n')
        cor_df.to_csv(f, index=False)
        f.write('\n# Per-participant correlations\n')
        pp_df.to_csv(f, index=False)
        f.write('\n# Effect tests (per participant, condition)\n')
        eff_df.to_csv(f, index=False)
    print(f'\nSaved CSV: {csv_out}')

    # Per-participant tidy CSV (used by m66 correlation page)
    tidy_cols = [
        'participant_id', 'condition',
        'seq_typing_mean_run_explore',
        'switch_rate_typing', 'switch_rate_topic', 'switch_rate_time',
        'n_switches_typing', 'n_switches_topic', 'n_switches_time',
    ]
    tidy_out = OUTPUT_DIR / 'm65_per_participant_streak_switches.csv'
    avg_df[tidy_cols].to_csv(tidy_out, index=False)
    print(f'Saved CSV: {tidy_out}')

    # ---- PDF ---------------------------------------------------------------
    pdf_path = OUTPUT_DIR / 'm65_streak_vs_switchrate.pdf'
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(make_correlation_table_page(cor_df, pp_df, eff_df), facecolor=BG)
        plt.close()
        pdf.savefig(make_scatter_page(qdf_clean, avg_df), facecolor=BG)
        plt.close()
    print(f'Saved PDF: {pdf_path}')
    print('\nDone.')


# ----------------------------------------------------------------------
# Plot pages
# ----------------------------------------------------------------------

def make_correlation_table_page(cor_df, pp_df, eff_df):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M65 - Streak vs switch-rate correlation and effect',
                 fontsize=13, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')

    lines = []
    lines.append('PER-QUESTION CORRELATIONS (each row = pair across all clean questions)')
    lines.append(f'{"Pair":<60} {"n":>4} {"Pearson r":>10} {"Pear p":>8} '
                 f'{"Spear r":>9}')
    lines.append('-' * 96)
    for _, r in cor_df.iterrows():
        lines.append(
            f'{r["pair"]:<60} {r["n"]:>4} '
            f'{r["pearson_r"]:>+10.3f} {r["pearson_p"]:>8.3g} '
            f'{r["spearman_r"]:>+9.3f}'
        )
    lines.append('')
    lines.append('PER-PARTICIPANT CORRELATIONS (averaged across both questions)')
    lines.append(f'{"Pair":<60} {"n":>4} {"Pearson r":>10} {"Pear p":>8} '
                 f'{"Spear r":>9}')
    lines.append('-' * 96)
    for _, r in pp_df.iterrows():
        lines.append(
            f'{r["pair"]:<60} {r["n"]:>4} '
            f'{r["pearson_r"]:>+10.3f} {r["pearson_p"]:>8.3g} '
            f'{r["spearman_r"]:>+9.3f}'
        )
    lines.append('')
    lines.append('EFFECT (Diffuse - Clumpy), per-participant averaged values')
    lines.append(f'{"Measure":<32} {"d_mean":>9} {"c_mean":>9} '
                 f'{"raw d":>7} {"raw p":>7} {"adj d":>7} {"adj p":>7}')
    lines.append('-' * 90)
    for _, r in eff_df.iterrows():
        lines.append(
            f'{r["measure"]:<32} {r["d_mean"]:>9.3f} {r["c_mean"]:>9.3f} '
            f'{r["raw_d"]:>+7.2f} {r["raw_p"]:>7.3f} '
            f'{r["adj_d"]:>+7.2f} {r["adj_p"]:>7.3f}'
        )
    lines.append('')
    lines.append('Switch rate = n_switches / (n_pages - 1).  Streak = mean run length of "no-typing" pages.')
    lines.append('If correlation is near -1 (high streak <-> low switches), the two measures are redundant.')

    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes, fontsize=8.6,
            family='monospace', va='top', color=TEXT, linespacing=1.4)
    return fig


def make_scatter_page(qdf_clean, avg_df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M65 - Streak vs switch measures (scatter)',
                 fontsize=13, fontweight='bold', color=TEXT, y=0.99)

    pairs_to_plot = [
        (qdf_clean, 'seq_typing_mean_run_explore', 'switch_rate_typing',
         'Per question (clean): streak vs switch rate'),
        (qdf_clean, 'seq_typing_mean_run_explore', 'n_switches_typing',
         'Per question (clean): streak vs n switches'),
        (avg_df, 'seq_typing_mean_run_explore', 'switch_rate_typing',
         'Per participant (avg): streak vs switch rate'),
        (avg_df, 'seq_typing_mean_run_explore', 'n_switches_typing',
         'Per participant (avg): streak vs n switches'),
    ]
    for ax, (df, x, y, title) in zip(axes.ravel(), pairs_to_plot):
        ax.set_facecolor(BG)
        sub = df[['condition', x, y]].dropna()
        for cond in ['diffuse', 'clumpy']:
            s = sub[sub['condition'] == cond]
            ax.scatter(s[x], s[y], color=COND_COLORS[cond], s=24, alpha=0.7,
                       edgecolors='#333', linewidth=0.3,
                       label=f'{cond} (n={len(s)})')
        if len(sub) >= 10 and sub[x].std() > 0 and sub[y].std() > 0:
            r, p = sp_stats.pearsonr(sub[x], sub[y])
        else:
            r, p = (np.nan, np.nan)
        ax.set_title(f'{title}\nPearson r={r:+.2f}  p={p:.3g}',
                     fontsize=10, color=TEXT, fontweight='bold')
        ax.set_xlabel(x, color=LABEL, fontsize=9)
        ax.set_ylabel(y, color=LABEL, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(fontsize=7, loc='best', framealpha=0.85)
        ax.grid(True, color=GRID, linewidth=0.4, zorder=0)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


if __name__ == '__main__':
    main()

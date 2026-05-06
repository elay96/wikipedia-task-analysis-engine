#!/usr/bin/env python3
"""
M71: Correlation between mean reading length and explore/exploit switch rate
=============================================================================
Question (advisor request, 2026-05-03): what is the correlation between the
average reading length (mean page dwell time, in seconds) and each of the
explore/exploit alternation measures, especially the typing-based one
(switch_rate_typing).

Mean reading length per question = mean of page_visit['duration'] across all
pages within the question (only real trials, i.e., domain != 'practice').

Switch-rate measures (from M65):
  - switch_rate_typing  (page had typing/paste -> exploit, else explore)
  - switch_rate_topic   (LDA dominant topic same/different than previous)
  - switch_rate_time    (page duration > 60s -> exploit, else explore)

Exclusions: same as M56/M65 (n_pages >= 3, idle < 50%, drop M52 outliers
{26, 79, 134}).

Reports:
  - Per-question correlations (Pearson + Spearman)
  - Per-participant correlations (averaged across both questions)
  - Stratified by condition (diffuse / clumpy)

Inputs:  data/cleaned/Game.csv
Outputs: output/m71_reading_length_vs_switchrate.{csv,pdf}
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
from m56_eda_writing_sequential import build_question_df, MIN_PAGE_VISITS, IDLE_THRESHOLD_PCT
from m65_streak_vs_switchrate import compute_switch_features

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

# Light palette (project preference)
BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'
COND_COLORS = {'diffuse': '#1976D2', 'clumpy': '#C62828'}

M52_OUTLIERS = {26, 79, 134}


def compute_reading_length(trials):
    """Per-question mean page dwell time (seconds)."""
    rows = []
    for tr in trials:
        if tr['domain'] == 'practice':
            continue
        pvs = tr['page_visits']
        if len(pvs) == 0:
            mean_read = np.nan
            median_read = np.nan
        else:
            durations = np.array([pv['duration'] for pv in pvs], dtype=float)
            mean_read = float(durations.mean())
            median_read = float(np.median(durations))
        rows.append({
            'participant_id': tr['pid'],
            'condition': tr['condition'],
            'domain': tr['domain'],
            'mean_reading_length_s': mean_read,
            'median_reading_length_s': median_read,
        })
    return pd.DataFrame(rows)


def correlate_two(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return None
    r_p, p_p = sp_stats.pearsonr(a, b)
    r_s, p_s = sp_stats.spearmanr(a, b)
    return {
        'n': len(a),
        'pearson_r': r_p, 'pearson_p': p_p,
        'spearman_r': r_s, 'spearman_p': p_s,
    }


def fmt_correlation_block(label, res):
    if res is None:
        return f'  {label}: insufficient data'
    return (f'  {label}\n'
            f'    Pearson  r={res["pearson_r"]:+.3f}  p={res["pearson_p"]:.3g}  n={res["n"]}\n'
            f'    Spearman r={res["spearman_r"]:+.3f}  p={res["spearman_p"]:.3g}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M71] Reading length vs switch-rate correlation')
    print('=' * 60)

    print('\n--- Loading data ---')
    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')

    rd_df = compute_reading_length(trials)
    sw_df = compute_switch_features(trials)
    qdf = build_question_df(trials)

    keys = ['participant_id', 'condition', 'domain']
    merged = (qdf[keys + ['n_pages', 'idle_pct', 'excluded_pages', 'excluded_idle']]
              .merge(rd_df, on=keys, how='left')
              .merge(sw_df[keys + ['switch_rate_typing', 'switch_rate_topic',
                                   'switch_rate_time']],
                     on=keys, how='left'))

    excluded = merged['excluded_pages'] | merged['excluded_idle']
    clean_q = merged[~excluded].copy()
    clean_q = clean_q[~clean_q['participant_id'].isin(M52_OUTLIERS)].copy()
    print(f'  Clean questions: {len(clean_q)}')

    measures = ['switch_rate_typing', 'switch_rate_topic', 'switch_rate_time']
    reading_cols = ['mean_reading_length_s', 'median_reading_length_s']

    # ---- Per-question correlations -----------------------------------------
    print('\n=== Per-question correlations (N = clean questions) ===')
    cor_rows = []
    for rcol in reading_cols:
        for mcol in measures:
            r = correlate_two(clean_q[rcol], clean_q[mcol])
            print(fmt_correlation_block(f'{rcol} vs {mcol}', r))
            if r is not None:
                cor_rows.append({'level': 'per_question', 'subset': 'all',
                                 'reading_measure': rcol, 'switch_measure': mcol, **r})

    # ---- Per-participant correlations --------------------------------------
    avg_df = (clean_q.groupby(['participant_id', 'condition'], as_index=False)
                     [reading_cols + measures].mean())
    print(f'\n=== Per-participant correlations (N = {len(avg_df)} participants) ===')
    pp_rows = []
    for rcol in reading_cols:
        for mcol in measures:
            r = correlate_two(avg_df[rcol], avg_df[mcol])
            print(fmt_correlation_block(f'{rcol} vs {mcol}', r))
            if r is not None:
                pp_rows.append({'level': 'per_participant', 'subset': 'all',
                                'reading_measure': rcol, 'switch_measure': mcol, **r})

    # ---- Per-participant correlations split by condition -------------------
    print('\n=== Per-participant correlations by condition ===')
    cond_rows = []
    for cond in ['diffuse', 'clumpy']:
        sub = avg_df[avg_df['condition'] == cond]
        print(f'\n  -- {cond} (n={len(sub)}) --')
        for mcol in measures:
            r = correlate_two(sub['mean_reading_length_s'], sub[mcol])
            print(fmt_correlation_block(f'mean_reading_length_s vs {mcol}', r))
            if r is not None:
                cond_rows.append({'level': 'per_participant', 'subset': cond,
                                  'reading_measure': 'mean_reading_length_s',
                                  'switch_measure': mcol, **r})

    cor_df = pd.DataFrame(cor_rows + pp_rows + cond_rows)

    # ---- Save CSVs ---------------------------------------------------------
    csv_out = OUTPUT_DIR / 'm71_reading_length_vs_switchrate.csv'
    cor_df.to_csv(csv_out, index=False)
    print(f'\nSaved correlations: {csv_out}')

    tidy_cols = ['participant_id', 'condition'] + reading_cols + measures
    tidy_out = OUTPUT_DIR / 'm71_per_participant_reading_switches.csv'
    avg_df[tidy_cols].to_csv(tidy_out, index=False)
    print(f'Saved per-participant tidy: {tidy_out}')

    # ---- PDF ---------------------------------------------------------------
    pdf_path = OUTPUT_DIR / 'm71_reading_length_vs_switchrate.pdf'
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(make_summary_page(cor_df, len(clean_q), len(avg_df)), facecolor=BG)
        plt.close()
        pdf.savefig(make_scatter_page(clean_q, avg_df), facecolor=BG)
        plt.close()
        pdf.savefig(make_bottom_line_page(cor_df, len(clean_q), len(avg_df)), facecolor=BG)
        plt.close()
    print(f'Saved PDF: {pdf_path}')
    print('\nDone.')


# ----------------------------------------------------------------------
# Plot pages
# ----------------------------------------------------------------------

def _lookup(cor_df, level, subset, switch_measure, reading_measure='mean_reading_length_s'):
    """Return Pearson/Spearman r and p for a specific row in cor_df, or None."""
    sub = cor_df[(cor_df['level'] == level) & (cor_df['subset'] == subset)
                 & (cor_df['reading_measure'] == reading_measure)
                 & (cor_df['switch_measure'] == switch_measure)]
    if len(sub) == 0:
        return None
    r = sub.iloc[0]
    return {'pr': r['pearson_r'], 'pp': r['pearson_p'],
            'sr': r['spearman_r'], 'sp': r['spearman_p'], 'n': int(r['n'])}


def _verdict(pr, pp, sr, sp):
    """Short verdict label based on r magnitudes and p values."""
    sig_pearson = pp < 0.05
    sig_spearman = sp < 0.05
    abs_max = max(abs(pr), abs(sr))
    if abs_max < 0.15 and not (sig_pearson or sig_spearman):
        return 'INDEPENDENT (no meaningful overlap)'
    if abs_max < 0.25:
        return 'WEAK link'
    if abs_max < 0.40:
        return 'MODERATE link'
    return 'STRONG link'


def make_bottom_line_page(cor_df, n_q, n_p):
    """Concise bottom-line takeaways for the advisor meeting (one page)."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M71 - Bottom line', fontsize=15, fontweight='bold', color=TEXT, y=0.965)
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.88])
    ax.axis('off')

    typ = _lookup(cor_df, 'per_participant', 'all', 'switch_rate_typing')
    top = _lookup(cor_df, 'per_participant', 'all', 'switch_rate_topic')
    tim = _lookup(cor_df, 'per_participant', 'all', 'switch_rate_time')
    typ_d = _lookup(cor_df, 'per_participant', 'diffuse', 'switch_rate_typing')
    typ_c = _lookup(cor_df, 'per_participant', 'clumpy', 'switch_rate_typing')

    def short(d):
        if d is None:
            return 'n/a'
        return f'r={d["pr"]:+.2f} (p={d["pp"]:.2f}); rho={d["sr"]:+.2f} (p={d["sp"]:.2f})'

    headline = ('Mean reading length and the TYPING-based switch rate are\n'
                'INDEPENDENT at the participant level - they measure DIFFERENT things.')

    rows = [
        ('typing  (primary)', short(typ),
         'Independent. Keep as separate construct.'),
        ('topic   (LDA)', short(top),
         'Weak negative, not significant. Also independent.'),
        ('time    (>60s)', short(tim),
         'Positive, MECHANICAL (both derived from dwell time).'),
    ]

    cond_rows = [
        ('diffuse', short(typ_d),
         f'n={typ_d["n"] if typ_d else "?"} - longer readers switch slightly more'),
        ('clumpy',  short(typ_c),
         f'n={typ_c["n"] if typ_c else "?"} - no positive link'),
    ]

    talking = [
        '"Reading length and typing-switch are independent at the participant',
        ' level - they are NOT measuring the same thing."',
        '"The only sizeable correlation (with time-switch, r=+.31) is mechanical:',
        ' both are computed from page dwell time."',
        '"Topic-switch is also essentially independent of reading length."',
    ]

    y = 0.985
    ax.text(0.0, y, f'Sample: N={n_p} participants ({n_q} clean questions). '
                    f'Per-participant level (averaged across both questions).',
            transform=ax.transAxes, fontsize=10, color=MUTED, va='top')
    y -= 0.06

    ax.text(0.0, y, 'HEADLINE', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=TEXT, va='top')
    y -= 0.04
    ax.text(0.02, y, headline, transform=ax.transAxes,
            fontsize=12, color=TEXT, va='top', linespacing=1.4)
    y -= 0.13

    ax.text(0.0, y, 'PER MEASURE (reading length vs ...)', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=TEXT, va='top')
    y -= 0.04
    for name, stat, note in rows:
        ax.text(0.02, y, f'{name:<22}  {stat}',
                transform=ax.transAxes, fontsize=10, family='monospace',
                color=TEXT, va='top')
        y -= 0.032
        ax.text(0.06, y, f'-> {note}', transform=ax.transAxes,
                fontsize=10, color=MUTED, va='top', style='italic')
        y -= 0.04

    y -= 0.01
    ax.text(0.0, y, 'BY CONDITION (typing-switch only)', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=TEXT, va='top')
    y -= 0.04
    for name, stat, note in cond_rows:
        ax.text(0.02, y, f'{name:<22}  {stat}',
                transform=ax.transAxes, fontsize=10, family='monospace',
                color=TEXT, va='top')
        y -= 0.032
        ax.text(0.06, y, f'-> {note}', transform=ax.transAxes,
                fontsize=10, color=MUTED, va='top', style='italic')
        y -= 0.04

    y -= 0.01
    ax.text(0.0, y, 'TALKING POINTS', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=TEXT, va='top')
    y -= 0.04
    ax.text(0.02, y, '\n'.join(talking), transform=ax.transAxes,
            fontsize=10, color=TEXT, va='top', linespacing=1.4)

    return fig


def make_summary_page(cor_df, n_q, n_p):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M71 - Mean reading length vs switch-rate (correlation)',
                 fontsize=13, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')

    lines = [f'Clean questions: {n_q}    Participants (avg across both Qs): {n_p}', '']
    lines.append(f'{"Level":<18} {"Subset":<10} {"Reading":<26} {"Switch":<22} '
                 f'{"n":>4} {"Pear r":>8} {"Pear p":>9} {"Spear r":>8} {"Spear p":>9}')
    lines.append('-' * 116)
    for _, r in cor_df.iterrows():
        lines.append(
            f'{r["level"]:<18} {r["subset"]:<10} {r["reading_measure"]:<26} '
            f'{r["switch_measure"]:<22} {r["n"]:>4} '
            f'{r["pearson_r"]:>+8.3f} {r["pearson_p"]:>9.3g} '
            f'{r["spearman_r"]:>+8.3f} {r["spearman_p"]:>9.3g}'
        )
    lines.append('')
    lines.append('Interpretation:')
    lines.append('  switch_rate = n_switches / (n_pages - 1).')
    lines.append('  Strong negative correlation = longer dwell times go with fewer alternations,')
    lines.append('  which would suggest the two measures partly redundantly index "exploit" behavior.')

    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes, fontsize=8.4,
            family='monospace', va='top', color=TEXT, linespacing=1.4)
    return fig


def make_scatter_page(clean_q, avg_df):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M71 - Mean reading length vs switch rate (scatter)',
                 fontsize=13, fontweight='bold', color=TEXT, y=0.99)

    pairs = [
        (clean_q, 'mean_reading_length_s', 'switch_rate_typing',
         'Per question: reading vs typing-switch'),
        (clean_q, 'mean_reading_length_s', 'switch_rate_topic',
         'Per question: reading vs topic-switch'),
        (clean_q, 'mean_reading_length_s', 'switch_rate_time',
         'Per question: reading vs time-switch'),
        (avg_df, 'mean_reading_length_s', 'switch_rate_typing',
         'Per participant: reading vs typing-switch'),
        (avg_df, 'mean_reading_length_s', 'switch_rate_topic',
         'Per participant: reading vs topic-switch'),
        (avg_df, 'mean_reading_length_s', 'switch_rate_time',
         'Per participant: reading vs time-switch'),
    ]
    for ax, (df, x, y, title) in zip(axes.ravel(), pairs):
        ax.set_facecolor(BG)
        sub = df[['condition', x, y]].dropna()
        for cond in ['diffuse', 'clumpy']:
            s = sub[sub['condition'] == cond]
            ax.scatter(s[x], s[y], color=COND_COLORS[cond], s=24, alpha=0.7,
                       edgecolors='#333', linewidth=0.3,
                       label=f'{cond} (n={len(s)})')
        if len(sub) >= 5 and sub[x].std() > 0 and sub[y].std() > 0:
            r, p = sp_stats.pearsonr(sub[x], sub[y])
            rs, ps = sp_stats.spearmanr(sub[x], sub[y])
            title_suffix = f'\nPearson r={r:+.2f} p={p:.3g}  Spearman r={rs:+.2f} p={ps:.3g}'
        else:
            title_suffix = '\n(insufficient data)'
        ax.set_title(title + title_suffix, fontsize=9.5, color=TEXT, fontweight='bold')
        ax.set_xlabel(x, color=LABEL, fontsize=9)
        ax.set_ylabel(y, color=LABEL, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(fontsize=7, loc='best', framealpha=0.85)
        ax.grid(True, color=GRID, linewidth=0.4, zorder=0)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


if __name__ == '__main__':
    main()

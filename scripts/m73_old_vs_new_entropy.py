#!/usr/bin/env python3
"""
M73: Old vs New sample comparison - Writing entropy
====================================================
Same structure as M72 but focused on a single measure:
seq_typing_entropy (Shannon entropy of the typing/no-typing per-page binary
sequence within each trial; matches m56's seq_typing_entropy).

Exclusions are kept identical to M52/M72 (<3 pages, idle>=50%, 3SD outliers
on the three count variables) so that the participant set is the same as
M72 - just the measure differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats

from helpers import load_trials, OUTPUT_DIR
from m18_typing_binary import page_had_typing_or_paste

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

THRESHOLD_S = 60.0
IDLE_THRESHOLD_PCT = 50.0
MIN_PAGE_VISITS = 3
OUTLIER_SD = 3

MEANINGFUL_ACTIONS = ['article_open', 'search', 'link_click', 'back_navigation', 'paste']
SNAPSHOT_ACTIONS = ['answer_snapshot', 'answer_snapshot_cursor_leave']

# Light palette
BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'
COND_COLORS = {'diffuse': '#1976D2', 'clumpy': '#C62828'}


# ---------- per-trial helpers -------------------------------------------------

def load_lda_assignments(topic_model_path: Path):
    with open(topic_model_path, encoding='utf-8') as f:
        tm = json.load(f)
    return {slug.replace('_', ' '): int(np.argmax(dist))
            for slug, dist in tm['topic_distributions'].items()}


def compute_idle_pct(events_df, t0, t_end):
    total_sec = (t_end - t0).total_seconds()
    if total_sec <= 0:
        return np.nan
    meaningful = events_df[events_df['Action'].isin(MEANINGFUL_ACTIONS)]
    snapshots = events_df[events_df['Action'].isin(SNAPSHOT_ACTIONS)].copy()
    if len(snapshots) > 1:
        snapshots['prev_len'] = snapshots['AnswerLength'].shift(1)
        ws = snapshots[snapshots['AnswerLength'] != snapshots['prev_len']]
    else:
        ws = snapshots.iloc[:0]
    am = pd.concat([meaningful, ws]).sort_values('Time')
    am = am[(am['Time'] >= t0) & (am['Time'] <= t_end)]
    if len(am) == 0:
        return 100.0
    last = am['Time'].iloc[-1]
    return ((t_end - last).total_seconds() / total_sec) * 100


def switch_count(labels):
    if len(labels) < 2:
        return np.nan
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def shannon_entropy_binary(labels):
    """Bits 0..1. Matches m56.shannon_entropy_binary exactly."""
    if len(labels) == 0:
        return np.nan
    arr = np.asarray(labels)
    p1 = (arr == arr[0]).mean()
    p2 = 1 - p1
    if p1 in (0, 1):
        return 0.0
    return -(p1 * np.log2(p1) + p2 * np.log2(p2))


# ---------- per-sample pipeline ----------------------------------------------

@dataclass
class SampleResult:
    label: str
    n_questions_total: int
    n_questions_kept: int
    n_outliers: int
    avg_df: pd.DataFrame


def build_question_df(trials, lda_assignments):
    rows = []
    for tr in trials:
        if tr['domain'] == 'practice':
            continue
        pvs = tr['page_visits']
        n_pages = len(pvs)
        t0 = tr['t0']
        t_end = t0 + pd.Timedelta(seconds=tr['duration'])
        idle_pct = compute_idle_pct(tr['events'], t0, t_end)

        time_lbl = ['exploit' if pv['duration'] > THRESHOLD_S else 'explore' for pv in pvs]
        topic_lbl = [lda_assignments.get(pv['title'], -1) for pv in pvs]
        typing_lbl = [page_had_typing_or_paste(pv, tr['typing_intervals'], tr['paste_times'])
                      for pv in pvs]

        rows.append({
            'participant_id': tr['pid'],
            'condition': tr['condition'],
            'domain': tr['domain'],
            'n_pages': n_pages,
            'idle_pct': idle_pct,
            'excluded_pages': n_pages < MIN_PAGE_VISITS,
            'excluded_idle': idle_pct >= IDLE_THRESHOLD_PCT,
            'count_time': switch_count(time_lbl),
            'count_topic': switch_count(topic_lbl),
            'count_typing': switch_count(typing_lbl),
            'seq_typing_entropy': shannon_entropy_binary(typing_lbl),
        })
    return pd.DataFrame(rows)


def run_pipeline(game_csv: Path, topic_model: Path, label: str) -> SampleResult:
    print(f'\n=== {label} ===')
    trials = load_trials(game_csv)
    lda = load_lda_assignments(topic_model)
    qdf = build_question_df(trials, lda)
    n_total = len(qdf)

    excluded = qdf['excluded_pages'] | qdf['excluded_idle']
    clean = qdf[~excluded].copy()
    print(f'  Questions kept: {len(clean)} / {n_total}')

    measures = ['count_time', 'count_topic', 'count_typing', 'seq_typing_entropy']
    avg = (clean.groupby(['participant_id', 'condition'], as_index=False)
                [measures].mean())
    avg = avg.dropna(subset=['count_time', 'count_topic', 'count_typing'])

    # 3 SD outlier filter on the three count variables (same as M52/M72)
    out_mask = pd.Series(False, index=avg.index)
    for col in ['count_time', 'count_topic', 'count_typing']:
        m, s = avg[col].mean(), avg[col].std()
        out_mask = out_mask | (avg[col] < m - OUTLIER_SD * s) | (avg[col] > m + OUTLIER_SD * s)
    n_out = int(out_mask.sum())
    if n_out:
        excluded_pids = sorted(avg.loc[out_mask, 'participant_id'].tolist())
        print(f'  3SD outliers ({n_out}): {excluded_pids}')
    final = avg[~out_mask].reset_index(drop=True)
    print(f'  Final N: {len(final)} '
          f'({(final["condition"]=="diffuse").sum()} diffuse / '
          f'{(final["condition"]=="clumpy").sum()} clumpy)')
    return SampleResult(label, n_total, len(clean), n_out, final)


# ---------- comparison stats --------------------------------------------------

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) /
                     (len(a) + len(b) - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def condition_stats(avg_df, var):
    d = avg_df.loc[avg_df['condition'] == 'diffuse', var].dropna().to_numpy()
    c = avg_df.loc[avg_df['condition'] == 'clumpy', var].dropna().to_numpy()
    if len(d) >= 2 and len(c) >= 2:
        t, p = sp_stats.ttest_ind(d, c, equal_var=False)
        cd = cohens_d(d, c)
    else:
        t, p, cd = np.nan, np.nan, np.nan
    return {
        'n_diffuse': len(d), 'mean_diffuse': float(np.mean(d)),
        'sd_diffuse': float(np.std(d, ddof=1)) if len(d) >= 2 else np.nan,
        'n_clumpy': len(c), 'mean_clumpy': float(np.mean(c)),
        'sd_clumpy': float(np.std(c, ddof=1)) if len(c) >= 2 else np.nan,
        'diff_d_minus_c': float(np.mean(d) - np.mean(c)),
        't': float(t), 'p': float(p), 'cohen_d': float(cd),
    }


# ---------- PDF ---------------------------------------------------------------

def _verdict(d_old, d_new, threshold=0.05):
    delta = abs(d_new) - abs(d_old)
    if abs(delta) < threshold:
        return 'unchanged', delta
    return ('strengthened' if delta > 0 else 'weakened'), delta


def _table_page(old, new, old_stats, new_stats):
    fig = plt.figure(figsize=(11, 4.0))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f'Writing entropy: Diffuse vs Clumpy across samples\n'
        f'N_old = {len(old.avg_df)}     N_new = {len(new.avg_df)}',
        fontsize=14, fontweight='bold', color=TEXT, y=0.93,
    )

    ax = fig.add_subplot(111)
    ax.axis('off')

    verdict, delta = _verdict(old_stats['cohen_d'], new_stats['cohen_d'])
    headers = ['Measure', 'd OLD', 'd NEW', 'Δ|d|', 'Verdict']
    body = [[
        'Writing entropy',
        f'{old_stats["cohen_d"]:+.2f}',
        f'{new_stats["cohen_d"]:+.2f}',
        f'{delta:+.2f}',
        verdict,
    ]]

    table = ax.table(cellText=body, colLabels=headers, loc='center',
                     cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.4)

    for col in range(len(headers)):
        cell = table[(0, col)]
        cell.set_facecolor('#1976D2')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor(BORDER)

    verdict_colors = {'strengthened': '#2E7D32',
                      'weakened': '#C62828',
                      'unchanged': MUTED}
    table[(1, 0)].set_text_props(ha='left')
    table[(1, 4)].set_text_props(color=verdict_colors[verdict], fontweight='bold')
    for col in range(len(headers)):
        table[(1, col)].set_edgecolor(BORDER)

    fig.text(0.5, 0.06,
             "d = signed Cohen's d for Diffuse - Clumpy (negative = clumpy higher).   "
             "Verdict based on |d|: strengthened if |d| grew by >= 0.05, weakened if it dropped by >= 0.05.",
             ha='center', fontsize=9, color=MUTED, style='italic')
    return fig


def _raincloud_page(sample, stats, sample_label):
    """Single-page raincloud for entropy by condition (M69-style layout)."""
    from m69_email_summary_pdf_violin import _raincloud_panel

    n_total = len(sample.avg_df)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f'{sample_label} sample (N = {n_total})\n'
        'Writing entropy by condition',
        fontsize=15, fontweight='bold', color=TEXT, y=0.965,
    )

    # Single banner with the entropy result
    banner_y = 0.69
    banner_h = 0.19
    sig_color = '#1B5E20' if stats['p'] < 0.05 else '#888888'
    box_face = '#E8F5E9' if stats['p'] < 0.05 else '#F5F5F5'
    box_edge = '#2E7D32' if stats['p'] < 0.05 else '#999999'

    ax_b = fig.add_axes([0.20, banner_y, 0.60, banner_h])
    ax_b.axis('off')
    ax_b.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_b.transAxes,
        facecolor=box_face, edgecolor=box_edge, linewidth=1.5,
    ))
    ax_b.text(0.5, 0.84, 'Writing entropy (typing / no-typing per page)',
              transform=ax_b.transAxes, ha='center', va='top',
              fontsize=12, fontweight='bold', color=sig_color)
    ax_b.text(0.5, 0.55,
              f'Diffuse (n={stats["n_diffuse"]}): M = {stats["mean_diffuse"]:.2f}     '
              f'Clumpy (n={stats["n_clumpy"]}): M = {stats["mean_clumpy"]:.2f}',
              transform=ax_b.transAxes, ha='center', va='center',
              fontsize=10.5, color=MUTED)
    ax_b.text(0.5, 0.30,
              f'd = {stats["cohen_d"]:+.2f},  p = {stats["p"]:.3f}',
              transform=ax_b.transAxes, ha='center', va='center',
              fontsize=14, fontweight='bold', color=sig_color)
    direction = ('Clumpy > Diffuse' if stats['cohen_d'] < 0
                 else 'Diffuse > Clumpy' if stats['cohen_d'] > 0
                 else 'no difference')
    ax_b.text(0.5, 0.08, direction,
              transform=ax_b.transAxes, ha='center', va='center',
              fontsize=10.5, fontweight='bold', color=sig_color)

    # Raincloud
    ax_rc = fig.add_axes([0.28, 0.18, 0.46, 0.42])
    d_vals = sample.avg_df.loc[sample.avg_df['condition'] == 'diffuse',
                               'seq_typing_entropy'].dropna().to_numpy()
    c_vals = sample.avg_df.loc[sample.avg_df['condition'] == 'clumpy',
                               'seq_typing_entropy'].dropna().to_numpy()
    _raincloud_panel(
        ax_rc,
        groups=[d_vals, c_vals],
        group_labels=[f'Diffuse (n={len(d_vals)})', f'Clumpy (n={len(c_vals)})'],
        group_colors=[COND_COLORS['diffuse'], COND_COLORS['clumpy']],
        ylabel='Writing entropy (bits, 0 to 1)',
        title=('Higher = writing distributed evenly across reading and writing pages\n'
               'Lower = pages are more uniform (mostly all-reading or all-writing)'),
    )

    return fig


def write_pdf(pdf_path: Path, old, new, old_stats, new_stats):
    with PdfPages(pdf_path) as pdf:
        for fig in [
            _table_page(old, new, old_stats, new_stats),
            _raincloud_page(old, old_stats, 'OLD'),
            _raincloud_page(new, new_stats, 'NEW'),
        ]:
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f'Saved PDF: {pdf_path}')


# ---------- main --------------------------------------------------------------

def hebrew_summary(old, new, old_stats, new_stats):
    verdict, delta = _verdict(old_stats['cohen_d'], new_stats['cohen_d'])
    he_verdict = {'strengthened': 'התחזק',
                  'weakened': 'נחלש',
                  'unchanged': 'נשאר ללא שינוי'}[verdict]
    lines = []
    lines.append('=' * 70)
    lines.append('M73 - אנטרופיית כתיבה: השוואה ישן מול חדש')
    lines.append('=' * 70)
    lines.append(f'מדגם ישן  N = {len(old.avg_df)}')
    lines.append(f'מדגם חדש  N = {len(new.avg_df)}')
    lines.append('')
    lines.append('הבדל בין תנאים (Diffuse - Clumpy):')
    lines.append(f'  ישן: D={old_stats["mean_diffuse"]:.2f}  C={old_stats["mean_clumpy"]:.2f}  '
                 f'd={old_stats["cohen_d"]:+.2f}  p={old_stats["p"]:.3f}')
    lines.append(f'  חדש: D={new_stats["mean_diffuse"]:.2f}  C={new_stats["mean_clumpy"]:.2f}  '
                 f'd={new_stats["cohen_d"]:+.2f}  p={new_stats["p"]:.3f}')
    lines.append('')
    lines.append(f'Δ|d| = {delta:+.2f}   ->   האפקט {he_verdict}')
    return '\n'.join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    old = run_pipeline(
        DATA_DIR / 'cleaned' / 'Game.csv',
        DATA_DIR / 'cleaned' / 'topic_model.json',
        'OLD sample',
    )
    new = run_pipeline(
        DATA_DIR / 'cleaned_new' / 'Game.csv',
        DATA_DIR / 'cleaned_new' / 'topic_model.json',
        'NEW sample',
    )

    old_stats = condition_stats(old.avg_df, 'seq_typing_entropy')
    new_stats = condition_stats(new.avg_df, 'seq_typing_entropy')

    # CSV
    rows = []
    for label, sr, st in [('OLD', old, old_stats), ('NEW', new, new_stats)]:
        rows.append({'sample': label, 'n_total': len(sr.avg_df),
                     'n_diffuse': st['n_diffuse'], 'mean_diffuse': st['mean_diffuse'],
                     'sd_diffuse': st['sd_diffuse'],
                     'n_clumpy': st['n_clumpy'], 'mean_clumpy': st['mean_clumpy'],
                     'sd_clumpy': st['sd_clumpy'],
                     'diff_d_minus_c': st['diff_d_minus_c'],
                     't': st['t'], 'p': st['p'], 'cohen_d': st['cohen_d']})
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / 'm73_entropy_old_vs_new.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Saved CSV: {csv_path}')

    # Per-participant
    old.avg_df[['participant_id', 'condition', 'seq_typing_entropy']].to_csv(
        OUTPUT_DIR / 'm73_old_per_participant_entropy.csv', index=False)
    new.avg_df[['participant_id', 'condition', 'seq_typing_entropy']].to_csv(
        OUTPUT_DIR / 'm73_new_per_participant_entropy.csv', index=False)

    # PDF
    write_pdf(OUTPUT_DIR / 'm73_entropy_old_vs_new.pdf', old, new, old_stats, new_stats)

    # Hebrew summary
    summary = hebrew_summary(old, new, old_stats, new_stats)
    summary_path = OUTPUT_DIR / 'm73_summary_he.txt'
    summary_path.write_text(summary, encoding='utf-8')
    print(f'Saved Hebrew summary: {summary_path}')
    try:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        print('\n' + summary)
    except (UnicodeEncodeError, AttributeError):
        print('(summary contains Hebrew, printed only to file)')


if __name__ == '__main__':
    main()

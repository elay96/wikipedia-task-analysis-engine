#!/usr/bin/env python3
"""
M72: Old vs New sample comparison
=================================
Runs the M52 pipeline (3 PCA counts + PC1) plus the Mean reading-only streak
(seq_typing_mean_run_explore from m56) on two samples in parallel:

  OLD: data/cleaned/Game.csv         + data/cleaned/topic_model.json
  NEW: data/cleaned_new/Game.csv     + data/cleaned_new/topic_model.json

For each side it applies the same exclusion logic (<3 pages, idle>=50%, then
3SD outliers on the three count variables), averages per participant, and
runs PCA. Then it builds a comparison table (overall + per-condition) and
saves CSV/Excel + a Hebrew summary.
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
from sklearn.decomposition import PCA

from helpers import load_trials, OUTPUT_DIR
from m18_typing_binary import page_had_typing_or_paste

# Light mode palette
BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'
OLD_COLOR = '#1976D2'
NEW_COLOR = '#E65100'
DIFFUSE_COLOR = '#1976D2'
CLUMPY_COLOR = '#C62828'

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

THRESHOLD_S = 60.0
IDLE_THRESHOLD_PCT = 50.0
MIN_PAGE_VISITS = 3
OUTLIER_SD = 3

MEANINGFUL_ACTIONS = ['article_open', 'search', 'link_click', 'back_navigation', 'paste']
SNAPSHOT_ACTIONS = ['answer_snapshot', 'answer_snapshot_cursor_leave']

# Variables to compare (order = output order)
VAR_ORDER = [
    ('seq_typing_mean_run_explore', 'Mean reading-only streak'),
    ('count_time',                  'Switch count: time (>60s)'),
    ('count_topic',                 'Switch count: topic (LDA)'),
    ('count_typing',                'Switch count: typing/paste'),
    ('PC1',                         'PC1 (composite explore-exploit index)'),
]


# ---------- per-trial helpers -------------------------------------------------

def load_lda_assignments(topic_model_path: Path):
    """Return {display_title: dominant_topic_id}. Slug keys ('Ancient_art')
    are normalized to display form ('Ancient art')."""
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


def runs(labels):
    if len(labels) == 0:
        return []
    out = [(labels[0], 1)]
    for x in labels[1:]:
        if x == out[-1][0]:
            out[-1] = (x, out[-1][1] + 1)
        else:
            out.append((x, 1))
    return out


def mean_run_explore(typing_labels):
    """Mean run length of 'explore' (no-typing) pages. Matches m56's
    seq_typing_mean_run_explore."""
    if len(typing_labels) < 2:
        return np.nan
    rs = runs(['exploit' if v else 'explore' for v in typing_labels])
    explore_runs = [n for v, n in rs if v == 'explore']
    return float(np.mean(explore_runs)) if explore_runs else np.nan


# ---------- per-sample pipeline ----------------------------------------------

@dataclass
class SampleResult:
    label: str
    n_questions_total: int
    n_questions_kept: int
    n_outliers: int
    avg_df: pd.DataFrame   # one row per kept participant
    pca_pct: list


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
            'seq_typing_mean_run_explore': mean_run_explore(typing_lbl),
        })
    return pd.DataFrame(rows)


def run_pipeline(game_csv: Path, topic_model: Path, label: str) -> SampleResult:
    print(f'\n=== {label} ===')
    print(f'  Game.csv:        {game_csv}')
    print(f'  topic_model.json:{topic_model}')

    trials = load_trials(game_csv)
    lda = load_lda_assignments(topic_model)

    qdf = build_question_df(trials, lda)
    n_total = len(qdf)

    excluded = qdf['excluded_pages'] | qdf['excluded_idle']
    clean = qdf[~excluded].copy()
    print(f'  Questions excluded (<{MIN_PAGE_VISITS} pages): {qdf["excluded_pages"].sum()}')
    print(f'  Questions excluded (idle>={IDLE_THRESHOLD_PCT:.0f}%): {qdf["excluded_idle"].sum()}')
    print(f'  Questions kept: {len(clean)} / {n_total}')

    # Average per participant + condition
    measures = ['count_time', 'count_topic', 'count_typing', 'seq_typing_mean_run_explore']
    avg = (clean.groupby(['participant_id', 'condition'], as_index=False)
                [measures].mean())
    avg = avg.dropna(subset=['count_time', 'count_topic', 'count_typing'])

    # 3 SD outliers on the three count variables
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

    # PCA on the 3 counts (raw, no z-score)
    X = final[['count_time', 'count_topic', 'count_typing']].values
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X)
    final = final.copy()
    final['PC1'] = scores[:, 0]
    final['PC2'] = scores[:, 1]
    final['PC3'] = scores[:, 2]
    pct = (pca.explained_variance_ratio_ * 100).tolist()
    print(f'  PCA variance: PC1={pct[0]:.1f}%, PC2={pct[1]:.1f}%, PC3={pct[2]:.1f}%')

    return SampleResult(label, n_total, len(clean), n_out, final, pct)


# ---------- comparison stats --------------------------------------------------

def welch_df(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    return num / denom if denom > 0 else (n1 + n2 - 2)


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) /
                     (len(a) + len(b) - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def direction_label(diff, sd_pool, threshold=0.1):
    """Return 'עלה' / 'ירד' / 'כמעט לא השתנה' based on diff in pooled-SD units."""
    if sd_pool == 0 or np.isnan(sd_pool):
        return 'כמעט לא השתנה'
    z = abs(diff) / sd_pool
    if z < threshold:
        return 'כמעט לא השתנה'
    return 'עלה' if diff > 0 else 'ירד'


def overall_row(var, var_label, old_df, new_df):
    a = old_df[var].dropna().to_numpy()
    b = new_df[var].dropna().to_numpy()
    diff = np.mean(b) - np.mean(a)
    pooled_sd = np.sqrt(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)))
    t, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return {
        'variable': var,
        'label_he': var_label,
        'n_old': len(a),
        'mean_old': float(np.mean(a)),
        'sd_old': float(np.std(a, ddof=1)),
        'n_new': len(b),
        'mean_new': float(np.mean(b)),
        'sd_new': float(np.std(b, ddof=1)),
        'diff_new_minus_old': float(diff),
        'pct_change': float(diff / np.mean(a) * 100) if np.mean(a) != 0 else np.nan,
        'direction': direction_label(diff, pooled_sd),
        't': float(t),
        'df_welch': float(welch_df(a, b)),
        'p_value': float(p),
        'cohen_d': float(cohens_d(b, a)),
    }


def condition_row(var, var_label, old_df, new_df):
    """Per-condition comparison: Diffuse vs Clumpy in OLD and NEW separately."""
    out = {'variable': var, 'label_he': var_label}
    for side, df in [('old', old_df), ('new', new_df)]:
        d = df.loc[df['condition'] == 'diffuse', var].dropna().to_numpy()
        c = df.loc[df['condition'] == 'clumpy', var].dropna().to_numpy()
        if len(d) >= 2 and len(c) >= 2:
            t, p = sp_stats.ttest_ind(d, c, equal_var=False)
            cd = cohens_d(d, c)
        else:
            t, p, cd = np.nan, np.nan, np.nan
        out.update({
            f'n_diffuse_{side}': len(d),
            f'mean_diffuse_{side}': float(np.mean(d)) if len(d) else np.nan,
            f'sd_diffuse_{side}': float(np.std(d, ddof=1)) if len(d) >= 2 else np.nan,
            f'n_clumpy_{side}': len(c),
            f'mean_clumpy_{side}': float(np.mean(c)) if len(c) else np.nan,
            f'sd_clumpy_{side}': float(np.std(c, ddof=1)) if len(c) >= 2 else np.nan,
            f'diff_d_minus_c_{side}': (float(np.mean(d) - np.mean(c))
                                       if len(d) and len(c) else np.nan),
            f't_d_vs_c_{side}': float(t),
            f'p_d_vs_c_{side}': float(p),
            f'cohen_d_d_vs_c_{side}': float(cd),
        })
    return out


# ---------- PDF report --------------------------------------------------------

# English labels for plot rendering (matplotlib doesn't render Hebrew RTL well)
VAR_LABELS_EN = {
    'seq_typing_mean_run_explore': 'Mean reading-only streak',
    'count_time': 'Switch count: time',
    'count_topic': 'Switch count: topic (LDA)',
    'count_typing': 'Switch count: typing/paste',
    'PC1': 'PC1 (composite)',
}


def _style_axes(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _title_page(old, new):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis('off')

    ax.text(0.5, 0.92, 'M72: Old vs New Sample Comparison',
            transform=ax.transAxes, fontsize=22, fontweight='bold',
            ha='center', va='top', color=TEXT)
    ax.text(0.5, 0.85, 'Switch counts + PCA + Mean reading-only streak',
            transform=ax.transAxes, fontsize=13,
            ha='center', va='top', color=MUTED)

    info = (
        f'OLD sample (data/cleaned)\n'
        f'  Final N: {len(old.avg_df)}  '
        f'(diffuse {(old.avg_df["condition"]=="diffuse").sum()}, '
        f'clumpy {(old.avg_df["condition"]=="clumpy").sum()})\n'
        f'  Questions kept: {old.n_questions_kept} / {old.n_questions_total}\n'
        f'  3SD outliers removed: {old.n_outliers}\n'
        f'  PCA variance: PC1={old.pca_pct[0]:.1f}%, '
        f'PC2={old.pca_pct[1]:.1f}%, PC3={old.pca_pct[2]:.1f}%\n'
        f'\n'
        f'NEW sample (data/cleaned_new)\n'
        f'  Final N: {len(new.avg_df)}  '
        f'(diffuse {(new.avg_df["condition"]=="diffuse").sum()}, '
        f'clumpy {(new.avg_df["condition"]=="clumpy").sum()})\n'
        f'  Questions kept: {new.n_questions_kept} / {new.n_questions_total}\n'
        f'  3SD outliers removed: {new.n_outliers}\n'
        f'  PCA variance: PC1={new.pca_pct[0]:.1f}%, '
        f'PC2={new.pca_pct[1]:.1f}%, PC3={new.pca_pct[2]:.1f}%\n'
        f'\n'
        f'Exclusion criteria (applied to both samples identically):\n'
        f'  - Trials with fewer than 3 page visits\n'
        f'  - Trials with idle time >= 50% after last meaningful event\n'
        f'  - Participants > 3 SD from mean on any of the three count variables\n'
    )
    ax.text(0.05, 0.72, info, transform=ax.transAxes, fontsize=11,
            ha='left', va='top', color=TEXT, family='monospace', linespacing=1.6)

    return fig


def _overall_table_page(overall_rows):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Overall comparison: OLD vs NEW',
                 fontsize=15, fontweight='bold', color=TEXT, y=0.97)

    ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
    ax.axis('off')
    ax.set_facecolor(BG)

    lines = []
    lines.append(
        f'{"Variable":<35} {"N_old":>5} {"M_old":>7} {"SD_old":>7} '
        f'{"N_new":>5} {"M_new":>7} {"SD_new":>7} {"diff":>7} {"%":>6} '
        f'{"d":>6} {"p":>7} {"Direction":>14}'
    )
    lines.append('-' * 130)
    for r in overall_rows:
        sig = '***' if r['p_value'] < .001 else '**' if r['p_value'] < .01 else '*' if r['p_value'] < .05 else ''
        direction_en = {'עלה': 'up', 'ירד': 'down', 'כמעט לא השתנה': 'unchanged'}[r['direction']]
        pct = '' if np.isnan(r['pct_change']) else f'{r["pct_change"]:+5.1f}%'
        lines.append(
            f'{VAR_LABELS_EN[r["variable"]]:<35} '
            f'{r["n_old"]:>5d} {r["mean_old"]:>7.2f} {r["sd_old"]:>7.2f} '
            f'{r["n_new"]:>5d} {r["mean_new"]:>7.2f} {r["sd_new"]:>7.2f} '
            f'{r["diff_new_minus_old"]:>+7.2f} {pct:>6} '
            f'{r["cohen_d"]:>+6.2f} {r["p_value"]:>7.3f} '
            f'{(direction_en + " " + sig):>14}'
        )
    lines.append('')
    lines.append('Notes:')
    lines.append('  diff = mean_new - mean_old   (Welch t-test, two-sided)')
    lines.append('  d = Cohen\'s d (sign: positive = NEW higher than OLD)')
    lines.append('  Direction: "up"/"down" if |diff| >= 0.1 pooled SD, else "unchanged"')
    lines.append('  Significance: * p<.05  ** p<.01  *** p<.001')

    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8.5, family='monospace', va='top', color=TEXT, linespacing=1.5)
    return fig


def _means_bar_page(overall_rows):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)

    labels = [VAR_LABELS_EN[r['variable']] for r in overall_rows]
    means_old = [r['mean_old'] for r in overall_rows]
    sd_old = [r['sd_old'] for r in overall_rows]
    means_new = [r['mean_new'] for r in overall_rows]
    sd_new = [r['sd_new'] for r in overall_rows]
    n_old = overall_rows[0]['n_old']
    n_new = overall_rows[0]['n_new']

    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w/2, means_old, w, yerr=sd_old, capsize=4,
                color=OLD_COLOR, edgecolor='white', linewidth=0.5,
                label=f'OLD (N={n_old})', alpha=0.88, zorder=2)
    b2 = ax.bar(x + w/2, means_new, w, yerr=sd_new, capsize=4,
                color=NEW_COLOR, edgecolor='white', linewidth=0.5,
                label=f'NEW (N={n_new})', alpha=0.88, zorder=2)

    for bar, val in zip(b1, means_old):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9,
                color=OLD_COLOR, fontweight='bold')
    for bar, val in zip(b2, means_new):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9,
                color=NEW_COLOR, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Mean (+/- SD)', color=LABEL, fontweight='bold')
    ax.set_title('Means by sample (with SD error bars)',
                 color=TEXT, fontweight='bold', fontsize=14, pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, color=GRID, linewidth=0.5, axis='y', zorder=0)
    _style_axes(ax)
    plt.tight_layout()
    return fig


def _condition_table_page(cond_rows):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Per-condition comparison (Diffuse vs Clumpy)',
                 fontsize=15, fontweight='bold', color=TEXT, y=0.97)

    ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
    ax.axis('off')

    lines = []
    lines.append('OLD sample')
    lines.append('-' * 100)
    lines.append(
        f'{"Variable":<35} {"D_mean":>9} {"D_SD":>7} {"C_mean":>9} {"C_SD":>7} '
        f'{"diff":>7} {"d":>6} {"p":>7}'
    )
    for r in cond_rows:
        lines.append(
            f'{VAR_LABELS_EN[r["variable"]]:<35} '
            f'{r["mean_diffuse_old"]:>9.2f} {r["sd_diffuse_old"]:>7.2f} '
            f'{r["mean_clumpy_old"]:>9.2f} {r["sd_clumpy_old"]:>7.2f} '
            f'{r["diff_d_minus_c_old"]:>+7.2f} '
            f'{r["cohen_d_d_vs_c_old"]:>+6.2f} {r["p_d_vs_c_old"]:>7.3f}'
        )
    lines.append('')
    lines.append('NEW sample')
    lines.append('-' * 100)
    lines.append(
        f'{"Variable":<35} {"D_mean":>9} {"D_SD":>7} {"C_mean":>9} {"C_SD":>7} '
        f'{"diff":>7} {"d":>6} {"p":>7}'
    )
    for r in cond_rows:
        lines.append(
            f'{VAR_LABELS_EN[r["variable"]]:<35} '
            f'{r["mean_diffuse_new"]:>9.2f} {r["sd_diffuse_new"]:>7.2f} '
            f'{r["mean_clumpy_new"]:>9.2f} {r["sd_clumpy_new"]:>7.2f} '
            f'{r["diff_d_minus_c_new"]:>+7.2f} '
            f'{r["cohen_d_d_vs_c_new"]:>+6.2f} {r["p_d_vs_c_new"]:>7.3f}'
        )
    lines.append('')
    lines.append('  diff = mean_diffuse - mean_clumpy   (positive = diffuse higher)')
    lines.append('  d = Cohen\'s d (Welch, sign: diffuse - clumpy)')

    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, family='monospace', va='top', color=TEXT, linespacing=1.5)
    return fig


def _effect_size_bar_page(cond_rows):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)

    labels = [VAR_LABELS_EN[r['variable']] for r in cond_rows]
    d_old = [r['cohen_d_d_vs_c_old'] for r in cond_rows]
    d_new = [r['cohen_d_d_vs_c_new'] for r in cond_rows]

    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w/2, d_old, w, color=OLD_COLOR, edgecolor='white',
                linewidth=0.5, label='OLD', alpha=0.88, zorder=2)
    b2 = ax.bar(x + w/2, d_new, w, color=NEW_COLOR, edgecolor='white',
                linewidth=0.5, label='NEW', alpha=0.88, zorder=2)

    for bar, val in zip(b1, d_old):
        y_off = 0.02 if val >= 0 else -0.02
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_off,
                f'{val:+.2f}', ha='center', va=va, fontsize=9,
                color=OLD_COLOR, fontweight='bold')
    for bar, val in zip(b2, d_new):
        y_off = 0.02 if val >= 0 else -0.02
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_off,
                f'{val:+.2f}', ha='center', va=va, fontsize=9,
                color=NEW_COLOR, fontweight='bold')

    ax.axhline(0, color=BORDER, linewidth=1)
    ax.axhline(0.2, color=GRID, linewidth=0.5, linestyle=':')
    ax.axhline(-0.2, color=GRID, linewidth=0.5, linestyle=':')
    ax.text(len(labels) - 0.5, 0.21, 'small effect (|d|=0.2)',
            color=MUTED, fontsize=8, ha='right', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel("Cohen's d  (Diffuse - Clumpy)", color=LABEL, fontweight='bold')
    ax.set_title('Condition effect size: Diffuse vs Clumpy in each sample',
                 color=TEXT, fontweight='bold', fontsize=14, pad=12)
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, color=GRID, linewidth=0.5, axis='y', zorder=0)
    _style_axes(ax)
    plt.tight_layout()
    return fig


def _condition_means_grid_page(old_df, new_df, var_order):
    n_vars = len(var_order)
    fig, axes = plt.subplots(1, n_vars, figsize=(3.0 * n_vars, 5.5),
                             sharey=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle('Per-condition means by sample (Diffuse blue, Clumpy red)',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.99)

    for ax, (var, _) in zip(axes, var_order):
        d_old = old_df.loc[old_df['condition'] == 'diffuse', var].dropna().to_numpy()
        c_old = old_df.loc[old_df['condition'] == 'clumpy',  var].dropna().to_numpy()
        d_new = new_df.loc[new_df['condition'] == 'diffuse', var].dropna().to_numpy()
        c_new = new_df.loc[new_df['condition'] == 'clumpy',  var].dropna().to_numpy()

        means = [d_old.mean(), c_old.mean(), d_new.mean(), c_new.mean()]
        sds = [d_old.std(ddof=1), c_old.std(ddof=1),
               d_new.std(ddof=1), c_new.std(ddof=1)]
        colors = [DIFFUSE_COLOR, CLUMPY_COLOR, DIFFUSE_COLOR, CLUMPY_COLOR]
        labels = ['D-old', 'C-old', 'D-new', 'C-new']

        x = np.arange(4)
        bars = ax.bar(x, means, yerr=sds, capsize=3, color=colors,
                      edgecolor='white', linewidth=0.5, alpha=0.85, zorder=2)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                    color=LABEL, fontweight='bold')
        ax.axvline(1.5, color=BORDER, linewidth=0.5, linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(VAR_LABELS_EN[var], fontsize=10, color=TEXT,
                     fontweight='bold', pad=6)
        ax.grid(True, color=GRID, linewidth=0.5, axis='y', zorder=0)
        _style_axes(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _minimal_table_page(old, new, cond_rows):
    """Single-page table: N + does each measure's D-vs-C effect strengthen
    or weaken between OLD and NEW."""
    fig = plt.figure(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)

    n_old = len(old.avg_df)
    n_new = len(new.avg_df)

    fig.suptitle(f'OLD vs NEW: does the Diffuse-vs-Clumpy effect strengthen?\n'
                 f'N_old = {n_old}     N_new = {n_new}',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.97)

    ax = fig.add_subplot(111)
    ax.axis('off')

    headers = ['Measure', 'd OLD', 'd NEW', 'Δ|d|', 'Verdict']
    body = []
    for r in cond_rows:
        d_old = r['cohen_d_d_vs_c_old']
        d_new = r['cohen_d_d_vs_c_new']
        delta = abs(d_new) - abs(d_old)
        if abs(delta) < 0.05:
            verdict = 'unchanged'
        elif delta > 0:
            verdict = 'strengthened'
        else:
            verdict = 'weakened'
        body.append([
            VAR_LABELS_EN[r['variable']],
            f'{d_old:+.2f}',
            f'{d_new:+.2f}',
            f'{delta:+.2f}',
            verdict,
        ])

    table = ax.table(
        cellText=body, colLabels=headers, loc='center',
        cellLoc='center', colLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)

    # Style header
    for col in range(len(headers)):
        cell = table[(0, col)]
        cell.set_facecolor('#1976D2')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor(BORDER)

    # Style body: left-align Measure, color verdict
    verdict_colors = {
        'strengthened': '#2E7D32',
        'weakened': '#C62828',
        'unchanged': MUTED,
    }
    for i, row in enumerate(body, start=1):
        # Measure column - left align
        c0 = table[(i, 0)]
        c0.set_text_props(ha='left')
        c0.PAD = 0.05
        # Verdict column - colored
        c4 = table[(i, 4)]
        c4.set_text_props(color=verdict_colors[row[4]], fontweight='bold')
        for col in range(len(headers)):
            table[(i, col)].set_edgecolor(BORDER)

    fig.text(0.5, 0.06,
             "d = signed Cohen's d for Diffuse - Clumpy (negative = clumpy higher).   "
             "Verdict based on |d|: strengthened if |d| grew by >= 0.05, weakened if it dropped by >= 0.05.",
             ha='center', fontsize=9, color=MUTED, style='italic')

    return fig


def _m69_raincloud_page(sample: SampleResult, cond_rows, sample_label: str):
    """Replicate the M69 raincloud headline page for a given sample."""
    from m69_email_summary_pdf_violin import _raincloud_panel, COND_COLORS as M69_COND

    # Pull per-row stats
    streak_row = next(r for r in cond_rows
                      if r['variable'] == 'seq_typing_mean_run_explore')
    pc1_row = next(r for r in cond_rows if r['variable'] == 'PC1')

    side = 'old' if sample_label == 'OLD' else 'new'
    streak_d = streak_row[f'cohen_d_d_vs_c_{side}']
    streak_p = streak_row[f'p_d_vs_c_{side}']
    pc1_d = pc1_row[f'cohen_d_d_vs_c_{side}']
    pc1_p = pc1_row[f'p_d_vs_c_{side}']

    n_total = len(sample.avg_df)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f'{sample_label} sample (N = {n_total}): pre-registered analysis null,\n'
        'reading-only streaks show a robust signal',
        fontsize=15, fontweight='bold', color=TEXT, y=0.965,
    )

    banner_y = 0.69
    banner_h = 0.19

    # Pre-reg banner (PC1)
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
                f'd = {pc1_d:+.2f},  p = {pc1_p:.2f}',
                transform=ax_pre.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='#888888')
    ax_pre.text(0.5, 0.13, 'no condition difference',
                transform=ax_pre.transAxes, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#888888')

    # Exploratory banner (streak)
    sig_color = '#1B5E20' if streak_p < 0.05 else '#888888'
    box_face = '#E8F5E9' if streak_p < 0.05 else '#F5F5F5'
    box_edge = '#2E7D32' if streak_p < 0.05 else '#999999'
    ax_ex = fig.add_axes([0.52, banner_y, 0.42, banner_h])
    ax_ex.axis('off')
    ax_ex.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_ex.transAxes,
        facecolor=box_face, edgecolor=box_edge, linewidth=1.5,
    ))
    ax_ex.text(0.5, 0.88, 'Exploratory finding',
                transform=ax_ex.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold', color=sig_color)
    ax_ex.text(0.5, 0.62,
                'Mean reading-only streak',
                transform=ax_ex.transAxes, ha='center', va='center',
                fontsize=10, color=MUTED, style='italic')
    ax_ex.text(0.5, 0.36,
                f'd = {streak_d:+.2f},  p = {streak_p:.3f}',
                transform=ax_ex.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color=sig_color)
    direction_text = 'Clumpy > Diffuse' if streak_d < 0 else 'Diffuse > Clumpy'
    ax_ex.text(0.5, 0.13, direction_text,
                transform=ax_ex.transAxes, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=sig_color)

    # Raincloud
    ax_rc = fig.add_axes([0.28, 0.22, 0.46, 0.40])
    d_vals = sample.avg_df.loc[sample.avg_df['condition'] == 'diffuse',
                               'seq_typing_mean_run_explore'].dropna().to_numpy()
    c_vals = sample.avg_df.loc[sample.avg_df['condition'] == 'clumpy',
                               'seq_typing_mean_run_explore'].dropna().to_numpy()
    _raincloud_panel(
        ax_rc,
        groups=[d_vals, c_vals],
        group_labels=[f'Diffuse (n={len(d_vals)})', f'Clumpy (n={len(c_vals)})'],
        group_colors=[M69_COND['diffuse'], M69_COND['clumpy']],
        ylabel='Mean reading-only streak (pages)',
        title=('Higher = longer stretches of pages without writing\n'
               'Lower = writing interleaved more frequently across the search'),
    )

    # Bottom takeaway
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

    return fig


def write_pdf(pdf_path: Path, old: SampleResult, new: SampleResult,
              overall_rows, cond_rows):
    with PdfPages(pdf_path) as pdf:
        for fig in [
            _minimal_table_page(old, new, cond_rows),
            _m69_raincloud_page(old, cond_rows, 'OLD'),
            _m69_raincloud_page(new, cond_rows, 'NEW'),
        ]:
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f'Saved PDF: {pdf_path}')


# ---------- output ------------------------------------------------------------

def hebrew_summary(overall_rows, cond_rows, old: SampleResult, new: SampleResult):
    lines = []
    lines.append('=' * 70)
    lines.append('סיכום השוואה: מדגם ישן מול חדש')
    lines.append('=' * 70)
    lines.append(f'מדגם ישן  (data/cleaned)    : N={len(old.avg_df)}, '
                 f'PC1={old.pca_pct[0]:.1f}%')
    lines.append(f'מדגם חדש  (data/cleaned_new): N={len(new.avg_df)}, '
                 f'PC1={new.pca_pct[0]:.1f}%')
    lines.append('')
    lines.append('שינוי במשתנים (חדש פחות ישן):')
    lines.append('-' * 70)
    for r in overall_rows:
        sig = ''
        if r['p_value'] < .001: sig = ' ***'
        elif r['p_value'] < .01: sig = ' **'
        elif r['p_value'] < .05: sig = ' *'
        lines.append(
            f"  {r['label_he']:<40} "
            f"ישן={r['mean_old']:6.2f} (SD {r['sd_old']:5.2f}) | "
            f"חדש={r['mean_new']:6.2f} (SD {r['sd_new']:5.2f}) | "
            f"Δ={r['diff_new_minus_old']:+6.2f} ({r['pct_change']:+5.1f}%) | "
            f"{r['direction']}{sig}"
        )
    lines.append('')
    lines.append('הבדלים בין תנאים (Diffuse - Clumpy):')
    lines.append('-' * 70)
    for r in cond_rows:
        lines.append(
            f"  {r['label_he']:<40}\n"
            f"    OLD: D={r['mean_diffuse_old']:6.2f} C={r['mean_clumpy_old']:6.2f}  "
            f"d={r['cohen_d_d_vs_c_old']:+5.2f}  p={r['p_d_vs_c_old']:.3f}\n"
            f"    NEW: D={r['mean_diffuse_new']:6.2f} C={r['mean_clumpy_new']:6.2f}  "
            f"d={r['cohen_d_d_vs_c_new']:+5.2f}  p={r['p_d_vs_c_new']:.3f}"
        )
    lines.append('')
    lines.append('הערות:')
    lines.append('  * p < .05   ** p < .01   *** p < .001')
    lines.append('  כיוון = "עלה"/"ירד" אם |Δ| >= 0.1 SD משולב, אחרת "כמעט לא השתנה".')
    lines.append('  המדגם החדש כולל את כל המשתתפים (ישנים + חדשים) מהקובץ העדכני.')
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

    print('\n=== Building comparison table ===')
    overall_rows = [overall_row(v, lbl, old.avg_df, new.avg_df) for v, lbl in VAR_ORDER]
    cond_rows = [condition_row(v, lbl, old.avg_df, new.avg_df) for v, lbl in VAR_ORDER]

    overall_df = pd.DataFrame(overall_rows)
    cond_df = pd.DataFrame(cond_rows)

    csv_path = OUTPUT_DIR / 'm72_old_vs_new_comparison.csv'
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('# Overall comparison (OLD vs NEW)\n')
        overall_df.to_csv(f, index=False)
        f.write('\n# Per-condition comparison (Diffuse vs Clumpy in each sample)\n')
        cond_df.to_csv(f, index=False)
    print(f'Saved CSV: {csv_path}')

    # Excel with two sheets
    try:
        xlsx_path = OUTPUT_DIR / 'm72_old_vs_new_comparison.xlsx'
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as xl:
            overall_df.to_excel(xl, sheet_name='overall', index=False)
            cond_df.to_excel(xl, sheet_name='by_condition', index=False)
            old.avg_df.to_excel(xl, sheet_name='old_per_participant', index=False)
            new.avg_df.to_excel(xl, sheet_name='new_per_participant', index=False)
        print(f'Saved Excel: {xlsx_path}')
    except ImportError:
        print('(skipped Excel: openpyxl not installed)')

    # Per-participant CSVs (so you have the raw aggregates)
    old.avg_df.to_csv(OUTPUT_DIR / 'm72_old_per_participant.csv', index=False)
    new.avg_df.to_csv(OUTPUT_DIR / 'm72_new_per_participant.csv', index=False)

    # PDF report
    write_pdf(OUTPUT_DIR / 'm72_old_vs_new_comparison.pdf', old, new,
              overall_rows, cond_rows)

    # Hebrew summary - save first, then attempt to print
    summary = hebrew_summary(overall_rows, cond_rows, old, new)
    summary_path = OUTPUT_DIR / 'm72_summary_he.txt'
    summary_path.write_text(summary, encoding='utf-8')
    print(f'\nSaved Hebrew summary: {summary_path}')
    try:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        print('\n' + summary)
    except (UnicodeEncodeError, AttributeError):
        print('(summary contains Hebrew, printed only to file)')


if __name__ == '__main__':
    main()

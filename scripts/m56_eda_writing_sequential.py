#!/usr/bin/env python3
"""
M56: EDA on continuous writing + sequential pattern measures
============================================================
Goal: find a measure that shows clumpy vs diffuse difference, since the pre-registered
binary switch counts (M52) showed null effect (d ~ 0.04 on PC1).

Two families of measures (per question, averaged per participant):

A. Continuous writing measures
   - typed_chars, pasted_chars, type_paste_ratio
   - n_typing_bursts, typing_duration_s
   - n_paste_events
   - prop_pages_with_writing
   - writing_concentration (Gini-like spread of chars across pages)
   - first_writing_page (1-indexed) and first_writing_time_s
   - final_answer_length

B. Sequential / run-length / entropy measures (on each of 3 binary signals)
   - mean_run_explore, mean_run_exploit
   - max_run
   - shannon_entropy of the binary sequence
   - lag1_autocorr

Same exclusions as M52 (>=3 pages, idle <50%, 3 SD).

Outputs:
  output/m56_eda_writing_sequential.csv  - per-participant measures
  output/m56_eda_writing_sequential.pdf  - report (light mode)
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

from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR
from m18_typing_binary import page_had_typing_or_paste

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

THRESHOLD_S = 60.0
IDLE_THRESHOLD_PCT = 50.0
MIN_PAGE_VISITS = 3
OUTLIER_SD = 3

# Light palette (project preference)
BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'
COND_COLORS = {'diffuse': '#1976D2', 'clumpy': '#C62828'}

MEANINGFUL_ACTIONS = ['article_open', 'search', 'link_click', 'back_navigation', 'paste']
SNAPSHOT_ACTIONS = ['answer_snapshot', 'answer_snapshot_cursor_leave']


# ----------------------------------------------------------------------
# Helpers from M52
# ----------------------------------------------------------------------

def load_lda_assignments():
    with open(DATA_DIR / 'cleaned' / 'topic_model.json', encoding='utf-8') as f:
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
        writing_snaps = snapshots[snapshots['AnswerLength'] != snapshots['prev_len']]
    else:
        writing_snaps = snapshots.iloc[:0]
    all_meaningful = pd.concat([meaningful, writing_snaps]).sort_values('Time')
    all_meaningful = all_meaningful[(all_meaningful['Time'] >= t0)
                                    & (all_meaningful['Time'] <= t_end)]
    if len(all_meaningful) == 0:
        return 100.0
    last = all_meaningful['Time'].iloc[-1]
    return ((t_end - last).total_seconds() / total_sec) * 100


# ----------------------------------------------------------------------
# Continuous writing measures
# ----------------------------------------------------------------------

def per_page_writing_stats(trial):
    """Return list of per-page dicts with typed_chars, pasted_chars, has_writing."""
    events = trial['events']
    t0 = trial['t0']
    pvs = trial['page_visits']

    # Snapshot stream (sorted) with seconds-from-start
    snaps = events[events['Action'].isin(SNAPSHOT_ACTIONS)].sort_values('Time').copy()
    snaps['t_sec'] = (snaps['Time'] - t0).dt.total_seconds()
    snap_t = snaps['t_sec'].to_numpy()
    snap_len = snaps['AnswerLength'].astype(float).fillna(0).to_numpy()

    # Pastes
    pastes = events[events['Action'] == 'paste'].sort_values('Time').copy()
    pastes['t_sec'] = (pastes['Time'] - t0).dt.total_seconds()
    paste_t = pastes['t_sec'].to_numpy()
    paste_len = pastes['PastedTextLength'].astype(float).fillna(0).to_numpy()

    out = []
    for pv in pvs:
        s, e = pv['start'], pv['end']

        # Snapshot deltas inside [s, e): chars added on this page (typed + pasted)
        idx_in = np.where((snap_t >= s) & (snap_t < e))[0]
        added_total = 0.0
        if len(idx_in) > 0:
            # Use last snapshot before s as baseline if exists, else 0
            prev_idx = np.where(snap_t < s)[0]
            baseline = snap_len[prev_idx[-1]] if len(prev_idx) else 0.0
            seq = np.concatenate([[baseline], snap_len[idx_in]])
            deltas = np.diff(seq)
            added_total = deltas[deltas > 0].sum()

        # Pastes inside [s, e)
        p_in = np.where((paste_t >= s) & (paste_t < e))[0]
        pasted_chars = paste_len[p_in].sum()
        n_pastes = int(len(p_in))

        # Typed = added minus pasted (clamp at 0; can be neg from edits/autocorrect)
        typed_chars = max(0.0, added_total - pasted_chars)

        out.append({
            'typed_chars': typed_chars,
            'pasted_chars': pasted_chars,
            'n_pastes': n_pastes,
            'has_writing': (typed_chars > 0) or (pasted_chars > 0),
            'duration': pv['duration'],
        })
    return out


def gini(x):
    """Gini concentration on non-negative array; 0 = uniform, 1 = one bucket."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def trial_writing_features(trial):
    pvs = trial['page_visits']
    n_pages = len(pvs)
    pp = per_page_writing_stats(trial)

    typed_per_page = np.array([p['typed_chars'] for p in pp])
    pasted_per_page = np.array([p['pasted_chars'] for p in pp])
    written_per_page = typed_per_page + pasted_per_page
    has_w = np.array([p['has_writing'] for p in pp], dtype=bool)

    typed_total = float(typed_per_page.sum())
    pasted_total = float(pasted_per_page.sum())
    n_pastes = sum(p['n_pastes'] for p in pp)

    # Type/paste ratio - safe log-like (avoid div-by-zero)
    if pasted_total + typed_total > 0:
        typed_share = typed_total / (typed_total + pasted_total)
    else:
        typed_share = np.nan

    # Typing bursts and duration from helpers'-style intervals
    bursts = trial['typing_intervals']
    n_bursts = len(bursts)
    burst_dur = sum(b - a for a, b in bursts)

    # Concentration of writing across pages (Gini)
    write_gini = gini(written_per_page) if has_w.any() else np.nan

    # Latency to first writing
    first_pos = next((i for i, w in enumerate(has_w) if w), None)
    if first_pos is not None:
        first_writing_page = first_pos + 1  # 1-indexed
        first_writing_time_s = pvs[first_pos]['start']
    else:
        first_writing_page = np.nan
        first_writing_time_s = np.nan

    # Final answer length
    end_rows = trial['events'][trial['events']['Action'] == 'task_end']
    if len(end_rows) > 0:
        fa = end_rows.iloc[0].get('FinalAnswerLength', np.nan)
        final_len = float(fa) if pd.notna(fa) else np.nan
    else:
        final_len = np.nan

    return {
        'typed_chars': typed_total,
        'pasted_chars': pasted_total,
        'typed_share': typed_share,
        'n_typing_bursts': n_bursts,
        'typing_duration_s': burst_dur,
        'n_paste_events': n_pastes,
        'prop_pages_with_writing': has_w.mean() if n_pages else np.nan,
        'writing_gini': write_gini,
        'first_writing_page': first_writing_page,
        'first_writing_time_s': first_writing_time_s,
        'final_answer_length': final_len,
    }


# ----------------------------------------------------------------------
# Sequential / run-length / entropy measures
# ----------------------------------------------------------------------

def runs(labels):
    """Return list of (label, run_length) tuples."""
    if len(labels) == 0:
        return []
    out = [(labels[0], 1)]
    for x in labels[1:]:
        if x == out[-1][0]:
            out[-1] = (x, out[-1][1] + 1)
        else:
            out.append((x, 1))
    return out


def shannon_entropy_binary(labels):
    """Entropy of binary sequence in bits (0..1)."""
    if len(labels) == 0:
        return np.nan
    arr = np.asarray(labels)
    p1 = (arr == arr[0]).mean()
    p2 = 1 - p1
    if p1 in (0, 1):
        return 0.0
    return -(p1 * np.log2(p1) + p2 * np.log2(p2))


def lag1_autocorr_binary(labels):
    """Lag-1 autocorrelation of binary 0/1 sequence; nan if no variance."""
    if len(labels) < 3:
        return np.nan
    arr = np.array([1 if x else 0 for x in labels], dtype=float)
    if arr.std() == 0:
        return np.nan
    a, b = arr[:-1], arr[1:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sequential_features(labels, exploit_value, explore_value, prefix):
    """Compute sequential features given a binary label list."""
    rs = runs(labels)
    explore_runs = [n for v, n in rs if v == explore_value]
    exploit_runs = [n for v, n in rs if v == exploit_value]
    return {
        f'{prefix}_mean_run_explore': np.mean(explore_runs) if explore_runs else np.nan,
        f'{prefix}_mean_run_exploit': np.mean(exploit_runs) if exploit_runs else np.nan,
        f'{prefix}_max_run': max(n for _, n in rs) if rs else np.nan,
        f'{prefix}_entropy': shannon_entropy_binary(labels),
        f'{prefix}_lag1_autocorr': lag1_autocorr_binary(labels),
    }


# ----------------------------------------------------------------------
# Build per-question dataframe
# ----------------------------------------------------------------------

def build_question_df(trials):
    lda = load_lda_assignments()
    rows = []
    for tr in trials:
        if tr['domain'] == 'practice':
            continue
        pvs = tr['page_visits']
        n_pages = len(pvs)
        t0_abs = tr['t0']
        t_end_abs = t0_abs + pd.Timedelta(seconds=tr['duration'])
        idle_pct = compute_idle_pct(tr['events'], t0_abs, t_end_abs)

        excluded_pages = n_pages < MIN_PAGE_VISITS
        excluded_idle = idle_pct >= IDLE_THRESHOLD_PCT

        # Binary labels
        time_lbl = ['exploit' if pv['duration'] > THRESHOLD_S else 'explore' for pv in pvs]
        topic_lbl_raw = [lda.get(pv['title'], -1) for pv in pvs]
        # For topic: "exploit" = topic same as previous. Build a binary "stayed" sequence
        # length n_pages (first page treated as "stayed" by convention; we use it for run/entropy).
        topic_lbl = ['exploit']
        for i in range(1, len(topic_lbl_raw)):
            topic_lbl.append('exploit' if topic_lbl_raw[i] == topic_lbl_raw[i - 1] else 'explore')
        typing_lbl = [page_had_typing_or_paste(pv, tr['typing_intervals'], tr['paste_times'])
                      for pv in pvs]
        # Coerce typing_lbl to 'exploit'/'explore' strings to match format
        typing_lbl = ['exploit' if v else 'explore' for v in typing_lbl]

        # Sequential features per signal (NaN if too short)
        if n_pages >= 2:
            seq_t = sequential_features(time_lbl, 'exploit', 'explore', 'seq_time')
            seq_topic = sequential_features(topic_lbl, 'exploit', 'explore', 'seq_topic')
            seq_typ = sequential_features(typing_lbl, 'exploit', 'explore', 'seq_typing')
        else:
            seq_t = {f'seq_time_{k}': np.nan for k in
                     ['mean_run_explore', 'mean_run_exploit', 'max_run', 'entropy', 'lag1_autocorr']}
            seq_topic = {f'seq_topic_{k}': np.nan for k in
                         ['mean_run_explore', 'mean_run_exploit', 'max_run', 'entropy', 'lag1_autocorr']}
            seq_typ = {f'seq_typing_{k}': np.nan for k in
                       ['mean_run_explore', 'mean_run_exploit', 'max_run', 'entropy', 'lag1_autocorr']}

        wf = trial_writing_features(tr)

        row = {
            'participant_id': tr['pid'],
            'condition': tr['condition'],
            'domain': tr['domain'],
            'n_pages': n_pages,
            'idle_pct': idle_pct,
            'excluded_pages': excluded_pages,
            'excluded_idle': excluded_idle,
            **wf,
            **seq_t, **seq_topic, **seq_typ,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Apply exclusions and average per participant
# ----------------------------------------------------------------------

# Measures we'll test (will be set after we know columns)
def get_measure_cols(df):
    skip = {'participant_id', 'condition', 'domain', 'n_pages',
            'idle_pct', 'excluded_pages', 'excluded_idle'}
    return [c for c in df.columns if c not in skip]


def apply_exclusions_and_average(question_df):
    excluded_mask = question_df['excluded_pages'] | question_df['excluded_idle']
    clean_q = question_df[~excluded_mask].copy()
    excl_pages = int(question_df['excluded_pages'].sum())
    excl_idle = int(question_df['excluded_idle'].sum())
    print(f'  Questions excluded (<{MIN_PAGE_VISITS} pages): {excl_pages}')
    print(f'  Questions excluded (idle>={IDLE_THRESHOLD_PCT:.0f}%): {excl_idle}')
    print(f'  Questions remaining: {len(clean_q)} / {len(question_df)}')

    measure_cols = get_measure_cols(clean_q)
    avg = (clean_q.groupby(['participant_id', 'condition'], as_index=False)
                  [measure_cols].mean())

    # 3 SD outlier removal: applied on the three core M52 counts NOT here, since
    # those measures aren't in this script. We apply 3 SD on PC1-equivalent measures:
    # the original three (typed_chars, pasted_chars, prop_pages_with_writing) as a
    # conservative proxy for stability. To stay aligned with M52, we instead remove
    # outliers on the three M52-style counts we don't compute - so skip this step
    # and document. The original 3 outliers in M52 (P26, P79, P134) we filter manually.
    M52_OUTLIER_PIDS = {26, 79, 134}
    n_before = len(avg)
    avg = avg[~avg['participant_id'].isin(M52_OUTLIER_PIDS)].copy()
    print(f'  Removed M52 3-SD outliers ({sorted(M52_OUTLIER_PIDS)}): '
          f'{n_before - len(avg)} dropped')
    print(f'  Final N: {len(avg)} participants '
          f'({(avg["condition"]=="diffuse").sum()} diffuse, '
          f'{(avg["condition"]=="clumpy").sum()} clumpy)')
    return avg, measure_cols


# ----------------------------------------------------------------------
# Statistical tests
# ----------------------------------------------------------------------

def welch_df(a, b):
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    return num / denom if denom > 0 else (n1 + n2 - 2)


def cohens_d(d_vals, c_vals):
    """Cohen's d with sign = diffuse - clumpy (positive = diffuse higher)."""
    d_vals = np.asarray(d_vals, dtype=float)
    c_vals = np.asarray(c_vals, dtype=float)
    d_vals = d_vals[~np.isnan(d_vals)]
    c_vals = c_vals[~np.isnan(c_vals)]
    if len(d_vals) < 2 or len(c_vals) < 2:
        return np.nan
    pooled = np.sqrt(((len(d_vals) - 1) * d_vals.std(ddof=1) ** 2
                      + (len(c_vals) - 1) * c_vals.std(ddof=1) ** 2)
                     / (len(d_vals) + len(c_vals) - 2))
    if pooled == 0:
        return 0.0
    return (d_vals.mean() - c_vals.mean()) / pooled


def test_all_measures(avg_df, measure_cols):
    rows = []
    for col in measure_cols:
        d_vals = avg_df.loc[avg_df['condition'] == 'diffuse', col].dropna().to_numpy()
        c_vals = avg_df.loc[avg_df['condition'] == 'clumpy', col].dropna().to_numpy()
        if len(d_vals) < 3 or len(c_vals) < 3:
            continue
        t_stat, p_val = sp_stats.ttest_ind(d_vals, c_vals, equal_var=False, nan_policy='omit')
        df_w = welch_df(d_vals, c_vals)
        d_eff = cohens_d(d_vals, c_vals)
        # Mann-Whitney U as non-parametric companion
        try:
            u_stat, mw_p = sp_stats.mannwhitneyu(d_vals, c_vals, alternative='two-sided')
        except ValueError:
            u_stat, mw_p = (np.nan, np.nan)
        rows.append({
            'measure': col,
            'n_d': len(d_vals), 'n_c': len(c_vals),
            'd_mean': d_vals.mean(), 'd_sd': d_vals.std(ddof=1),
            'c_mean': c_vals.mean(), 'c_sd': c_vals.std(ddof=1),
            't': t_stat, 'df': df_w, 'p_welch': p_val,
            'mw_p': mw_p,
            'cohen_d': d_eff,
            'abs_d': abs(d_eff),
        })
    return pd.DataFrame(rows).sort_values('abs_d', ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

MEASURE_GROUPS = {
    'Continuous writing': [
        'typed_chars', 'pasted_chars', 'typed_share', 'n_typing_bursts',
        'typing_duration_s', 'n_paste_events', 'prop_pages_with_writing',
        'writing_gini', 'first_writing_page', 'first_writing_time_s',
        'final_answer_length',
    ],
    'Sequential - time signal': [
        'seq_time_mean_run_explore', 'seq_time_mean_run_exploit',
        'seq_time_max_run', 'seq_time_entropy', 'seq_time_lag1_autocorr',
    ],
    'Sequential - topic signal': [
        'seq_topic_mean_run_explore', 'seq_topic_mean_run_exploit',
        'seq_topic_max_run', 'seq_topic_entropy', 'seq_topic_lag1_autocorr',
    ],
    'Sequential - typing signal': [
        'seq_typing_mean_run_explore', 'seq_typing_mean_run_exploit',
        'seq_typing_max_run', 'seq_typing_entropy', 'seq_typing_lag1_autocorr',
    ],
}


def plot_top_measures(avg_df, results_df, n_top=6):
    """Strip plots for the top n_top measures by |Cohen's d|."""
    top = results_df.head(n_top)
    n = len(top)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows))
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes).ravel()

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i]
        ax.set_facecolor(BG)
        col = row['measure']
        d_vals = avg_df.loc[avg_df['condition'] == 'diffuse', col].dropna().to_numpy()
        c_vals = avg_df.loc[avg_df['condition'] == 'clumpy', col].dropna().to_numpy()
        rng = np.random.default_rng(i + 1)
        ax.scatter(np.zeros(len(d_vals)) + rng.uniform(-0.08, 0.08, len(d_vals)),
                   d_vals, color=COND_COLORS['diffuse'], s=35, alpha=0.7,
                   edgecolors='#333', linewidth=0.4)
        ax.scatter(np.ones(len(c_vals)) + rng.uniform(-0.08, 0.08, len(c_vals)),
                   c_vals, color=COND_COLORS['clumpy'], s=35, alpha=0.7,
                   edgecolors='#333', linewidth=0.4)
        ax.hlines(d_vals.mean(), -0.3, 0.3, color=COND_COLORS['diffuse'], linewidth=2.2)
        ax.hlines(c_vals.mean(), 0.7, 1.3, color=COND_COLORS['clumpy'], linewidth=2.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Diffuse', 'Clumpy'], fontsize=9)
        ax.set_title(f'{col}\nd={row["cohen_d"]:+.2f}, p={row["p_welch"]:.3f}',
                     fontsize=9.5, color=TEXT, fontweight='bold')
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.4, axis='y', zorder=0)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(len(top), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    return fig


def make_summary_table_page(results_df):
    """One page with the full results table sorted by |d|."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M56: All measures sorted by |Cohen\'s d| (Diffuse - Clumpy)',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')

    lines = []
    lines.append(f'{"Measure":<32} {"d_mean":>9} {"c_mean":>9} {"t":>6} '
                 f'{"p":>7} {"MW p":>7} {"d":>7}  Sig')
    lines.append('-' * 98)
    for _, r in results_df.iterrows():
        sig = '***' if r['p_welch'] < .001 else '**' if r['p_welch'] < .01 \
              else '*' if r['p_welch'] < .05 else ''
        lines.append(
            f'{r["measure"]:<32} {r["d_mean"]:>9.2f} {r["c_mean"]:>9.2f} '
            f'{r["t"]:>6.2f} {r["p_welch"]:>7.3f} {r["mw_p"]:>7.3f} '
            f'{r["cohen_d"]:>+7.2f}  {sig}'
        )
    lines.append('')
    lines.append('  * p<.05  ** p<.01  *** p<.001')
    lines.append('  d sign: positive = Diffuse > Clumpy (= predicted direction for '
                 'switch-related measures)')
    text = '\n'.join(lines)
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=8,
            family='monospace', va='top', color=TEXT, linespacing=1.3)
    return fig


def make_group_pages(pdf, avg_df, results_df):
    """One page per measure group with strip plots for all measures in the group."""
    res_by_meas = {r['measure']: r for _, r in results_df.iterrows()}
    for group_name, cols in MEASURE_GROUPS.items():
        cols = [c for c in cols if c in res_by_meas]
        if not cols:
            continue
        n = len(cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.0 * nrows))
        fig.patch.set_facecolor(BG)
        fig.suptitle(group_name, fontsize=14, fontweight='bold', color=TEXT, y=0.99)
        axes = np.atleast_2d(axes).ravel()

        for i, col in enumerate(cols):
            ax = axes[i]
            ax.set_facecolor(BG)
            r = res_by_meas[col]
            d_vals = avg_df.loc[avg_df['condition'] == 'diffuse', col].dropna().to_numpy()
            c_vals = avg_df.loc[avg_df['condition'] == 'clumpy', col].dropna().to_numpy()
            rng = np.random.default_rng(hash(col) % 10000)
            ax.scatter(np.zeros(len(d_vals)) + rng.uniform(-0.08, 0.08, len(d_vals)),
                       d_vals, color=COND_COLORS['diffuse'], s=22, alpha=0.65,
                       edgecolors='#333', linewidth=0.3)
            ax.scatter(np.ones(len(c_vals)) + rng.uniform(-0.08, 0.08, len(c_vals)),
                       c_vals, color=COND_COLORS['clumpy'], s=22, alpha=0.65,
                       edgecolors='#333', linewidth=0.3)
            if len(d_vals):
                ax.hlines(d_vals.mean(), -0.3, 0.3, color=COND_COLORS['diffuse'], linewidth=2)
            if len(c_vals):
                ax.hlines(c_vals.mean(), 0.7, 1.3, color=COND_COLORS['clumpy'], linewidth=2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Diffuse', 'Clumpy'], fontsize=8)
            ax.set_title(f'{col}\nd={r["cohen_d"]:+.2f}  p={r["p_welch"]:.3f}',
                         fontsize=8.5, color=TEXT, fontweight='bold')
            ax.tick_params(colors=MUTED, labelsize=7)
            ax.grid(True, color=GRID, linewidth=0.3, axis='y', zorder=0)
            for sp in ax.spines.values():
                sp.set_color(BORDER)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        for j in range(len(cols), len(axes)):
            axes[j].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M56] EDA: Continuous writing + sequential pattern measures')
    print('=' * 60)

    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    print(f'\n--- Loaded {len(trials)} trials ---')

    print('\n--- Building per-question feature matrix ---')
    qdf = build_question_df(trials)
    print(f'  Total questions: {len(qdf)}')

    print('\n--- Applying exclusions ---')
    avg_df, measure_cols = apply_exclusions_and_average(qdf)

    print('\n--- Running Welch t-tests on each measure ---')
    results_df = test_all_measures(avg_df, measure_cols)
    print(f'  Computed tests for {len(results_df)} measures')

    # Print top 10
    print('\n--- Top 10 by |Cohen\'s d| ---')
    cols_show = ['measure', 'd_mean', 'c_mean', 't', 'p_welch', 'cohen_d']
    print(results_df[cols_show].head(10).to_string(index=False))

    # Save CSV
    csv_path = OUTPUT_DIR / 'm56_eda_writing_sequential.csv'
    avg_df.to_csv(csv_path, index=False)
    print(f'\nSaved per-participant measures: {csv_path}')

    results_csv = OUTPUT_DIR / 'm56_eda_writing_sequential_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f'Saved test results: {results_csv}')

    # PDF report
    pdf_path = OUTPUT_DIR / 'm56_eda_writing_sequential.pdf'
    with PdfPages(pdf_path) as pdf:
        # Page 1: top measures
        fig = plot_top_measures(avg_df, results_df, n_top=6)
        fig.suptitle('M56: Top 6 measures by |Cohen\'s d| (Diffuse vs Clumpy)',
                     fontsize=14, fontweight='bold', color=TEXT, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)
        # Page 2: summary table
        fig = make_summary_table_page(results_df)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)
        # Pages 3..: group pages
        make_group_pages(pdf, avg_df, results_df)
    print(f'Saved PDF report: {pdf_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()

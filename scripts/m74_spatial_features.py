#!/usr/bin/env python3
"""
M74: Spatial Search feature extraction + per-participant visualisation
======================================================================
Inputs:
  data/cleaned/spatial_search/Game.csv
  data/cleaned/spatial_search/KeyTable.csv
  output/m72_new_per_participant.csv          # 138 wiki IDs (analysis cohort)

Outputs:
  output/m74_per_trial_features.csv           # 5 rows per pid (15 features)
  output/m74_spatial_features.csv             # 1 row per pid (30 features = mean+sd)
  output/m74_spatial_maps.pdf                 # 1 page per pid

Cohort filtering (cumulative):
  1. inner-join with the 138 wiki-cohort IDs
  2. keep only participants with EXACTLY 5 'Real' trials
  -> N=132 (72 Clumpy + 60 Diffuse)

Per-trial features (15) -- see design doc for definitions:
  Phase:    explore_dur_mean, explore_dur_median, exploit_dur_mean,
            exploit_dur_median, pct_time_exploit, n_transitions
  Resource: total_collected, time_to_first_resource, inter_resource_mean,
            collection_rate
  Spatial:  path_length, coverage_entropy, heading_variability
  Hart:     patch_leaving_distance, levy_alpha
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'output'

GAME_CSV = DATA_DIR / 'cleaned' / 'spatial_search' / 'Game.csv'
WIKI_COHORT_CSV = OUTPUT_DIR / 'm72_new_per_participant.csv'

PER_TRIAL_OUT = OUTPUT_DIR / 'm74_per_trial_features.csv'
PER_PID_OUT = OUTPUT_DIR / 'm74_spatial_features.csv'
PDF_OUT = OUTPUT_DIR / 'm74_spatial_maps.pdf'

EXPECTED_TRIALS_PER_PID = 5
COORD_SCALE = 3.0          # ResourceX/Y are in 1/3 world units
GRID_BINS = 20             # for coverage_entropy
LEVY_LMIN = 1.0            # min step length for power-law fit

EXPLORE_COLOR = '#1976D2'
EXPLOIT_COLOR = '#C62828'
UNKNOWN_COLOR = '#9E9E9E'   # for trials with zero Enter_*_mode events
RESOURCE_COLOR = '#2E7D32'
START_COLOR = '#1a1a1a'
END_COLOR = '#888888'
BG = '#FFFFFF'

PHASE_FEATURES = [
    'explore_dur_mean', 'explore_dur_median',
    'exploit_dur_mean', 'exploit_dur_median',
    'pct_time_exploit', 'n_transitions',
]
RESOURCE_FEATURES = [
    'total_collected', 'time_to_first_resource',
    'inter_resource_mean', 'collection_rate',
]
SPATIAL_FEATURES = ['path_length', 'coverage_entropy', 'heading_variability']
HART_FEATURES = ['patch_leaving_distance', 'levy_alpha']
ALL_FEATURES = PHASE_FEATURES + RESOURCE_FEATURES + SPATIAL_FEATURES + HART_FEATURES


# ---------- time parsing -----------------------------------------------------

def parse_time_to_seconds(s):
    """Parse 'HH:MM:SS[.mmm]' to seconds-since-midnight. Returns NaN on bad input."""
    if not isinstance(s, str):
        return np.nan
    parts = s.split(':')
    if len(parts) != 3:
        return np.nan
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, TypeError):
        return np.nan


def trial_window(events):
    """Return (t0, t_end) seconds for a trial. Falls back to first event when
    Round_start is missing. Handles same-trial midnight rollover defensively."""
    end_rows = events[events['Action'] == 'Round_end']
    if len(end_rows) == 0:
        return np.nan, np.nan
    t_end = end_rows.iloc[0]['t_sec']
    start_rows = events[events['Action'] == 'Round_start']
    if len(start_rows) > 0:
        t0 = start_rows.iloc[0]['t_sec']
    else:
        t0 = events['t_sec'].dropna().iloc[0] if events['t_sec'].notna().any() else np.nan
    if pd.notna(t0) and pd.notna(t_end) and t_end < t0:
        t_end += 24 * 3600
    return t0, t_end


# ---------- phase extraction -------------------------------------------------

def extract_phases(events, t0, t_end):
    """Walk Enter_*_mode events and produce (mode, duration_sec) phases.
    Default mode before the first Enter event = 'explore' (per spec).
    n_transitions counts mode-changes only (deduplicated)."""
    enters = events[events['Action'].isin(['Enter_explore_mode', 'Enter_exploit_mode'])]
    enters = enters.sort_values('t_sec')

    phases = []
    cur_mode = 'explore'
    cur_start = t0
    n_transitions = 0

    for _, row in enters.iterrows():
        new_mode = 'explore' if row['Action'] == 'Enter_explore_mode' else 'exploit'
        if new_mode == cur_mode:
            continue
        phases.append((cur_mode, row['t_sec'] - cur_start))
        cur_mode = new_mode
        cur_start = row['t_sec']
        n_transitions += 1

    phases.append((cur_mode, t_end - cur_start))
    phases = [(m, d) for m, d in phases if pd.notna(d) and d > 0]
    return phases, n_transitions


def phase_metrics(events, t0, t_end):
    phases, n_trans = extract_phases(events, t0, t_end)
    out = {f: np.nan for f in PHASE_FEATURES}
    out['n_transitions'] = n_trans

    if n_trans == 0:
        # spec rule: don't compute the rest when no transitions
        return out

    explore_durs = [d for m, d in phases if m == 'explore']
    exploit_durs = [d for m, d in phases if m == 'exploit']
    total_dur = t_end - t0

    if explore_durs:
        out['explore_dur_mean'] = float(np.mean(explore_durs))
        out['explore_dur_median'] = float(np.median(explore_durs))
    if exploit_durs:
        out['exploit_dur_mean'] = float(np.mean(exploit_durs))
        out['exploit_dur_median'] = float(np.median(exploit_durs))
    if total_dur > 0:
        out['pct_time_exploit'] = 100.0 * sum(exploit_durs) / total_dur
    return out


# ---------- resource metrics -------------------------------------------------

def resource_metrics(events, t0, t_end):
    out = {f: np.nan for f in RESOURCE_FEATURES}

    end_rows = events[events['Action'] == 'Round_end']
    if len(end_rows) > 0:
        out['total_collected'] = float(end_rows.iloc[0].get('TotalCollected', np.nan))

    found = events[events['Action'] == 'Resource_found'].sort_values('t_sec')
    times = found['t_sec'].values
    if len(times) >= 1 and pd.notna(t0):
        out['time_to_first_resource'] = float(times[0] - t0)
    if len(times) >= 2:
        gaps = np.diff(times)
        out['inter_resource_mean'] = float(np.mean(gaps))

    duration = t_end - t0
    if pd.notna(out['total_collected']) and duration and duration > 0:
        out['collection_rate'] = float(out['total_collected'] / duration)
    return out


# ---------- spatial metrics --------------------------------------------------

def _wrap_deg(d):
    """Wrap signed degree differences into [-180, 180)."""
    return ((d + 180.0) % 360.0) - 180.0


def spatial_metrics(events, x_range, y_range):
    out = {f: np.nan for f in SPATIAL_FEATURES}
    samples = events[events['Action'] == 'Movement_sample'].sort_values('t_sec')
    xs = samples['X'].values.astype(float)
    ys = samples['Y'].values.astype(float)
    if len(xs) < 2:
        return out

    dx = np.diff(xs)
    dy = np.diff(ys)
    out['path_length'] = float(np.sum(np.hypot(dx, dy)))

    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    if x_hi > x_lo and y_hi > y_lo:
        x_bins = np.linspace(x_lo, x_hi, GRID_BINS + 1)
        y_bins = np.linspace(y_lo, y_hi, GRID_BINS + 1)
        counts, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
        total = counts.sum()
        if total > 0:
            p = counts.flatten() / total
            p = p[p > 0]
            out['coverage_entropy'] = float(-np.sum(p * np.log2(p)))

    headings = samples['Heading'].dropna().values.astype(float)
    if len(headings) >= 3:
        diffs = _wrap_deg(np.diff(headings))
        out['heading_variability'] = float(np.std(diffs, ddof=1))
    return out


# ---------- Hart-inspired metrics --------------------------------------------

def hart_metrics(events, t0, t_end):
    out = {f: np.nan for f in HART_FEATURES}

    # patch_leaving_distance: at every exploit->explore switch, distance from
    # the last Resource_found inside that exploit phase to the player position
    # at the switch.
    enters = events[events['Action'].isin(['Enter_explore_mode', 'Enter_exploit_mode'])]
    enters = enters.sort_values('t_sec')

    cur_mode = 'explore'
    last_exploit_start = None
    distances = []
    for _, row in enters.iterrows():
        new_mode = 'explore' if row['Action'] == 'Enter_explore_mode' else 'exploit'
        if new_mode == cur_mode:
            continue
        if cur_mode == 'exploit' and new_mode == 'explore' and last_exploit_start is not None:
            window = events[(events['t_sec'] >= last_exploit_start) &
                            (events['t_sec'] <= row['t_sec'])]
            res = window[window['Action'] == 'Resource_found']
            samples_before = events[(events['Action'] == 'Movement_sample') &
                                    (events['t_sec'] <= row['t_sec'])]
            if len(res) > 0 and len(samples_before) > 0:
                last_res = res.iloc[-1]
                rx = float(last_res['ResourceX']) * COORD_SCALE
                ry = float(last_res['ResourceY']) * COORD_SCALE
                pos = samples_before.iloc[-1]
                px, py = float(pos['X']), float(pos['Y'])
                distances.append(np.hypot(px - rx, py - ry))
        if new_mode == 'exploit':
            last_exploit_start = row['t_sec']
        cur_mode = new_mode
    if distances:
        out['patch_leaving_distance'] = float(np.mean(distances))

    # levy_alpha: Hill MLE on step lengths between consecutive Resource_found
    # events, in world coordinates, with a fixed L_min.
    found = events[events['Action'] == 'Resource_found'].sort_values('t_sec')
    if len(found) >= 3:
        rx = found['ResourceX'].values.astype(float) * COORD_SCALE
        ry = found['ResourceY'].values.astype(float) * COORD_SCALE
        steps = np.hypot(np.diff(rx), np.diff(ry))
        steps = steps[steps >= LEVY_LMIN]
        if len(steps) >= 5:
            n = len(steps)
            out['levy_alpha'] = 1.0 + n / float(np.sum(np.log(steps / LEVY_LMIN)))
    return out


# ---------- per-trial pipeline -----------------------------------------------

def features_for_trial(events, x_range, y_range):
    t0, t_end = trial_window(events)
    if pd.isna(t0) or pd.isna(t_end) or t_end <= t0:
        return {f: np.nan for f in ALL_FEATURES}
    feats = {}
    feats.update(phase_metrics(events, t0, t_end))
    feats.update(resource_metrics(events, t0, t_end))
    feats.update(spatial_metrics(events, x_range, y_range))
    feats.update(hart_metrics(events, t0, t_end))
    return feats


# ---------- loading & cohort filtering ---------------------------------------

def load_and_filter(game_csv, cohort_csv):
    print(f'Loading {game_csv}...')
    df = pd.read_csv(game_csv, low_memory=False)
    print(f'  raw: {len(df):,} rows, {df["ID"].nunique()} participants')

    cohort = pd.read_csv(cohort_csv)
    cohort_ids = set(cohort['participant_id'].astype(int).tolist())
    df = df[df['ID'].isin(cohort_ids)].copy()
    print(f'  after wiki-cohort filter (138 IDs): {df["ID"].nunique()} participants')

    df = df[df['GameCondition'].isin(['Clumpy', 'Diffuse'])].copy()
    df['t_sec'] = df['Time'].apply(parse_time_to_seconds)

    trial_counts = df.groupby('ID')['Trial'].nunique()
    keep_ids = trial_counts[trial_counts == EXPECTED_TRIALS_PER_PID].index.tolist()
    dropped = sorted(set(trial_counts.index) - set(keep_ids))
    if dropped:
        details = ', '.join(f'{pid}({trial_counts[pid]})' for pid in dropped)
        print(f'  dropping {len(dropped)} pids with !=5 trials: {details}')
    df = df[df['ID'].isin(keep_ids)].copy()
    print(f'  final cohort: {df["ID"].nunique()} participants, {len(df):,} rows')

    cond_counts = df.groupby('ID')['GameCondition'].first().value_counts()
    print(f'  condition split: {cond_counts.to_dict()}')
    return df


def global_xy_range(df):
    samples = df[df['Action'] == 'Movement_sample']
    return ((float(samples['X'].min()), float(samples['X'].max())),
            (float(samples['Y'].min()), float(samples['Y'].max())))


# ---------- aggregation ------------------------------------------------------

def aggregate_per_pid(per_trial):
    pid_rows = []
    for (pid, cond), grp in per_trial.groupby(['participant_id', 'condition']):
        row = {'participant_id': pid, 'condition': cond}
        for f in ALL_FEATURES:
            vals = grp[f].dropna().values
            row[f'{f}_mean'] = float(np.mean(vals)) if len(vals) > 0 else np.nan
            row[f'{f}_sd'] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
        pid_rows.append(row)
    return pd.DataFrame(pid_rows).sort_values('participant_id').reset_index(drop=True)


# ---------- visualisation ----------------------------------------------------

def plot_participant_page(pdf, pid, condition, trials_data, x_range, y_range, summary_rows):
    fig = plt.figure(figsize=(16, 6.5), facecolor=BG)
    gs = fig.add_gridspec(2, 5, height_ratios=[3.0, 1.2], hspace=0.35, wspace=0.18,
                          left=0.04, right=0.99, top=0.88, bottom=0.06)

    total = int(sum(r['n_collected'] for r in summary_rows if pd.notna(r['n_collected'])))
    fig.suptitle(f'Participant {pid}  |  Condition: {condition}  |  '
                 f'Total collected: {total}',
                 fontsize=14, color='#1a1a1a', y=0.96)

    for col, td in enumerate(trials_data):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
        events = td['events']
        samples = events[events['Action'] == 'Movement_sample'].sort_values('t_sec')
        xs = samples['X'].values.astype(float)
        ys = samples['Y'].values.astype(float)
        modes = samples['_mode'].values

        # path: draw segment-by-segment in the active mode's colour
        if len(xs) >= 2:
            for i in range(len(xs) - 1):
                m = modes[i]
                if m == 'explore':
                    c = EXPLORE_COLOR
                elif m == 'exploit':
                    c = EXPLOIT_COLOR
                else:
                    c = UNKNOWN_COLOR
                ax.plot(xs[i:i+2], ys[i:i+2], color=c, linewidth=0.7, alpha=0.85)

        res = events[events['Action'] == 'Resource_found']
        if len(res) > 0:
            rx = res['ResourceX'].values.astype(float) * COORD_SCALE
            ry = res['ResourceY'].values.astype(float) * COORD_SCALE
            ax.scatter(rx, ry, marker='*', s=22, color=RESOURCE_COLOR,
                       edgecolors='none', zorder=3)

        if len(xs) > 0:
            ax.scatter([xs[0]], [ys[0]], marker='o', s=42, color=START_COLOR, zorder=4)
            ax.scatter([xs[-1]], [ys[-1]], marker='D', s=30, color=END_COLOR, zorder=4)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.invert_yaxis()  # screen-style: Y grows downward
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        nc = summary_rows[col]['n_collected']
        n_collected_str = str(int(nc)) if pd.notna(nc) else '-'
        no_modes = (len(modes) > 0 and all(m == 'unknown' for m in modes))
        suffix = '  |  no mode events' if no_modes else ''
        ax.set_title(f'Trial {td["trial"]}  |  {td["map_id"]}  |  '
                     f'N={n_collected_str}{suffix}',
                     fontsize=10, color='#333333')

    # summary table
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis('off')
    headers = ['Trial', 'MapID', 'Duration (s)', '% Exploit', 'N collected', 'Path length']
    cell_text = []
    for r in summary_rows:
        cell_text.append([
            f'{r["trial"]}',
            f'{r["map_id"]}',
            f'{r["duration"]:.1f}' if pd.notna(r['duration']) else '-',
            f'{r["pct_exploit"]:.1f}' if pd.notna(r['pct_exploit']) else '-',
            f'{int(r["n_collected"]) if pd.notna(r["n_collected"]) else "-"}',
            f'{r["path_length"]:.0f}' if pd.notna(r['path_length']) else '-',
        ])
    tbl = ax_tbl.table(cellText=cell_text, colLabels=headers, loc='center',
                      cellLoc='center', colLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ---------- main -------------------------------------------------------------

def annotate_mode(events):
    """Label each row's _mode based on the last Enter_*_mode event before it.
    Default starting mode = 'explore'. Trials with NO Enter_*_mode events at
    all are marked 'unknown' end-to-end (rendered grey in the PDF)."""
    events = events.sort_values('t_sec').reset_index(drop=True)
    has_enter = events['Action'].isin(['Enter_explore_mode', 'Enter_exploit_mode']).any()
    if not has_enter:
        events['_mode'] = 'unknown'
        return events
    modes = []
    cur = 'explore'
    for _, row in events.iterrows():
        if row['Action'] == 'Enter_explore_mode':
            cur = 'explore'
        elif row['Action'] == 'Enter_exploit_mode':
            cur = 'exploit'
        modes.append(cur)
    events['_mode'] = modes
    return events


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_and_filter(GAME_CSV, WIKI_COHORT_CSV)
    x_range, y_range = global_xy_range(df)
    print(f'  global X range: {x_range}, Y range: {y_range}')

    per_trial_rows = []
    pid_pages = defaultdict(list)
    pid_summary = defaultdict(list)

    for pid, pid_df in df.groupby('ID'):
        condition = pid_df['GameCondition'].iloc[0]
        for trial, trial_df in pid_df.groupby('Trial'):
            trial_df = annotate_mode(trial_df)
            map_id = trial_df['MapID'].dropna().iloc[0] if trial_df['MapID'].notna().any() else ''
            feats = features_for_trial(trial_df, x_range, y_range)
            row = {'participant_id': int(pid), 'condition': condition,
                   'trial': int(trial), 'map_id': map_id, **feats}
            per_trial_rows.append(row)
            pid_pages[int(pid)].append({'trial': int(trial), 'map_id': map_id,
                                        'events': trial_df})
            t0, t_end = trial_window(trial_df)
            duration = (t_end - t0) if (pd.notna(t0) and pd.notna(t_end)) else np.nan
            pid_summary[int(pid)].append({
                'trial': int(trial), 'map_id': map_id,
                'duration': duration,
                'pct_exploit': feats['pct_time_exploit'],
                'n_collected': feats['total_collected'],
                'path_length': feats['path_length'],
            })

    per_trial_df = pd.DataFrame(per_trial_rows).sort_values(['participant_id', 'trial'])
    per_trial_df.to_csv(PER_TRIAL_OUT, index=False, float_format='%.6g')
    print(f'wrote {PER_TRIAL_OUT.name}: {len(per_trial_df)} rows')

    pid_df_out = aggregate_per_pid(per_trial_df)
    pid_df_out.to_csv(PER_PID_OUT, index=False, float_format='%.6g')
    print(f'wrote {PER_PID_OUT.name}: {len(pid_df_out)} rows, '
          f'{len(pid_df_out.columns)} cols')

    print(f'rendering {PDF_OUT.name}...')
    pid_to_cond = dict(zip(pid_df_out['participant_id'], pid_df_out['condition']))
    with PdfPages(PDF_OUT) as pdf:
        for pid in sorted(pid_pages.keys()):
            trials_data = sorted(pid_pages[pid], key=lambda x: x['trial'])
            summary = sorted(pid_summary[pid], key=lambda x: x['trial'])
            plot_participant_page(pdf, pid, pid_to_cond[pid],
                                  trials_data, x_range, y_range, summary)
    print(f'wrote {PDF_OUT.name}')


if __name__ == '__main__':
    main()

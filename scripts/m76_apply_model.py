#!/usr/bin/env python3
"""
M76: Apply the M75e-calibrated mode classifier to all 132 participants.
=======================================================================
Replaces M74's Enter-events mode tagging with the learned classifier
(GBM + 17 behavioural features + 7-second median smoothing).

Inputs:
  data/cleaned/spatial_search/Game.csv
  data/manual labeling/m75_labels_*.json   (used to refit the GBM in-script)
  output/m72_new_per_participant.csv       (138 wiki cohort -> filter to 132)

Outputs:
  output/m76_per_trial_features.csv        660 rows
  output/m76_spatial_features.csv          132 rows × 30 features
  output/m76_spatial_maps.pdf              one page per participant
  docs/m76_model_explanation.html          model logic + accuracy report
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'output'
DOCS_DIR = SCRIPT_DIR.parent / 'docs'

GAME_CSV = DATA_DIR / 'cleaned' / 'spatial_search' / 'Game.csv'
LABEL_DIR = DATA_DIR / 'manual labeling'
WIKI_COHORT_CSV = OUTPUT_DIR / 'm72_new_per_participant.csv'

PER_TRIAL_OUT = OUTPUT_DIR / 'm76_per_trial_features.csv'
PER_PID_OUT = OUTPUT_DIR / 'm76_spatial_features.csv'
PDF_OUT = OUTPUT_DIR / 'm76_spatial_maps.pdf'
HTML_OUT = DOCS_DIR / 'm76_model_explanation.html'

EXPECTED_TRIALS_PER_PID = 5
COORD_SCALE = 3.0
WIN_1S = 10
WIN_2S = 20
WIN_3S = 30
EPS = 0.5
GRID_BINS = 20
LEVY_LMIN = 1.0
SMOOTH_WIN = 71
DECISION_THRESHOLD = 0.45

EXPLORE_COLOR = '#1976D2'
EXPLOIT_COLOR = '#C62828'
RESOURCE_COLOR = '#2E7D32'
START_COLOR = '#1a1a1a'
END_COLOR = '#888888'
BG = '#FFFFFF'

FEATURE_NAMES = [
    'angular_speed', 'sinuosity', 'inst_speed', 'speed_sd',
    'dist_to_nearest_resource', 'resource_density_60',
    'resource_density_30', 'resource_density_100',
    'time_since_last_resource', 'recent_collection_count_2s',
    'displacement_2s', 'path_length_2s', 'radius_of_gyration_2s',
    'heading_sd_2s', 'angular_speed_3s',
    'fpt_30', 'fpt_50',
]

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


# ---------- helpers ----------------------------------------------------------

def parse_time_to_seconds(s):
    if not isinstance(s, str):
        return np.nan
    p = s.split(':')
    if len(p) != 3:
        return np.nan
    try:
        return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    except (ValueError, TypeError):
        return np.nan


def wrap_deg(d):
    return ((d + 180.0) % 360.0) - 180.0


def rolling_mean(arr, w):
    return pd.Series(arr).rolling(w, center=True, min_periods=3).mean().values


def rolling_std(arr, w):
    return pd.Series(arr).rolling(w, center=True, min_periods=3).std().values


# ---------- features (matches M75e) ------------------------------------------

def compute_features(samples, rxy, resource_t):
    xs = samples['X'].values.astype(float)
    ys = samples['Y'].values.astype(float)
    hs = samples['Heading'].values.astype(float)
    ts = samples['t_sec'].values.astype(float)
    n = len(xs)
    feats = {}

    if n < 2:
        for name in FEATURE_NAMES:
            feats[name] = np.full(n, np.nan)
        return feats

    dh = np.abs(wrap_deg(np.diff(hs)))
    dt = np.maximum(np.diff(ts), 1e-3)
    inst_ang = np.concatenate([[0.0], dh / dt])
    feats['angular_speed'] = rolling_mean(inst_ang, WIN_1S)
    feats['angular_speed_3s'] = rolling_mean(inst_ang, WIN_3S)

    dx = np.diff(xs); dy = np.diff(ys)
    step_dist = np.hypot(dx, dy)
    inst_v = np.concatenate([[0.0], step_dist / dt])
    feats['inst_speed'] = rolling_mean(inst_v, WIN_1S)
    feats['speed_sd'] = rolling_std(inst_v, WIN_1S)

    feats['heading_sd_2s'] = rolling_std(
        np.concatenate([[0.0], wrap_deg(np.diff(hs))]), WIN_2S)

    sin_v = np.full(n, np.nan)
    disp_2s = np.full(n, np.nan)
    path_2s = np.full(n, np.nan)
    rog_2s = np.full(n, np.nan)
    half_1s = WIN_1S // 2
    half_2s = WIN_2S // 2
    for i in range(n):
        a = max(0, i - half_1s); b = min(n, i + half_1s + 1)
        if b - a >= 3:
            px = xs[a:b]; py = ys[a:b]
            path = float(np.sum(np.hypot(np.diff(px), np.diff(py))))
            disp = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
            sin_v[i] = 50.0 if disp < EPS else path / disp
        a = max(0, i - half_2s); b = min(n, i + half_2s + 1)
        if b - a >= 5:
            px = xs[a:b]; py = ys[a:b]
            disp_2s[i] = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
            path_2s[i] = float(np.sum(np.hypot(np.diff(px), np.diff(py))))
            cx = px.mean(); cy = py.mean()
            rog_2s[i] = float(np.sqrt(np.mean((px - cx) ** 2 + (py - cy) ** 2)))
    feats['sinuosity'] = sin_v
    feats['displacement_2s'] = disp_2s
    feats['path_length_2s'] = path_2s
    feats['radius_of_gyration_2s'] = rog_2s

    if len(rxy) > 0:
        dx_r = xs[:, None] - rxy[None, :, 0]
        dy_r = ys[:, None] - rxy[None, :, 1]
        d2 = dx_r ** 2 + dy_r ** 2
        feats['dist_to_nearest_resource'] = np.sqrt(d2.min(axis=1))
        feats['resource_density_30'] = (d2 <= 30 ** 2).sum(axis=1).astype(float)
        feats['resource_density_60'] = (d2 <= 60 ** 2).sum(axis=1).astype(float)
        feats['resource_density_100'] = (d2 <= 100 ** 2).sum(axis=1).astype(float)
    else:
        feats['dist_to_nearest_resource'] = np.full(n, 9999.0)
        feats['resource_density_30'] = np.zeros(n)
        feats['resource_density_60'] = np.zeros(n)
        feats['resource_density_100'] = np.zeros(n)

    if len(resource_t) > 0:
        time_since = np.full(n, 30.0)
        recent_count_2s = np.zeros(n)
        sorted_rt = np.sort(resource_t)
        for i in range(n):
            t = ts[i]
            past = sorted_rt[sorted_rt <= t]
            if len(past) > 0:
                time_since[i] = min(30.0, t - past[-1])
            recent_count_2s[i] = int(np.sum((sorted_rt >= t - 2.0) & (sorted_rt <= t)))
        feats['time_since_last_resource'] = time_since
        feats['recent_collection_count_2s'] = recent_count_2s
    else:
        feats['time_since_last_resource'] = np.full(n, 30.0)
        feats['recent_collection_count_2s'] = np.zeros(n)

    # First-passage times
    for R, name in [(30.0, 'fpt_30'), (50.0, 'fpt_50')]:
        fpt = np.zeros(n)
        R2 = R ** 2
        for i in range(n):
            t_back = ts[i]
            for j in range(i - 1, -1, -1):
                if (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2 > R2:
                    t_back = ts[j]; break
            else:
                t_back = ts[0]
            t_fwd = ts[i]
            for j in range(i + 1, n):
                if (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2 > R2:
                    t_fwd = ts[j]; break
            else:
                t_fwd = ts[-1]
            fpt[i] = max(0.0, t_fwd - t_back)
        feats[name] = fpt

    return feats


# ---------- model fitting (refits in-script) ---------------------------------

def make_gbm():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, random_state=0)


def load_label_files():
    files = sorted(LABEL_DIR.glob('m75_labels_*.json'))
    return [json.loads(f.read_text(encoding='utf-8')) for f in files]


def assemble_training_set(label_files, df):
    rows = []
    for d in label_files:
        pid, trial = d['pid'], d['trial']
        sub = df[(df['ID'] == pid) & (df['Trial'] == trial)].copy()
        sub['t_sec'] = sub['Time'].apply(parse_time_to_seconds)
        sub = sub.sort_values('t_sec')
        samples = sub[sub['Action'] == 'Movement_sample'].sort_values('t_sec').reset_index(drop=True)
        res = sub[sub['Action'] == 'Resource_found']
        rxy = []
        rt = []
        for _, r in res.iterrows():
            if pd.notna(r['ResourceX']) and pd.notna(r['ResourceY']):
                rxy.append([float(r['ResourceX']) * COORD_SCALE,
                            float(r['ResourceY']) * COORD_SCALE])
                rt.append(float(r['t_sec']))
        rxy = np.array(rxy) if rxy else np.zeros((0, 2))
        rt = np.array(rt) if rt else np.zeros(0)
        n = min(len(samples), len(d['labels']))
        feats = compute_features(samples.iloc[:n], rxy, rt)
        for i in range(n):
            row = {'pid': pid, 'trial': trial, 'label': d['labels'][i],
                   'trial_key': f'{pid}_{trial}'}
            for fname in FEATURE_NAMES:
                row[fname] = feats[fname][i]
            rows.append(row)
    return pd.DataFrame(rows)


def fit_classifier(label_files, df_labeled_pids):
    print('refitting GBM on labelled data...')
    train = assemble_training_set(label_files, df_labeled_pids)
    train = train[train['label'].isin(['explore', 'exploit'])].copy()
    finite = np.all(np.isfinite(train[FEATURE_NAMES].values.astype(float)), axis=1)
    train = train.loc[finite].reset_index(drop=True)
    X = train[FEATURE_NAMES].values.astype(float)
    y = (train['label'] == 'exploit').astype(int).values
    groups = train['trial_key'].values

    # quick LOTO sanity check (re-confirms ~89.9% from M75e)
    logo = LeaveOneGroupOut()
    pred = np.zeros(len(y), dtype=float)
    for tr, te in logo.split(X, y, groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = make_gbm()
        clf.fit(Xtr, y[tr])
        pred[te] = clf.predict_proba(Xte)[:, 1]
    smoothed = pd.Series(pred).rolling(SMOOTH_WIN, center=True, min_periods=1).median().values
    auc = roc_auc_score(y, smoothed)
    acc = accuracy_score(y, (smoothed > DECISION_THRESHOLD).astype(int))
    print(f'  LOTO sanity check: AUC={auc:.3f}, ACC@thr{DECISION_THRESHOLD}={acc:.3f}')

    final_scaler = StandardScaler().fit(X)
    final_clf = make_gbm()
    final_clf.fit(final_scaler.transform(X), y)
    return final_clf, final_scaler, train, {'loto_auc': auc, 'loto_acc': acc}


# ---------- cohort filtering -------------------------------------------------

def load_cohort(game_csv, wiki_csv):
    print(f'Loading {game_csv}...')
    df = pd.read_csv(game_csv, low_memory=False)
    cohort = pd.read_csv(wiki_csv)
    cohort_ids = set(cohort['participant_id'].astype(int).tolist())
    df = df[df['ID'].isin(cohort_ids)].copy()
    df = df[df['GameCondition'].isin(['Clumpy', 'Diffuse'])].copy()
    df['t_sec'] = df['Time'].apply(parse_time_to_seconds)
    trial_counts = df.groupby('ID')['Trial'].nunique()
    keep_ids = trial_counts[trial_counts == EXPECTED_TRIALS_PER_PID].index.tolist()
    df = df[df['ID'].isin(keep_ids)].copy()
    print(f'  cohort: {df["ID"].nunique()} pids, {len(df):,} rows')
    return df


def trial_resources(trial_df):
    res = trial_df[trial_df['Action'] == 'Resource_found']
    rxy = []
    rt = []
    for _, r in res.iterrows():
        if pd.notna(r['ResourceX']) and pd.notna(r['ResourceY']):
            rxy.append([float(r['ResourceX']) * COORD_SCALE,
                        float(r['ResourceY']) * COORD_SCALE])
            rt.append(float(r['t_sec']))
    rxy = np.array(rxy) if rxy else np.zeros((0, 2))
    rt = np.array(rt) if rt else np.zeros(0)
    return rxy, rt


def predict_modes_for_trial(samples, rxy, rt, clf, scaler):
    n = len(samples)
    if n == 0:
        return np.array([]), np.array([])
    feats_d = compute_features(samples, rxy, rt)
    F = np.column_stack([feats_d[name] for name in FEATURE_NAMES]).astype(float)
    finite = np.all(np.isfinite(F), axis=1)
    prob = np.full(n, 0.5)
    if finite.any():
        Fs = scaler.transform(F[finite])
        prob[finite] = clf.predict_proba(Fs)[:, 1]
    smoothed = pd.Series(prob).rolling(SMOOTH_WIN, center=True, min_periods=1).median().values
    modes = np.where(smoothed > DECISION_THRESHOLD, 'exploit', 'explore')
    return modes, smoothed


# ---------- per-trial feature extraction (M74-style, mode-aware) -------------

def trial_window(trial_df):
    end = trial_df[trial_df['Action'] == 'Round_end']
    if len(end) == 0:
        return np.nan, np.nan
    t_end = float(end.iloc[0]['t_sec'])
    start = trial_df[trial_df['Action'] == 'Round_start']
    if len(start) > 0:
        t0 = float(start.iloc[0]['t_sec'])
    else:
        t0 = float(trial_df['t_sec'].dropna().iloc[0])
    if pd.notna(t0) and pd.notna(t_end) and t_end < t0:
        t_end += 24 * 3600
    return t0, t_end


def phase_metrics_from_modes(mode_arr, sample_t):
    """Convert per-sample mode array into phase durations + transitions."""
    out = {f: np.nan for f in PHASE_FEATURES}
    if len(mode_arr) == 0:
        return out
    explore_durs = []
    exploit_durs = []
    n_transitions = 0
    cur = mode_arr[0]
    cur_start = sample_t[0]
    for i in range(1, len(mode_arr)):
        if mode_arr[i] != cur:
            dur = sample_t[i] - cur_start
            if dur > 0:
                (explore_durs if cur == 'explore' else exploit_durs).append(dur)
            cur = mode_arr[i]
            cur_start = sample_t[i]
            n_transitions += 1
    last_dur = sample_t[-1] - cur_start
    if last_dur > 0:
        (explore_durs if cur == 'explore' else exploit_durs).append(last_dur)

    out['n_transitions'] = n_transitions
    if explore_durs:
        out['explore_dur_mean'] = float(np.mean(explore_durs))
        out['explore_dur_median'] = float(np.median(explore_durs))
    if exploit_durs:
        out['exploit_dur_mean'] = float(np.mean(exploit_durs))
        out['exploit_dur_median'] = float(np.median(exploit_durs))
    total = sum(explore_durs) + sum(exploit_durs)
    if total > 0:
        out['pct_time_exploit'] = 100.0 * sum(exploit_durs) / total
    return out


def resource_metrics(trial_df, t0, t_end):
    out = {f: np.nan for f in RESOURCE_FEATURES}
    end = trial_df[trial_df['Action'] == 'Round_end']
    if len(end) > 0:
        out['total_collected'] = float(end.iloc[0].get('TotalCollected', np.nan))
    found = trial_df[trial_df['Action'] == 'Resource_found'].sort_values('t_sec')
    times = found['t_sec'].values
    if len(times) >= 1 and pd.notna(t0):
        out['time_to_first_resource'] = float(times[0] - t0)
    if len(times) >= 2:
        out['inter_resource_mean'] = float(np.mean(np.diff(times)))
    duration = t_end - t0
    if pd.notna(out['total_collected']) and duration and duration > 0:
        out['collection_rate'] = float(out['total_collected'] / duration)
    return out


def spatial_metrics(samples, x_range, y_range):
    out = {f: np.nan for f in SPATIAL_FEATURES}
    xs = samples['X'].values.astype(float)
    ys = samples['Y'].values.astype(float)
    if len(xs) < 2:
        return out
    out['path_length'] = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    x_lo, x_hi = x_range; y_lo, y_hi = y_range
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
        out['heading_variability'] = float(
            np.std(wrap_deg(np.diff(headings)), ddof=1))
    return out


def hart_metrics(samples, mode_arr, rxy):
    out = {f: np.nan for f in HART_FEATURES}
    n = len(samples)

    # patch_leaving_distance: at every exploit -> explore transition, the
    # distance from the last exploit-phase Resource_found to the player
    # position at the transition.
    if n >= 2 and len(mode_arr) == n:
        sample_xy = np.column_stack([samples['X'].values.astype(float),
                                      samples['Y'].values.astype(float)])
        sample_t = samples['t_sec'].values.astype(float)
        # find exploit phases (start, end) indices
        distances = []
        in_exploit = (mode_arr == 'exploit')
        i = 0
        while i < n:
            if in_exploit[i]:
                j = i
                while j + 1 < n and in_exploit[j + 1]:
                    j += 1
                # exploit phase from index i to j; transition out at j+1
                if j + 1 < n and not in_exploit[j + 1] and len(rxy) > 0:
                    t_exit = sample_t[j + 1]
                    t_enter = sample_t[i]
                    # last Resource_found in [t_enter, t_exit]?
                    # we don't have resource_t here directly; approximate:
                    # use the Resource_found nearest to player AT the exit
                    # in the exploit window -- via resource positions only.
                    # Compute distance from the player's exit position to the
                    # nearest resource that the player was within 60 px of
                    # during this exploit phase.
                    px_exit, py_exit = sample_xy[j + 1]
                    in_window_mask = np.full(len(rxy), False)
                    for k in range(i, j + 1):
                        d2 = (rxy[:, 0] - sample_xy[k, 0]) ** 2 + \
                             (rxy[:, 1] - sample_xy[k, 1]) ** 2
                        in_window_mask |= (d2 <= 60 ** 2)
                    if in_window_mask.any():
                        cand = rxy[in_window_mask]
                        d = np.hypot(cand[:, 0] - px_exit, cand[:, 1] - py_exit)
                        distances.append(float(d.min()))
                i = j + 1
            else:
                i += 1
        if distances:
            out['patch_leaving_distance'] = float(np.mean(distances))

    # levy_alpha: same as M74 -- Hill MLE on inter-Resource step lengths
    if len(rxy) >= 3:
        steps = np.hypot(np.diff(rxy[:, 0]), np.diff(rxy[:, 1]))
        steps = steps[steps >= LEVY_LMIN]
        if len(steps) >= 5:
            out['levy_alpha'] = 1.0 + len(steps) / float(np.sum(np.log(steps / LEVY_LMIN)))
    return out


def features_for_trial(trial_df, samples, mode_arr, rxy, x_range, y_range):
    feats = {}
    t0, t_end = trial_window(trial_df)
    if pd.isna(t0) or pd.isna(t_end) or t_end <= t0:
        return {f: np.nan for f in ALL_FEATURES}
    sample_t = samples['t_sec'].values.astype(float)
    feats.update(phase_metrics_from_modes(mode_arr, sample_t))
    feats.update(resource_metrics(trial_df, t0, t_end))
    feats.update(spatial_metrics(samples, x_range, y_range))
    feats.update(hart_metrics(samples, mode_arr, rxy))
    return feats


# ---------- aggregation ------------------------------------------------------

def aggregate_per_pid(per_trial):
    rows = []
    for (pid, cond), grp in per_trial.groupby(['participant_id', 'condition']):
        row = {'participant_id': pid, 'condition': cond}
        for f in ALL_FEATURES:
            vals = grp[f].dropna().values
            row[f'{f}_mean'] = float(np.mean(vals)) if len(vals) > 0 else np.nan
            row[f'{f}_sd'] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values('participant_id').reset_index(drop=True)


# ---------- visualisation ----------------------------------------------------

def plot_participant_page(pdf, pid, condition, trials_data, x_range, y_range, summary_rows):
    fig = plt.figure(figsize=(16, 6.5), facecolor=BG)
    gs = fig.add_gridspec(2, 5, height_ratios=[3.0, 1.2], hspace=0.35, wspace=0.18,
                          left=0.04, right=0.99, top=0.88, bottom=0.06)

    total = int(sum(r['n_collected'] for r in summary_rows if pd.notna(r['n_collected'])))
    fig.suptitle(f'Participant {pid}  |  Condition: {condition}  |  '
                 f'Total collected: {total}  (M76: GBM-classified modes)',
                 fontsize=14, color='#1a1a1a', y=0.96)

    for col, td in enumerate(trials_data):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
        samples = td['samples']
        modes = td['modes']
        xs = samples['X'].values.astype(float)
        ys = samples['Y'].values.astype(float)
        if len(xs) >= 2:
            for i in range(len(xs) - 1):
                c = EXPLOIT_COLOR if (i < len(modes) and modes[i] == 'exploit') else EXPLORE_COLOR
                ax.plot(xs[i:i+2], ys[i:i+2], color=c, linewidth=0.7, alpha=0.85)
        if len(td['rxy']) > 0:
            ax.scatter(td['rxy'][:, 0], td['rxy'][:, 1], marker='*', s=22,
                       color=RESOURCE_COLOR, edgecolors='none', zorder=3)
        if len(xs) > 0:
            ax.scatter([xs[0]], [ys[0]], marker='o', s=42, color=START_COLOR, zorder=4)
            ax.scatter([xs[-1]], [ys[-1]], marker='D', s=30, color=END_COLOR, zorder=4)
        ax.set_xlim(x_range); ax.set_ylim(y_range)
        ax.invert_yaxis(); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        nc = summary_rows[col]['n_collected']
        nc_str = str(int(nc)) if pd.notna(nc) else '-'
        ax.set_title(f'Trial {td["trial"]}  |  {td["map_id"]}  |  N={nc_str}',
                     fontsize=10, color='#333333')

    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis('off')
    headers = ['Trial', 'MapID', 'Duration (s)', '% Exploit', 'N collected', 'Path length']
    cell_text = []
    for r in summary_rows:
        cell_text.append([
            f'{r["trial"]}', f'{r["map_id"]}',
            f'{r["duration"]:.1f}' if pd.notna(r['duration']) else '-',
            f'{r["pct_exploit"]:.1f}' if pd.notna(r['pct_exploit']) else '-',
            f'{int(r["n_collected"]) if pd.notna(r["n_collected"]) else "-"}',
            f'{r["path_length"]:.0f}' if pd.notna(r['path_length']) else '-',
        ])
    tbl = ax_tbl.table(cellText=cell_text, colLabels=headers, loc='center',
                      cellLoc='center', colLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.4)
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ---------- HTML explanation -------------------------------------------------

def write_html_explanation(clf, training_n, loto_metrics):
    feat_imp = list(zip(FEATURE_NAMES, clf.feature_importances_))
    feat_imp.sort(key=lambda x: -x[1])
    rows = '\n'.join(
        f'<tr><td><code>{n}</code></td><td>{imp*100:.1f}%</td>'
        f'<td><div style="background:#1976D2;height:14px;'
        f'width:{int(imp*600)}px;"></div></td></tr>'
        for n, imp in feat_imp
    )
    DOCS_DIR.mkdir(exist_ok=True)
    html = f'''<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>M76 - Mode classifier explanation</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif;
         background: #FFFFFF; color: #1a1a1a; line-height: 1.6;
         max-width: 960px; margin: 32px auto; padding: 0 24px; }}
  h1 {{ color: #1a1a1a; border-bottom: 2px solid #1976D2; padding-bottom: 8px; }}
  h2 {{ color: #1976D2; margin-top: 36px; border-bottom: 1px solid #E0E0E0;
       padding-bottom: 4px; }}
  code, pre {{ direction: ltr; text-align: left; background: #F5F5F5;
              border: 1px solid #E0E0E0; border-radius: 4px;
              font-family: Consolas, monospace; }}
  code {{ padding: 2px 6px; }}
  pre {{ padding: 12px; overflow-x: auto; }}
  table {{ border-collapse: collapse; margin: 12px 0; width: 100%; direction: ltr; }}
  th, td {{ border: 1px solid #CCC; padding: 8px 12px; text-align: left; }}
  th {{ background: #F0F4F8; }}
  tr:nth-child(even) {{ background: #FAFAFA; }}
  .info {{ background: #E3F2FD; border-right: 4px solid #1976D2;
          padding: 12px 16px; margin: 12px 0; border-radius: 4px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>M76 - מסווג מצב Explore/Exploit (הסבר המודל)</h1>
<div class="meta">
  Script: <code>scripts/m76_apply_model.py</code> &middot;
  מודל מבוסס: M75e it24_gbm_fpt_smooth71 &middot;
  אומן על: {training_n} דגימות מ-6 trials מתויגים ידנית
</div>

<h2>1. הקונטקסט</h2>
<p>
  ב-M74 הוקצה מצב (explore/exploit) על סמך אירועי <code>Enter_*_mode</code>
  שהשחקן עצמו לחץ. מהשוואה לתיוג ידני התברר שזה תואם פחות מ-51% מהמקרים -
  כלומר ה-Enter events כמעט לא משקפים את ההתנהגות בפועל. M76 מחליף את זה
  במסווג שלמד מ-3,438 דגימות שתויגו ידנית ב-6 ניסויים מייצגים.
</p>

<h2>2. ארכיטקטורה</h2>
<ol>
  <li>
    <strong>חישוב 17 פיצ'רים לכל Movement_sample</strong> (כל ~0.1 שנייה):
    גיאומטריה של תנועה, צפיפות משאבים, וזמני first-passage.
  </li>
  <li>
    <strong>Gradient Boosting (200 עצים, עומק 3, learning_rate=0.05)</strong> -
    ensemble של עצי החלטה רדודים שלומדים אינטראקציות לא-ליניאריות בין
    הפיצ'רים. כל עץ מתקן את שגיאות הקודמים.
  </li>
  <li>
    <strong>Median smoothing</strong> על פני {SMOOTH_WIN} דגימות (~7 שניות) -
    מסיר flickering בין מצבים, ומתאים לסגנון התיוג של בלוקים גדולים.
  </li>
  <li>
    <strong>Decision threshold = {DECISION_THRESHOLD}</strong> (כויל אוטומטית).
    הסתברות חזויה מעל הסף = exploit.
  </li>
</ol>

<h2>3. הפיצ'רים (חשיבות יחסית במודל)</h2>
<table>
  <tr><th>Feature</th><th>Importance</th><th></th></tr>
  {rows}
</table>
<div class="info">
  <strong>תרגום למילים:</strong> אתה ב-exploit אם 7 השניות סביבך עברו בעיקר
  באזור עתיר משאבים, עם הרבה סיבוב או תזוזה מועטה. אתה ב-explore אם נסעת
  במהירות בקווים ישרים דרך אזור דליל. כ-67% מההחלטה מבוסס על שלושת הסיגנלים
  הדומיננטיים: צפיפות משאבים, קצב סיבוב, ורדיוס תזוזה.
</div>

<h2>4. ביצועים</h2>
<table>
  <tr><th>שיטה</th><th>Accuracy מול תיוג ידני</th></tr>
  <tr><td>Old Enter events (M74)</td><td>50.9%</td></tr>
  <tr><td>Single feature (resource_density &gt; 29)</td><td>77.5%</td></tr>
  <tr><td>Logistic regression (6 features)</td><td>74.6%</td></tr>
  <tr><td>GBM (15 features) + smoothing 1s</td><td>82.0%</td></tr>
  <tr><td><strong>M76: GBM (17 features) + smoothing 7s + tuned threshold</strong></td>
      <td><strong>{loto_metrics["loto_acc"]*100:.1f}%</strong> (LOTO CV, AUC={loto_metrics["loto_auc"]:.3f})</td></tr>
</table>

<h2>5. שלבי הצינור</h2>
<pre>refit GBM on labelled data
  -> 6 trials × ~580 samples = 3,438 labelled samples
  -> LOTO CV sanity: AUC={loto_metrics["loto_auc"]:.3f}, ACC={loto_metrics["loto_acc"]*100:.1f}%

apply to all 132 cohort participants × 5 trials = 660 trials:
  for each trial:
    compute 17 features per Movement_sample
    standardise (using saved scaler)
    GBM.predict_proba -> p(exploit)
    median-smooth p over {SMOOTH_WIN} samples
    threshold at {DECISION_THRESHOLD} -> binary mode

recompute M74-style per-trial features using new modes:
  - 6 phase metrics (explore/exploit durations, %time, transitions)
  - patch_leaving_distance (mode-aware)
  - 8 resource/spatial/Hart metrics (mode-independent, unchanged)
aggregate to mean+sd per pid -> 30 features × 132 rows
render PDF: 1 page per pid, 5 sub-plots, mode-coloured paths</pre>

<h2>6. קבצי פלט</h2>
<table>
  <tr><th>קובץ</th><th>תוכן</th></tr>
  <tr><td><code>output/m76_per_trial_features.csv</code></td>
      <td>660 שורות (132 pids × 5 trials), 15 פיצ'רים מבוססי המודל</td></tr>
  <tr><td><code>output/m76_spatial_features.csv</code></td>
      <td>132 שורות, 30 עמודות (mean + sd לכל פיצ'ר)</td></tr>
  <tr><td><code>output/m76_spatial_maps.pdf</code></td>
      <td>132 עמודים, מפות עם צביעת המסלולים לפי המודל החדש</td></tr>
</table>

<h2>7. הבדלים מ-M74</h2>
<ul>
  <li>צביעת המסלולים זהה (כחול/אדום) אבל מבוססת על המודל ולא על Enter events.</li>
  <li>אין יותר "trials עם 0 transitions" שצובעים אפור - לכל סבב יש סיווג.</li>
  <li>ה-30 פיצ'רים הסופיים זהים בשמותיהם ל-M74, אבל הערכים של 7 הפיצ'רים
      התלויים-מצב (6 phase + patch_leaving_distance) משתנים בעקבות הסיווג החדש.
      8 הפיצ'רים האחרים (resource/spatial/Hart non-mode) זהים.</li>
</ul>

</body>
</html>'''
    HTML_OUT.write_text(html, encoding='utf-8')
    print(f'wrote {HTML_OUT}')


# ---------- main -------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load Game.csv (full file - we'll filter labelled pids separately)
    df_full_raw = pd.read_csv(GAME_CSV, low_memory=False)
    df_full_raw = df_full_raw[df_full_raw['GameCondition'].isin(['Clumpy', 'Diffuse'])].copy()
    df_full_raw['t_sec'] = df_full_raw['Time'].apply(parse_time_to_seconds)

    # 2. Refit GBM on labelled trials
    label_files = load_label_files()
    labelled_pids = sorted({d['pid'] for d in label_files})
    df_labelled = df_full_raw[df_full_raw['ID'].isin(labelled_pids)].copy()
    clf, scaler, train_df, loto_metrics = fit_classifier(label_files, df_labelled)

    # 3. Filter to N=132 cohort
    df = load_cohort(GAME_CSV, WIKI_COHORT_CSV)
    samples_full = df[df['Action'] == 'Movement_sample']
    x_range = (float(samples_full['X'].min()), float(samples_full['X'].max()))
    y_range = (float(samples_full['Y'].min()), float(samples_full['Y'].max()))
    print(f'  global X range: {x_range}, Y range: {y_range}')

    # 4. Apply model + extract per-trial features
    per_trial_rows = []
    pid_pages = defaultdict(list)
    pid_summary = defaultdict(list)

    n_pids = df['ID'].nunique()
    print(f'\nApplying model to {n_pids} participants × {EXPECTED_TRIALS_PER_PID} trials...')
    for k, (pid, pid_df) in enumerate(df.groupby('ID'), 1):
        condition = pid_df['GameCondition'].iloc[0]
        for trial, trial_df in pid_df.groupby('Trial'):
            trial_df = trial_df.sort_values('t_sec').reset_index(drop=True)
            samples = trial_df[trial_df['Action'] == 'Movement_sample'].sort_values('t_sec').reset_index(drop=True)
            rxy, rt = trial_resources(trial_df)
            map_id = (trial_df['MapID'].dropna().iloc[0]
                      if trial_df['MapID'].notna().any() else '')

            modes, _ = predict_modes_for_trial(samples, rxy, rt, clf, scaler)
            feats = features_for_trial(trial_df, samples, modes, rxy, x_range, y_range)

            row = {'participant_id': int(pid), 'condition': condition,
                   'trial': int(trial), 'map_id': map_id, **feats}
            per_trial_rows.append(row)
            pid_pages[int(pid)].append({
                'trial': int(trial), 'map_id': map_id,
                'samples': samples, 'modes': modes, 'rxy': rxy,
            })
            t0, t_end = trial_window(trial_df)
            duration = (t_end - t0) if (pd.notna(t0) and pd.notna(t_end)) else np.nan
            pid_summary[int(pid)].append({
                'trial': int(trial), 'map_id': map_id,
                'duration': duration,
                'pct_exploit': feats['pct_time_exploit'],
                'n_collected': feats['total_collected'],
                'path_length': feats['path_length'],
            })
        if k % 20 == 0 or k == n_pids:
            print(f'  processed {k}/{n_pids} participants')

    # 5. Save per-trial + per-pid CSVs
    per_trial_df = pd.DataFrame(per_trial_rows).sort_values(['participant_id', 'trial'])
    per_trial_df.to_csv(PER_TRIAL_OUT, index=False, float_format='%.6g')
    print(f'wrote {PER_TRIAL_OUT.name}: {len(per_trial_df)} rows')
    pid_df_out = aggregate_per_pid(per_trial_df)
    pid_df_out.to_csv(PER_PID_OUT, index=False, float_format='%.6g')
    print(f'wrote {PER_PID_OUT.name}: {len(pid_df_out)} rows, {len(pid_df_out.columns)} cols')

    # 6. PDF
    print(f'rendering {PDF_OUT.name}...')
    pid_to_cond = dict(zip(pid_df_out['participant_id'], pid_df_out['condition']))
    with PdfPages(PDF_OUT) as pdf:
        for pid in sorted(pid_pages.keys()):
            trials = sorted(pid_pages[pid], key=lambda x: x['trial'])
            summary = sorted(pid_summary[pid], key=lambda x: x['trial'])
            plot_participant_page(pdf, pid, pid_to_cond[pid], trials,
                                  x_range, y_range, summary)
    print(f'wrote {PDF_OUT.name}')

    # 7. HTML explanation
    write_html_explanation(clf, len(train_df), loto_metrics)


if __name__ == '__main__':
    main()

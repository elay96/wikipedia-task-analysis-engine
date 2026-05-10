#!/usr/bin/env python3
"""
M75e: Iterate over feature sets / models / post-processing to maximise the
match with the user's hand-labels.

Outputs:
  output/m75e_iteration_log.txt     summary of every iteration
  output/m75e_best_model.json       best classifier (coefs/feature spec)
  output/m75e_best_compare.pdf      best model vs user labels per trial
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'output'

GAME_CSV = DATA_DIR / 'cleaned' / 'spatial_search' / 'Game.csv'
LABEL_DIR = DATA_DIR / 'manual labeling'

LOG_OUT = OUTPUT_DIR / 'm75e_iteration_log.txt'
MODEL_OUT = OUTPUT_DIR / 'm75e_best_model.json'
PDF_OUT = OUTPUT_DIR / 'm75e_best_compare.pdf'

WIN_1S = 10
WIN_2S = 20
WIN_3S = 30
COORD_SCALE = 3.0
EPS = 0.5

EXPLORE_COLOR = '#1976D2'
EXPLOIT_COLOR = '#C62828'
RESOURCE_COLOR = '#2E7D32'
BG = '#FFFFFF'


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


def rolling_window_indices(n, half):
    for i in range(n):
        yield i, max(0, i - half), min(n, i + half + 1)


# ---------- features ---------------------------------------------------------

def compute_all_features(samples, rxy, resource_t):
    """Compute every candidate feature for one trial.
    samples: DataFrame sorted by t_sec with columns X, Y, Heading, t_sec.
    rxy: (M, 2) world coords of Resource_found events.
    resource_t: (M,) seconds-since-midnight of those events.
    Returns dict {feature_name -> (n,) array}."""
    xs = samples['X'].values.astype(float)
    ys = samples['Y'].values.astype(float)
    hs = samples['Heading'].values.astype(float)
    ts = samples['t_sec'].values.astype(float)
    n = len(xs)
    feats = {}

    if n < 2:
        empty = np.full(n, np.nan)
        return {name: empty.copy() for name in (
            'angular_speed', 'sinuosity', 'inst_speed', 'speed_sd',
            'dist_to_nearest_resource', 'resource_density_60',
            'resource_density_30', 'resource_density_100',
            'displacement_2s', 'path_length_2s', 'radius_of_gyration_2s',
            'heading_sd_2s', 'time_since_last_resource',
            'recent_collection_count_2s', 'angular_speed_3s',
        )}

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

    feats['heading_sd_2s'] = rolling_std(np.concatenate([[0.0], wrap_deg(np.diff(hs))]), WIN_2S)

    sin_v = np.full(n, np.nan)
    disp_2s = np.full(n, np.nan)
    path_2s = np.full(n, np.nan)
    rog_2s = np.full(n, np.nan)
    half_1s = WIN_1S // 2
    half_2s = WIN_2S // 2
    for i, a, b in rolling_window_indices(n, half_1s):
        if b - a < 3:
            continue
        px = xs[a:b]; py = ys[a:b]
        path = float(np.sum(np.hypot(np.diff(px), np.diff(py))))
        disp = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
        sin_v[i] = 50.0 if disp < EPS else path / disp
    for i, a, b in rolling_window_indices(n, half_2s):
        if b - a < 5:
            continue
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
        time_since = np.full(n, 30.0)        # cap at 30s
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

    # First-passage time at radius 50 px (classical ARS detector). For each
    # sample i, how long does the path stay within 50 px of (x_i, y_i),
    # walking both forward and backward in time from i?
    fpt = np.zeros(n)
    R2 = 50.0 ** 2
    for i in range(n):
        t_back = ts[i]
        for j in range(i - 1, -1, -1):
            d2 = (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2
            if d2 > R2:
                t_back = ts[j]
                break
        else:
            t_back = ts[0]
        t_fwd = ts[i]
        for j in range(i + 1, n):
            d2 = (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2
            if d2 > R2:
                t_fwd = ts[j]
                break
        else:
            t_fwd = ts[-1]
        fpt[i] = max(0.0, t_fwd - t_back)
    feats['fpt_50'] = fpt

    # Same at smaller radius (tighter ARS / harvesting in place)
    fpt_30 = np.zeros(n)
    R2_30 = 30.0 ** 2
    for i in range(n):
        t_back = ts[i]
        for j in range(i - 1, -1, -1):
            d2 = (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2
            if d2 > R2_30:
                t_back = ts[j]
                break
        else:
            t_back = ts[0]
        t_fwd = ts[i]
        for j in range(i + 1, n):
            d2 = (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2
            if d2 > R2_30:
                t_fwd = ts[j]
                break
        else:
            t_fwd = ts[-1]
        fpt_30[i] = max(0.0, t_fwd - t_back)
    feats['fpt_30'] = fpt_30

    return feats


# ---------- data loading -----------------------------------------------------

def load_labels():
    files = sorted(LABEL_DIR.glob('m75_labels_*.json'))
    return [json.loads(f.read_text(encoding='utf-8')) for f in files]


def load_trial_samples(df, pid, trial):
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
    return samples, rxy, rt


def assemble_dataset(label_files, df):
    rows = []
    raw_samples = {}
    for d in label_files:
        pid, trial = d['pid'], d['trial']
        samples, rxy, rt = load_trial_samples(df, pid, trial)
        n = min(len(samples), len(d['labels']))
        feats = compute_all_features(samples.iloc[:n], rxy, rt)
        raw_samples[(pid, trial)] = (samples.iloc[:n], rxy)
        for i in range(n):
            row = {
                'pid': pid, 'trial': trial, 'sample_idx': i,
                'condition': d['condition'], 'map_id': d['map_id'],
                'label': d['labels'][i],
                'trial_key': f'{pid}_{trial}',
            }
            for fname, arr in feats.items():
                row[fname] = arr[i]
            rows.append(row)
    return pd.DataFrame(rows), raw_samples


# ---------- evaluation -------------------------------------------------------

def smooth_per_trial(pred_prob, groups, win=11):
    """Median-filter predicted probabilities within each trial."""
    out = pred_prob.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] = pd.Series(pred_prob[idx]).rolling(win, center=True,
                                                      min_periods=1).median().values
    return out


def viterbi_smooth_per_trial(pred_prob, groups, switch_log_penalty=2.5):
    """2-state Viterbi smoother. Higher penalty => stickier state.
    Returns the *posterior-like* score in [0,1] (state==1 weight)."""
    out = np.empty_like(pred_prob)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        p = np.clip(pred_prob[idx], 1e-6, 1 - 1e-6)
        n = len(p)
        log_e = np.log(p)
        log_e0 = np.log(1 - p)
        # log V[t, s]
        V = np.full((n, 2), -np.inf)
        bp = np.zeros((n, 2), dtype=int)
        V[0, 0] = log_e0[0]
        V[0, 1] = log_e[0]
        for t in range(1, n):
            for s in (0, 1):
                stay = V[t-1, s]
                switch = V[t-1, 1-s] - switch_log_penalty
                if stay >= switch:
                    V[t, s] = stay + (log_e0[t] if s == 0 else log_e[t])
                    bp[t, s] = s
                else:
                    V[t, s] = switch + (log_e0[t] if s == 0 else log_e[t])
                    bp[t, s] = 1 - s
        # backtrack
        path = np.zeros(n, dtype=int)
        path[-1] = int(np.argmax(V[-1]))
        for t in range(n - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        out[idx] = path.astype(float)
    return out


def hysteresis_smooth_per_trial(pred_prob, groups, p_low=0.40, p_high=0.60,
                                 init_state=0):
    """Schmitt-trigger style: switch to 1 when prob crosses p_high; switch
    back to 0 when prob crosses p_low. Returns binary 0/1 in [0,1] format."""
    out = np.empty_like(pred_prob)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        s = init_state
        seq = np.empty(len(idx))
        for k, i in enumerate(idx):
            if s == 0 and pred_prob[i] > p_high:
                s = 1
            elif s == 1 and pred_prob[i] < p_low:
                s = 0
            seq[k] = s
        out[idx] = seq
    return out


def _raw_loto_predictions(X, y, groups, model_factory):
    logo = LeaveOneGroupOut()
    pred = np.zeros(len(y), dtype=float)
    for tr, te in logo.split(X, y, groups):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = model_factory()
        clf.fit(Xtr, y[tr])
        pred[te] = clf.predict_proba(Xte)[:, 1]
    return pred


def loto_evaluate_with_threshold(X, y, groups, model_factory,
                                  smooth_kind='median', smooth_win=0,
                                  vit_penalty=2.5):
    """Run LOTO, optionally apply a smoother, sweep threshold for best acc."""
    raw = _raw_loto_predictions(X, y, groups, model_factory)

    if smooth_kind == 'median' and smooth_win > 1:
        pred = smooth_per_trial(raw, groups, smooth_win)
    elif smooth_kind == 'viterbi':
        pred = viterbi_smooth_per_trial(raw, groups, vit_penalty)
    elif smooth_kind == 'hysteresis':
        pred = hysteresis_smooth_per_trial(raw, groups, p_low=0.40, p_high=0.60)
    else:
        pred = raw

    pred_lab = (pred > 0.5).astype(int)
    auc = roc_auc_score(y, pred) if len(np.unique(y)) > 1 else float('nan')
    acc = accuracy_score(y, pred_lab)

    best = (0.5, acc)
    for thr in np.linspace(0.05, 0.95, 91):
        a = accuracy_score(y, (pred > thr).astype(int))
        if a > best[1]:
            best = (thr, a)
    return {
        'auc': auc, 'acc': acc,
        'cm': confusion_matrix(y, pred_lab).tolist(),
        'pred': pred,
        'best_threshold': float(best[0]),
        'best_threshold_acc': float(best[1]),
    }


def ensemble_evaluate(X, y, groups, factories, weights, smooth_kind='median',
                      smooth_win=21):
    """Average raw LOTO probs across models, then apply a smoother."""
    preds = [_raw_loto_predictions(X, y, groups, f) for f in factories]
    w = np.array(weights, dtype=float); w /= w.sum()
    raw = np.zeros_like(preds[0])
    for wi, p in zip(w, preds):
        raw += wi * p
    if smooth_kind == 'median' and smooth_win > 1:
        pred = smooth_per_trial(raw, groups, smooth_win)
    elif smooth_kind == 'viterbi':
        pred = viterbi_smooth_per_trial(raw, groups, 2.5)
    else:
        pred = raw
    pred_lab = (pred > 0.5).astype(int)
    auc = roc_auc_score(y, pred)
    acc = accuracy_score(y, pred_lab)
    best = (0.5, acc)
    for thr in np.linspace(0.05, 0.95, 91):
        a = accuracy_score(y, (pred > thr).astype(int))
        if a > best[1]:
            best = (thr, a)
    return {
        'auc': auc, 'acc': acc,
        'cm': confusion_matrix(y, pred_lab).tolist(),
        'pred': pred,
        'best_threshold': float(best[0]),
        'best_threshold_acc': float(best[1]),
    }


# ---------- model factories --------------------------------------------------

MODEL_FACTORIES = {
    'logreg':           lambda: LogisticRegression(max_iter=2000, C=1.0),
    'logreg_balanced':  lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'),
    'gbm':              lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                            learning_rate=0.05, random_state=0),
    'gbm_deep':         lambda: GradientBoostingClassifier(n_estimators=500, max_depth=4,
                                                            learning_rate=0.03,
                                                            subsample=0.8, random_state=0),
    'gbm_slow':         lambda: GradientBoostingClassifier(n_estimators=1000, max_depth=3,
                                                            learning_rate=0.015,
                                                            subsample=0.8, random_state=0),
    'rf':               lambda: RandomForestClassifier(n_estimators=300, max_depth=8,
                                                        min_samples_leaf=10, random_state=0,
                                                        n_jobs=-1),
    'rf_deeper':        lambda: RandomForestClassifier(n_estimators=600, max_depth=14,
                                                        min_samples_leaf=4, random_state=0,
                                                        n_jobs=-1),
}


# ---------- iteration plan ---------------------------------------------------

FEATURE_SETS = {
    'baseline_6': [
        'angular_speed', 'sinuosity', 'inst_speed',
        'speed_sd', 'dist_to_nearest_resource', 'resource_density_60',
    ],
    'plus_temporal': [
        'angular_speed', 'sinuosity', 'inst_speed', 'speed_sd',
        'dist_to_nearest_resource', 'resource_density_60',
        'time_since_last_resource', 'recent_collection_count_2s',
        'displacement_2s', 'radius_of_gyration_2s',
    ],
    'all_features': [
        'angular_speed', 'sinuosity', 'inst_speed', 'speed_sd',
        'dist_to_nearest_resource', 'resource_density_60',
        'resource_density_30', 'resource_density_100',
        'time_since_last_resource', 'recent_collection_count_2s',
        'displacement_2s', 'path_length_2s', 'radius_of_gyration_2s',
        'heading_sd_2s', 'angular_speed_3s',
    ],
    'all_plus_fpt': [
        'angular_speed', 'sinuosity', 'inst_speed', 'speed_sd',
        'dist_to_nearest_resource', 'resource_density_60',
        'resource_density_30', 'resource_density_100',
        'time_since_last_resource', 'recent_collection_count_2s',
        'displacement_2s', 'path_length_2s', 'radius_of_gyration_2s',
        'heading_sd_2s', 'angular_speed_3s',
        'fpt_30', 'fpt_50',
    ],
}

# tuple: (name, feature_set, model, smooth_kind, smooth_win|penalty)
# smooth_kind in {'none','median','viterbi','hysteresis'}.
# smooth_win is samples for median, ignored for viterbi/hysteresis.
ITERATIONS = [
    ('it01_logreg_baseline',           'baseline_6',     'logreg',          'none', 0),
    ('it02_logreg_baseline_smooth',    'baseline_6',     'logreg',          'median', 11),
    ('it03_logreg_temporal',           'plus_temporal',  'logreg',          'none', 0),
    ('it04_logreg_temporal_smooth',    'plus_temporal',  'logreg',          'median', 11),
    ('it05_logreg_all',                'all_features',   'logreg',          'none', 0),
    ('it06_logreg_all_smooth',         'all_features',   'logreg',          'median', 11),
    ('it09_gbm_all',                   'all_features',   'gbm',             'none', 0),
    ('it10_gbm_all_smooth',            'all_features',   'gbm',             'median', 11),
    ('it13_gbm_fpt',                   'all_plus_fpt',   'gbm',             'none', 0),
    ('it14_gbm_fpt_smooth11',          'all_plus_fpt',   'gbm',             'median', 11),
    ('it15_gbm_fpt_smooth21',          'all_plus_fpt',   'gbm',             'median', 21),
    ('it16_gbm_fpt_smooth31',          'all_plus_fpt',   'gbm',             'median', 31),
    ('it22_gbm_fpt_smooth41',          'all_plus_fpt',   'gbm',             'median', 41),
    ('it23_gbm_fpt_smooth51',          'all_plus_fpt',   'gbm',             'median', 51),
    ('it24_gbm_fpt_smooth71',          'all_plus_fpt',   'gbm',             'median', 71),
    ('it25_gbm_fpt_viterbi',           'all_plus_fpt',   'gbm',             'viterbi', 0),
    ('it26_gbm_fpt_hysteresis',        'all_plus_fpt',   'gbm',             'hysteresis', 0),
    ('it27_gbm_deep_fpt_smooth31',     'all_plus_fpt',   'gbm_deep',        'median', 31),
    ('it28_gbm_slow_fpt_smooth31',     'all_plus_fpt',   'gbm_slow',        'median', 31),
    ('it29_gbm_slow_fpt_smooth51',     'all_plus_fpt',   'gbm_slow',        'median', 51),
    ('it30_rf_deeper_fpt_smooth31',    'all_plus_fpt',   'rf_deeper',       'median', 31),
    ('it31_rf_deeper_fpt_smooth51',    'all_plus_fpt',   'rf_deeper',       'median', 51),
    ('it37_gbm_fpt_smooth91',          'all_plus_fpt',   'gbm',             'median', 91),
    ('it38_gbm_fpt_smooth121',         'all_plus_fpt',   'gbm',             'median', 121),
    ('it39_gbm_fpt_smooth151',         'all_plus_fpt',   'gbm',             'median', 151),
    ('it40_gbm_slow_smooth91',         'all_plus_fpt',   'gbm_slow',        'median', 91),
    ('it41_gbm_slow_smooth121',        'all_plus_fpt',   'gbm_slow',        'median', 121),
    ('it42_rf_deeper_smooth91',        'all_plus_fpt',   'rf_deeper',       'median', 91),
]

# Ensemble iterations: avg of GBM + RF, then smooth
ENSEMBLES = [
    ('it32_ens_gbm_rf_smooth31',  ['gbm', 'rf_deeper'], [1, 1], 'median', 31),
    ('it33_ens_gbm_rf_smooth51',  ['gbm', 'rf_deeper'], [1, 1], 'median', 51),
    ('it34_ens_gbm_rf_viterbi',   ['gbm', 'rf_deeper'], [1, 1], 'viterbi', 0),
    ('it35_ens_3way_smooth31',    ['gbm', 'gbm_slow', 'rf_deeper'], [1, 1, 1], 'median', 31),
    ('it36_ens_3way_smooth51',    ['gbm', 'gbm_slow', 'rf_deeper'], [1, 1, 1], 'median', 51),
    ('it43_ens_gbm_rf_smooth91',  ['gbm', 'rf_deeper'], [1, 1], 'median', 91),
    ('it44_ens_3way_smooth91',    ['gbm', 'gbm_slow', 'rf_deeper'], [1, 1, 1], 'median', 91),
    ('it45_ens_3way_smooth121',   ['gbm', 'gbm_slow', 'rf_deeper'], [1, 1, 1], 'median', 121),
]


# ---------- visualisation ----------------------------------------------------

def color_for(label):
    if label == 'explore':
        return EXPLORE_COLOR
    if label == 'exploit':
        return EXPLOIT_COLOR
    return '#9E9E9E'


def render_best_pdf(label_files, raw_samples, full_df_use, pred_oof,
                    x_range, y_range, name):
    with PdfPages(PDF_OUT) as pdf:
        for d in label_files:
            pid, trial = d['pid'], d['trial']
            mask = (full_df_use['pid'] == pid) & (full_df_use['trial'] == trial)
            n = mask.sum()
            samples, rxy = raw_samples[(pid, trial)]
            samples = samples.iloc[:n]
            user_lab = full_df_use.loc[mask, 'label'].values
            mdl_pred = (pred_oof[mask.values] > 0.5).astype(int)
            mdl_lab = np.where(mdl_pred == 1, 'exploit', 'explore')

            fig = plt.figure(figsize=(12, 6), facecolor=BG)
            gs = fig.add_gridspec(1, 2, wspace=0.05, left=0.03, right=0.99,
                                  top=0.86, bottom=0.04)
            fig.suptitle(f'pid {pid} trial {trial}  |  {d["condition"]}  |  {d["map_id"]}'
                         f'   ({name})',
                         fontsize=12, color='#1a1a1a')

            for col, (lab, title) in enumerate(zip([user_lab, mdl_lab],
                                                   ['User labels', 'Model'])):
                ax = fig.add_subplot(gs[0, col])
                ax.set_facecolor(BG)
                xs = samples['X'].values.astype(float)
                ys = samples['Y'].values.astype(float)
                if len(xs) >= 2:
                    for i in range(len(xs) - 1):
                        ax.plot(xs[i:i+2], ys[i:i+2], color=color_for(lab[i]),
                                linewidth=0.9, alpha=0.9)
                if len(rxy) > 0:
                    ax.scatter(rxy[:, 0], rxy[:, 1], marker='*', s=22,
                               color=RESOURCE_COLOR, edgecolors='none', zorder=3)
                if len(xs) > 0:
                    ax.scatter([xs[0]], [ys[0]], marker='o', s=42, color='#1a1a1a', zorder=4)
                    ax.scatter([xs[-1]], [ys[-1]], marker='D', s=30, color='#888', zorder=4)
                ax.set_xlim(x_range); ax.set_ylim(y_range)
                ax.invert_yaxis(); ax.set_aspect('equal')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(title, fontsize=11, color='#333')
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f'wrote {PDF_OUT.name}')


# ---------- main -------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    label_files = load_labels()
    print(f'Loaded {len(label_files)} label files')

    print('Loading Game.csv (filtered to labelled pids)...')
    df = pd.read_csv(GAME_CSV, low_memory=False)
    df = df[df['GameCondition'].isin(['Clumpy', 'Diffuse'])].copy()
    pids = sorted({d['pid'] for d in label_files})
    df = df[df['ID'].isin(pids)].copy()

    full_df, raw_samples = assemble_dataset(label_files, df)
    print(f'Total samples: {len(full_df)}')
    full_df = full_df[full_df['label'].isin(['explore', 'exploit'])].copy()

    # use the union of all candidate features for NaN dropping
    all_feat_cols = list(FEATURE_SETS['all_features'])
    finite = np.all(np.isfinite(full_df[all_feat_cols].values.astype(float)), axis=1)
    print(f'Rows with all features finite: {finite.sum()} / {len(full_df)}')
    full_df_use = full_df.loc[finite].reset_index(drop=True)
    y = (full_df_use['label'] == 'exploit').astype(int).values
    groups = full_df_use['trial_key'].values

    samples_full = df[df['Action'] == 'Movement_sample']
    x_range = (float(samples_full['X'].min()), float(samples_full['X'].max()))
    y_range = (float(samples_full['Y'].min()), float(samples_full['Y'].max()))

    log_lines = ['=== M75e iteration log ===',
                 f'Trials: {sorted(set(groups.tolist()))}',
                 f'Samples: {len(full_df_use)}  '
                 f'(exploit={y.sum()}, explore={(1-y).sum()})',
                 '']
    log_lines.append(f'{"name":34s}  {"features":15s}  {"model":17s}  '
                     f'{"smooth":7s}  {"AUC":>5s}  {"ACC":>5s}')

    results = []
    for name, fset, mdl_name, kind, win in ITERATIONS:
        cols = FEATURE_SETS[fset]
        X = full_df_use[cols].values.astype(float)
        try:
            res = loto_evaluate_with_threshold(
                X, y, groups, MODEL_FACTORIES[mdl_name],
                smooth_kind=kind, smooth_win=win)
        except Exception as e:
            print(f'  {name}: FAILED ({e})')
            log_lines.append(f'{name:34s}  FAILED: {e}')
            continue
        line = (f'{name:34s}  {fset:13s}  {mdl_name:14s}  '
                f'{kind:10s}  w={win:>3d}  '
                f'AUC={res["auc"]:.3f}  ACC={res["acc"]:.3f}  '
                f'tuned={res["best_threshold_acc"]:.3f}@{res["best_threshold"]:.2f}')
        print(line)
        log_lines.append(line)
        results.append({
            'name': name, 'feature_set': fset, 'model': mdl_name,
            'smooth_kind': kind, 'smooth_win': win,
            'auc': res['auc'], 'acc': res['acc'],
            'tuned_thr': res['best_threshold'],
            'tuned_acc': res['best_threshold_acc'],
            'cm': res['cm'], 'pred': res['pred'], 'feature_cols': cols,
            'is_ensemble': False,
        })

    # Ensemble iterations
    print('\n--- ensembles ---')
    for name, models, weights, kind, win in ENSEMBLES:
        cols = FEATURE_SETS['all_plus_fpt']
        X = full_df_use[cols].values.astype(float)
        try:
            res = ensemble_evaluate(
                X, y, groups,
                [MODEL_FACTORIES[m] for m in models], weights,
                smooth_kind=kind, smooth_win=win)
        except Exception as e:
            print(f'  {name}: FAILED ({e})')
            continue
        mdl_label = '+'.join(models)
        line = (f'{name:34s}  all_plus_fpt   {mdl_label:24s}  '
                f'{kind:10s}  w={win:>3d}  '
                f'AUC={res["auc"]:.3f}  ACC={res["acc"]:.3f}  '
                f'tuned={res["best_threshold_acc"]:.3f}@{res["best_threshold"]:.2f}')
        print(line)
        log_lines.append(line)
        results.append({
            'name': name, 'feature_set': 'all_plus_fpt', 'model': mdl_label,
            'smooth_kind': kind, 'smooth_win': win,
            'auc': res['auc'], 'acc': res['acc'],
            'tuned_thr': res['best_threshold'],
            'tuned_acc': res['best_threshold_acc'],
            'cm': res['cm'], 'pred': res['pred'], 'feature_cols': cols,
            'is_ensemble': True, 'ensemble_models': models,
            'ensemble_weights': weights,
        })

    results.sort(key=lambda r: r['tuned_acc'], reverse=True)
    log_lines.append('')
    log_lines.append('=== Sorted by tuned-threshold accuracy ===')
    for r in results:
        log_lines.append(
            f'  {r["name"]:34s}  AUC={r["auc"]:.3f}  ACC={r["acc"]:.3f}'
            f'  tunedACC={r["tuned_acc"]:.3f}@thr={r["tuned_thr"]:.2f}'
            f'  CM={r["cm"]}'
        )
    LOG_OUT.write_text('\n'.join(log_lines))
    print(f'\nwrote {LOG_OUT.name}')

    best = results[0]
    print(f'\n=== Best iteration: {best["name"]} '
          f'(tunedACC={best["tuned_acc"]:.3f}@thr={best["tuned_thr"]:.2f},'
          f' AUC={best["auc"]:.3f}) ===')
    print(f'  feature set:    {best["feature_set"]}')
    print(f'  model:          {best["model"]}')
    print(f'  smooth:         {best["smooth_kind"]} (win={best["smooth_win"]})')

    cols = best['feature_cols']
    X = full_df_use[cols].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    payload = {
        'best_iteration_name': best['name'],
        'feature_set': best['feature_set'],
        'feature_columns': cols,
        'model': best['model'],
        'smooth_kind': best['smooth_kind'],
        'smooth_window_samples': int(best['smooth_win']),
        'decision_threshold': float(best['tuned_thr']),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'loto_auc': float(best['auc']),
        'loto_accuracy_at_0.5': float(best['acc']),
        'loto_accuracy_tuned': float(best['tuned_acc']),
        'window_samples_1s': WIN_1S,
        'window_samples_2s': WIN_2S,
    }
    if not best.get('is_ensemble'):
        final_clf = MODEL_FACTORIES[best['model']]()
        final_clf.fit(Xs, y)
        if best['model'].startswith('logreg'):
            payload['logistic_coef'] = final_clf.coef_.flatten().tolist()
            payload['logistic_intercept'] = float(final_clf.intercept_[0])
        else:
            payload['feature_importance'] = final_clf.feature_importances_.tolist()
    else:
        payload['ensemble_models'] = best['ensemble_models']
        payload['ensemble_weights'] = best['ensemble_weights']
    MODEL_OUT.write_text(json.dumps(payload, indent=2))
    print(f'wrote {MODEL_OUT.name}')

    render_best_pdf(label_files, raw_samples, full_df_use, best['pred'],
                    x_range, y_range, best['name'])


if __name__ == '__main__':
    main()

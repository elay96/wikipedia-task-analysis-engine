#!/usr/bin/env python3
"""
M75d: Calibrate explore/exploit classifier from user-labelled trials.

Inputs:
  data/manual labeling/m75_labels_*.json   (user's per-sample labels)
  data/cleaned/spatial_search/Game.csv     (raw movement+resource events)

Pipeline:
  1. For each Movement_sample in the 6 labelled trials, compute behavioural
     features (angular speed, sinuosity, inst speed, speed variability,
     distance to nearest resource, local resource density).
  2. Align each sample to its user label.
  3. Fit logistic regression with Leave-One-Trial-Out CV.
  4. Compare to best single-feature threshold + the old Enter-events labels.
  5. Refit on all labelled samples to produce final coefficients.

Outputs:
  output/m75d_classifier_report.txt    summary metrics + coefficients
  output/m75d_classifier_model.json    coefficients + scaling for reuse
  output/m75d_classifier_compare.pdf   per-trial: user vs model vs Enter
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score, classification_report)
from sklearn.model_selection import LeaveOneGroupOut

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'output'

GAME_CSV = DATA_DIR / 'cleaned' / 'spatial_search' / 'Game.csv'
LABEL_DIR = DATA_DIR / 'manual labeling'

REPORT_OUT = OUTPUT_DIR / 'm75d_classifier_report.txt'
MODEL_OUT = OUTPUT_DIR / 'm75d_classifier_model.json'
PDF_OUT = OUTPUT_DIR / 'm75d_classifier_compare.pdf'

WINDOW_SAMPLES = 10                # ~1 second at 10 Hz
COORD_SCALE = 3.0
DENSITY_RADIUS = 60.0              # px in world coords for local-density count
EPS = 0.5

EXPLORE_COLOR = '#1976D2'
EXPLOIT_COLOR = '#C62828'
RESOURCE_COLOR = '#2E7D32'
START_COLOR = '#1a1a1a'
END_COLOR = '#888888'
UNKNOWN_COLOR = '#9E9E9E'
BG = '#FFFFFF'

FEATURE_NAMES = [
    'angular_speed', 'sinuosity', 'inst_speed',
    'speed_sd', 'dist_to_nearest_resource', 'resource_density',
]


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


# ---------- feature extraction -----------------------------------------------

def features_for_trial(samples, resources_world):
    """Returns a (n_samples, n_features) array. samples is a sorted DataFrame
    with X, Y, Heading, t_sec. resources_world is array of [N,2] world coords."""
    xs = samples['X'].values.astype(float)
    ys = samples['Y'].values.astype(float)
    hs = samples['Heading'].values.astype(float)
    ts = samples['t_sec'].values.astype(float)
    n = len(xs)

    # angular speed (rolling mean of |dHeading|/dt)
    if n >= 2:
        dh = np.abs(wrap_deg(np.diff(hs)))
        dt = np.maximum(np.diff(ts), 1e-3)
        inst_ang = np.concatenate([[0.0], dh / dt])
        ang = pd.Series(inst_ang).rolling(WINDOW_SAMPLES, center=True, min_periods=3).mean().values
    else:
        ang = np.full(n, np.nan)

    # instantaneous speed (px/sec) and rolling smoothed speed + sd
    if n >= 2:
        dx = np.diff(xs); dy = np.diff(ys)
        dist = np.hypot(dx, dy)
        dt = np.maximum(np.diff(ts), 1e-3)
        inst_v = np.concatenate([[0.0], dist / dt])
        v_mean = pd.Series(inst_v).rolling(WINDOW_SAMPLES, center=True, min_periods=3).mean().values
        v_sd = pd.Series(inst_v).rolling(WINDOW_SAMPLES, center=True, min_periods=3).std().values
    else:
        v_mean = v_sd = np.full(n, np.nan)

    # sinuosity: per sample, ratio of path-length to displacement in window
    half = WINDOW_SAMPLES // 2
    sin_v = np.full(n, np.nan)
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        if b - a < 3:
            continue
        px = xs[a:b]; py = ys[a:b]
        path = float(np.sum(np.hypot(np.diff(px), np.diff(py))))
        disp = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
        sin_v[i] = 50.0 if disp < EPS else path / disp

    # distance to nearest resource + local density
    if len(resources_world) > 0:
        rx = resources_world[:, 0]
        ry = resources_world[:, 1]
        diff_x = xs[:, None] - rx[None, :]
        diff_y = ys[:, None] - ry[None, :]
        d2 = diff_x ** 2 + diff_y ** 2
        d_nearest = np.sqrt(d2.min(axis=1))
        density = (d2 <= DENSITY_RADIUS ** 2).sum(axis=1).astype(float)
    else:
        d_nearest = np.full(n, np.nan)
        density = np.zeros(n)

    return np.column_stack([ang, sin_v, v_mean, v_sd, d_nearest, density])


# ---------- load data --------------------------------------------------------

def load_labels():
    files = sorted(LABEL_DIR.glob('m75_labels_*.json'))
    out = []
    for f in files:
        d = json.loads(f.read_text(encoding='utf-8'))
        out.append(d)
    print(f'Loaded {len(out)} label files')
    return out


def load_trial_samples(df, pid, trial):
    sub = df[(df['ID'] == pid) & (df['Trial'] == trial)].copy()
    sub['t_sec'] = sub['Time'].apply(parse_time_to_seconds)
    sub = sub.sort_values('t_sec')
    samples = sub[sub['Action'] == 'Movement_sample'].sort_values('t_sec').reset_index(drop=True)
    res = sub[sub['Action'] == 'Resource_found']
    rxy = []
    for _, r in res.iterrows():
        if pd.notna(r['ResourceX']) and pd.notna(r['ResourceY']):
            rxy.append([float(r['ResourceX']) * COORD_SCALE,
                        float(r['ResourceY']) * COORD_SCALE])
    rxy = np.array(rxy) if rxy else np.zeros((0, 2))

    # mode from Enter events ("old" baseline) per sample
    has_enter = sub['Action'].isin(['Enter_explore_mode', 'Enter_exploit_mode']).any()
    if not has_enter:
        old_mode = np.array(['unknown'] * len(samples), dtype=object)
    else:
        cur = 'explore'
        old_mode = []
        ev = sub.sort_values('t_sec')
        sample_t = samples['t_sec'].values
        ev_idx = 0
        ev_list = ev[['Action', 't_sec']].values.tolist()
        for st in sample_t:
            while ev_idx < len(ev_list) and ev_list[ev_idx][1] <= st:
                act = ev_list[ev_idx][0]
                if act == 'Enter_explore_mode':
                    cur = 'explore'
                elif act == 'Enter_exploit_mode':
                    cur = 'exploit'
                ev_idx += 1
            old_mode.append(cur)
        old_mode = np.array(old_mode, dtype=object)

    return samples, rxy, old_mode


# ---------- training ---------------------------------------------------------

def assemble_dataset(label_files, df):
    rows = []
    for d in label_files:
        pid, trial = d['pid'], d['trial']
        samples, rxy, old_mode = load_trial_samples(df, pid, trial)
        if len(samples) != d['n_samples']:
            print(f'  WARNING: pid {pid} trial {trial}: '
                  f'{len(samples)} samples in CSV vs {d["n_samples"]} in label file')
        n = min(len(samples), len(d['labels']))
        feats = features_for_trial(samples.iloc[:n], rxy)
        for i in range(n):
            rows.append({
                'pid': pid, 'trial': trial, 'sample_idx': i,
                'condition': d['condition'], 'map_id': d['map_id'],
                'label': d['labels'][i], 'old_mode': old_mode[i] if i < len(old_mode) else 'unknown',
                **{name: feats[i, j] for j, name in enumerate(FEATURE_NAMES)},
            })
    return pd.DataFrame(rows)


def fit_logistic_loto(X, y, groups):
    """Leave-One-Trial-Out CV. Returns out-of-fold predictions and per-fold AUC."""
    logo = LeaveOneGroupOut()
    pred = np.zeros_like(y, dtype=float)
    fold_aucs = []
    for fold_i, (tr, te) in enumerate(logo.split(X, y, groups)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        pred[te] = clf.predict_proba(Xte)[:, 1]
        if len(np.unique(y[te])) > 1:
            fold_aucs.append(roc_auc_score(y[te], pred[te]))
        held_out = np.unique(groups[te])[0]
        print(f'  fold {fold_i+1}: held-out trial = {held_out}, '
              f'fold AUC = {fold_aucs[-1]:.3f}' if fold_aucs else '')
    return pred, fold_aucs


def fit_final_logistic(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xs, y)
    return clf, scaler


def best_single_feature_threshold(X, y, names):
    """For each feature, find the threshold maximising balanced accuracy."""
    best = None
    for j, name in enumerate(names):
        col = X[:, j]
        if np.all(np.isnan(col)):
            continue
        finite = ~np.isnan(col)
        candidates = np.unique(col[finite])
        candidates = candidates[::max(1, len(candidates) // 200)]   # subsample for speed
        for thr in candidates:
            pred = (col > thr).astype(int)
            pred[~finite] = 0
            if len(np.unique(pred[finite])) < 2:
                continue
            acc = accuracy_score(y[finite], pred[finite])
            if best is None or acc > best['acc']:
                best = {'feature': name, 'threshold': float(thr), 'acc': float(acc),
                        'n_used': int(finite.sum())}
    return best


# ---------- visualisation ----------------------------------------------------

def color_for(label):
    if label == 'explore':
        return EXPLORE_COLOR
    if label == 'exploit':
        return EXPLOIT_COLOR
    return UNKNOWN_COLOR


def draw_path(ax, samples, modes, rxy, x_range, y_range, title):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
    xs = samples['X'].values.astype(float)
    ys = samples['Y'].values.astype(float)
    if len(xs) >= 2:
        for i in range(len(xs) - 1):
            ax.plot(xs[i:i+2], ys[i:i+2], color=color_for(modes[i]),
                    linewidth=0.9, alpha=0.9)
    if len(rxy) > 0:
        ax.scatter(rxy[:, 0], rxy[:, 1], marker='*', s=22,
                   color=RESOURCE_COLOR, edgecolors='none', zorder=3)
    if len(xs) > 0:
        ax.scatter([xs[0]], [ys[0]], marker='o', s=42, color=START_COLOR, zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], marker='D', s=30, color=END_COLOR, zorder=4)
    ax.set_xlim(x_range); ax.set_ylim(y_range)
    ax.invert_yaxis(); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, color='#333333')


def render_compare_pdf(label_files, df_full, full_data, prob_oof, X_all,
                       feature_names, single_thr):
    """One page per labelled trial, 3 rows: user / logistic / single-feature."""
    samples_full = df_full[df_full['Action'] == 'Movement_sample']
    x_range = (float(samples_full['X'].min()), float(samples_full['X'].max()))
    y_range = (float(samples_full['Y'].min()), float(samples_full['Y'].max()))

    feat_idx = feature_names.index(single_thr['feature'])
    with PdfPages(PDF_OUT) as pdf:
        for d in label_files:
            pid, trial = d['pid'], d['trial']
            samples, rxy, old_mode = load_trial_samples(df_full, pid, trial)
            mask = (full_data['pid'] == pid) & (full_data['trial'] == trial)
            n = mask.sum()
            user_lab = full_data.loc[mask, 'label'].values
            logi_pred = (prob_oof[mask.values] > 0.5).astype(int)
            logi_lab = np.where(logi_pred == 1, 'exploit', 'explore')
            sgl_pred = (X_all[mask.values, feat_idx] > single_thr['threshold']).astype(int)
            sgl_lab = np.where(sgl_pred == 1, 'exploit', 'explore')

            fig = plt.figure(figsize=(8, 9), facecolor=BG)
            gs = fig.add_gridspec(3, 1, hspace=0.4, left=0.05, right=0.97,
                                  top=0.92, bottom=0.04)
            fig.suptitle(f'pid {pid} trial {trial}  |  {d["condition"]}  |  {d["map_id"]}',
                         fontsize=12, color='#1a1a1a')
            draw_path(fig.add_subplot(gs[0]), samples.iloc[:n], user_lab, rxy,
                      x_range, y_range, 'User labels (ground truth)')
            draw_path(fig.add_subplot(gs[1]), samples.iloc[:n], logi_lab, rxy,
                      x_range, y_range, 'Logistic regression (LOTO out-of-fold)')
            draw_path(fig.add_subplot(gs[2]), samples.iloc[:n], sgl_lab, rxy,
                      x_range, y_range,
                      f'Single feature: {single_thr["feature"]} > {single_thr["threshold"]:.2f}')
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f'wrote {PDF_OUT.name}')


# ---------- main -------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    label_files = load_labels()

    print('Loading Game.csv (filtered to labelled pids)...')
    df = pd.read_csv(GAME_CSV, low_memory=False)
    df = df[df['GameCondition'].isin(['Clumpy', 'Diffuse'])].copy()
    pids = sorted({d['pid'] for d in label_files})
    df = df[df['ID'].isin(pids)].copy()

    full_data = assemble_dataset(label_files, df)
    print(f'Total samples in dataset: {len(full_data)}')
    full_data = full_data[full_data['label'].isin(['explore', 'exploit'])].copy()
    print(f'After dropping unlabelled: {len(full_data)}')
    full_data['trial_key'] = full_data['pid'].astype(str) + '_' + full_data['trial'].astype(str)

    # Drop rows with NaN features (rolling-window edges)
    feat_arr = full_data[FEATURE_NAMES].values.astype(float)
    finite_mask = np.all(np.isfinite(feat_arr), axis=1)
    print(f'Rows with all features finite: {finite_mask.sum()} / {len(full_data)}')
    df_use = full_data.loc[finite_mask].reset_index(drop=True)
    X = df_use[FEATURE_NAMES].values.astype(float)
    y = (df_use['label'] == 'exploit').astype(int).values
    groups = df_use['trial_key'].values

    print(f'\n=== Class balance ===')
    print(f'  exploit: {y.sum()} ({100 * y.mean():.1f}%)')
    print(f'  explore: {(1 - y).sum()} ({100 * (1 - y.mean()):.1f}%)')

    print(f'\n=== Logistic regression (Leave-One-Trial-Out CV) ===')
    pred_oof, fold_aucs = fit_logistic_loto(X, y, groups)
    pred_oof_lab = (pred_oof > 0.5).astype(int)
    overall_auc = roc_auc_score(y, pred_oof) if len(np.unique(y)) > 1 else float('nan')
    overall_acc = accuracy_score(y, pred_oof_lab)
    print(f'  pooled AUC: {overall_auc:.3f}')
    print(f'  pooled accuracy: {overall_acc:.3f}')
    print(f'  mean fold AUC: {np.mean(fold_aucs):.3f}')

    print(f'\n=== Best single-feature baseline ===')
    sgl = best_single_feature_threshold(X, y, FEATURE_NAMES)
    print(f'  feature: {sgl["feature"]}, threshold: {sgl["threshold"]:.3f}, '
          f'accuracy: {sgl["acc"]:.3f}')

    print(f'\n=== Old Enter-events baseline ===')
    old = df_use['old_mode'].values
    old_pred = (old == 'exploit').astype(int)
    old_acc = accuracy_score(y, old_pred)
    print(f'  accuracy vs user labels: {old_acc:.3f}')

    print(f'\n=== Final logistic (fit on all labelled samples) ===')
    clf, scaler = fit_final_logistic(X, y)
    coefs = clf.coef_.flatten()
    intercept = float(clf.intercept_[0])
    print(f'  intercept: {intercept:.3f}')
    for name, c in zip(FEATURE_NAMES, coefs):
        print(f'    {name:30s}  coef = {c:+.3f}')

    final_pred = clf.predict(scaler.transform(X))
    final_acc = accuracy_score(y, final_pred)
    print(f'  in-sample accuracy (overfit-bound): {final_acc:.3f}')

    print('\n=== Confusion matrix (LOTO predictions) ===')
    cm = confusion_matrix(y, pred_oof_lab)
    print(f'              pred-explore  pred-exploit')
    print(f'true-explore        {cm[0,0]:5d}        {cm[0,1]:5d}')
    print(f'true-exploit        {cm[1,0]:5d}        {cm[1,1]:5d}')

    # save model
    model_payload = {
        'feature_names': FEATURE_NAMES,
        'window_samples': WINDOW_SAMPLES,
        'density_radius_world_px': DENSITY_RADIUS,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'logistic_coef': coefs.tolist(),
        'logistic_intercept': intercept,
        'positive_class': 'exploit',
        'training_n_samples': int(X.shape[0]),
        'training_trials': sorted(set(groups.tolist())),
        'loto_pooled_auc': float(overall_auc),
        'loto_pooled_accuracy': float(overall_acc),
        'single_feature_baseline': sgl,
        'old_enter_events_baseline_acc': float(old_acc),
    }
    MODEL_OUT.write_text(json.dumps(model_payload, indent=2))
    print(f'\nwrote {MODEL_OUT.name}')

    # full report on disk
    report_lines = [
        '=== M75d classifier calibration report ===',
        f'Trials used: {sorted(set(groups.tolist()))}',
        f'Total labelled samples: {len(full_data)}',
        f'After NaN drop: {len(df_use)}  (exploit={y.sum()}, explore={(1-y).sum()})',
        '',
        f'Logistic regression LOTO pooled AUC = {overall_auc:.3f}',
        f'Logistic regression LOTO pooled accuracy = {overall_acc:.3f}',
        f'Logistic regression in-sample accuracy   = {final_acc:.3f}',
        f'Best single-feature baseline = {sgl["feature"]} > {sgl["threshold"]:.3f} '
        f'-> acc = {sgl["acc"]:.3f}',
        f'Old Enter-events baseline acc = {old_acc:.3f}',
        '',
        'Coefficients (on standardised features):',
    ]
    for name, c in zip(FEATURE_NAMES, coefs):
        report_lines.append(f'  {name:30s}  coef = {c:+.3f}')
    report_lines.append(f'  intercept = {intercept:+.3f}')
    report_lines.append('')
    report_lines.append('Per-fold AUC:')
    for trial, auc in zip(sorted(set(groups.tolist())), fold_aucs):
        report_lines.append(f'  {trial}: {auc:.3f}')
    report_lines.append('')
    report_lines.append(classification_report(y, pred_oof_lab,
                                              target_names=['explore', 'exploit']))
    REPORT_OUT.write_text('\n'.join(report_lines))
    print(f'wrote {REPORT_OUT.name}')

    # PDF comparison
    full_data_use = df_use.copy()
    render_compare_pdf(label_files, df, full_data_use, pred_oof, X,
                       FEATURE_NAMES, sgl)


if __name__ == '__main__':
    main()

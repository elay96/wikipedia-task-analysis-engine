#!/usr/bin/env python3
"""
M75 (test): Mode-inference algorithm comparison
================================================
Compares two behaviour-derived classifiers for explore/exploit mode (replacing
the Enter_*_mode events). 4 participants x 5 trials.

  A. Angular Speed   - rolling |dHeading|/dt over a 1s window (deg/sec).
                       Above global median -> exploit, else -> explore.

  B. Sinuosity       - rolling path_length / displacement over a 1s window.
                       Above global median -> exploit, else -> explore.

Each PDF page has the same participant rendered twice:
  top row    = Algorithm A on 5 trials
  bottom row = Algorithm B on the same 5 trials

Output: output/m75_mode_inference_test.pdf
"""

from __future__ import annotations

from pathlib import Path

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
PDF_OUT = OUTPUT_DIR / 'm75_mode_inference_test.pdf'

TEST_PIDS = [1, 8, 13, 100]
WINDOW_SAMPLES = 10           # ~1 second at 10 Hz
COORD_SCALE = 3.0
EPS = 0.5                     # min displacement to compute sinuosity

EXPLORE_COLOR = '#1976D2'     # blue
EXPLOIT_COLOR = '#C62828'     # red
RESOURCE_COLOR = '#2E7D32'    # green
START_COLOR = '#1a1a1a'
END_COLOR = '#888888'
BG = '#FFFFFF'


# ---------- helpers ----------------------------------------------------------

def parse_time_to_seconds(s):
    if not isinstance(s, str):
        return np.nan
    parts = s.split(':')
    if len(parts) != 3:
        return np.nan
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, TypeError):
        return np.nan


def wrap_deg(d):
    return ((d + 180.0) % 360.0) - 180.0


# ---------- mode classifiers -------------------------------------------------

def angular_speed_per_sample(samples):
    """Smoothed |dHeading|/dt (deg/sec) per Movement_sample.
    Uses rolling MEAN: median is degenerate (>50% zeros from integer-rounded
    Heading + straight motion) so the mean preserves sub-second turn signal."""
    headings = samples['Heading'].values.astype(float)
    times = samples['t_sec'].values.astype(float)
    if len(headings) < 2:
        return np.full(len(headings), np.nan)
    dh = np.abs(wrap_deg(np.diff(headings)))
    dt = np.maximum(np.diff(times), 1e-3)
    inst = np.concatenate([[0.0], dh / dt])
    return pd.Series(inst).rolling(WINDOW_SAMPLES, center=True, min_periods=3).mean().values


def sinuosity_per_sample(samples):
    """Rolling path_length / displacement around each sample (1s window)."""
    xs = samples['X'].values.astype(float)
    ys = samples['Y'].values.astype(float)
    n = len(xs)
    half = WINDOW_SAMPLES // 2
    out = np.full(n, np.nan)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        if b - a < 3:
            continue
        px = xs[a:b]
        py = ys[a:b]
        path = float(np.sum(np.hypot(np.diff(px), np.diff(py))))
        disp = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
        if disp < EPS:
            out[i] = 50.0   # cap stationary windows at a high but finite value
        else:
            out[i] = path / disp
    return out


def modes_from_metric(metric, threshold):
    """metric > threshold -> 'exploit', else 'explore'. NaN -> 'explore'."""
    out = np.where(metric > threshold, 'exploit', 'explore')
    out[np.isnan(metric)] = 'explore'
    return out


# ---------- data loading -----------------------------------------------------

def load_cohort(game_csv, pids):
    df = pd.read_csv(game_csv, low_memory=False)
    df = df[df['ID'].isin(pids)].copy()
    df = df[df['GameCondition'].isin(['Clumpy', 'Diffuse'])].copy()
    df['t_sec'] = df['Time'].apply(parse_time_to_seconds)
    return df


def load_all_movement_samples(game_csv, cohort_csv):
    """Load Movement_samples for the FULL 132-cohort, used to compute the
    global thresholds for both algorithms."""
    cohort = pd.read_csv(cohort_csv)
    cohort_ids = set(cohort['participant_id'].astype(int).tolist())
    df = pd.read_csv(game_csv, low_memory=False)
    df = df[df['ID'].isin(cohort_ids)].copy()
    df = df[df['GameCondition'].isin(['Clumpy', 'Diffuse'])].copy()
    df['t_sec'] = df['Time'].apply(parse_time_to_seconds)
    trial_counts = df.groupby('ID')['Trial'].nunique()
    keep = trial_counts[trial_counts == 5].index.tolist()
    return df[df['ID'].isin(keep)].copy()


def compute_global_thresholds(df_cohort):
    print('computing global thresholds across N=132...')
    ang_all = []
    sin_all = []
    for (_, _), grp in df_cohort.groupby(['ID', 'Trial']):
        samp = grp[grp['Action'] == 'Movement_sample'].sort_values('t_sec')
        if len(samp) < 5:
            continue
        ang_all.append(angular_speed_per_sample(samp))
        sin_all.append(sinuosity_per_sample(samp))
    ang_all = np.concatenate(ang_all)
    sin_all = np.concatenate(sin_all)

    print('  Angular speed (rolling mean) percentiles (deg/s):')
    for p in [25, 40, 50, 60, 70, 75, 80, 90]:
        print(f'    p{p}: {np.nanpercentile(ang_all, p):.2f}')
    print('  Sinuosity percentiles:')
    for p in [25, 40, 50, 60, 70, 75, 80, 90]:
        print(f'    p{p}: {np.nanpercentile(sin_all, p):.3f}')
    return ang_all, sin_all


# ---------- plotting ---------------------------------------------------------

def draw_trial(ax, samples, modes, resource_xs, resource_ys, x_range, y_range,
               trial_n, map_id):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
    xs = samples['X'].values.astype(float)
    ys = samples['Y'].values.astype(float)

    if len(xs) >= 2:
        for i in range(len(xs) - 1):
            c = EXPLOIT_COLOR if modes[i] == 'exploit' else EXPLORE_COLOR
            ax.plot(xs[i:i+2], ys[i:i+2], color=c, linewidth=0.8, alpha=0.9)

    if len(resource_xs) > 0:
        ax.scatter(resource_xs, resource_ys, marker='*', s=22,
                   color=RESOURCE_COLOR, edgecolors='none', zorder=3)
    if len(xs) > 0:
        ax.scatter([xs[0]], [ys[0]], marker='o', s=42, color=START_COLOR, zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], marker='D', s=30, color=END_COLOR, zorder=4)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Trial {trial_n}  |  {map_id}', fontsize=10, color='#333333')


def render_pid_page(pdf, pid, pid_df, x_range, y_range, threshold_specs):
    """Render one PDF page per participant with N rows of trials.
    threshold_specs: list of (algorithm, threshold_value, label) tuples,
    one per row. algorithm in {'A', 'B'}."""
    condition = pid_df['GameCondition'].iloc[0]
    n_rows = len(threshold_specs)
    fig = plt.figure(figsize=(16, 3.0 * n_rows + 1.2), facecolor=BG)
    gs = fig.add_gridspec(n_rows, 5, hspace=0.55, wspace=0.18,
                          left=0.05, right=0.99,
                          top=1 - 0.9 / (3.0 * n_rows + 1.2),
                          bottom=0.04)

    fig.suptitle(f'Participant {pid}  |  Condition: {condition}',
                 fontsize=13, color='#1a1a1a', y=0.985)

    trial_groups = sorted(pid_df['Trial'].unique())[:5]
    for col, trial in enumerate(trial_groups):
        trial_df = pid_df[pid_df['Trial'] == trial].sort_values('t_sec')
        samples = trial_df[trial_df['Action'] == 'Movement_sample'].sort_values('t_sec')
        if len(samples) < 5:
            continue
        res = trial_df[trial_df['Action'] == 'Resource_found']
        rx = res['ResourceX'].values.astype(float) * COORD_SCALE
        ry = res['ResourceY'].values.astype(float) * COORD_SCALE
        map_id = (trial_df['MapID'].dropna().iloc[0]
                  if trial_df['MapID'].notna().any() else '')
        ang = angular_speed_per_sample(samples)
        sin_v = sinuosity_per_sample(samples)

        for row, (alg, thr, _) in enumerate(threshold_specs):
            metric = ang if alg == 'A' else sin_v
            modes = modes_from_metric(metric, thr)
            ax = fig.add_subplot(gs[row, col])
            draw_trial(ax, samples, modes, rx, ry, x_range, y_range, trial, map_id)

    # row labels (left of first column)
    for row, (alg, thr, label) in enumerate(threshold_specs):
        fig.text(0.005, gs[row, 0].get_position(fig).y0
                 + gs[row, 0].get_position(fig).height / 2,
                 label, rotation=90, va='center', ha='left',
                 fontsize=10, color='#333333', fontweight='bold')

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ---------- main -------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cohort_csv = OUTPUT_DIR / 'm72_new_per_participant.csv'

    df_full = load_all_movement_samples(GAME_CSV, cohort_csv)
    samples_full = df_full[df_full['Action'] == 'Movement_sample']
    x_range = (float(samples_full['X'].min()), float(samples_full['X'].max()))
    y_range = (float(samples_full['Y'].min()), float(samples_full['Y'].max()))

    compute_global_thresholds(df_full)

    # 4 rows per page: 2 thresholds for each algorithm (strict & loose).
    # Tweaks here are quick to iterate.
    threshold_specs = [
        ('A', 30.0,  'A: Angular @ 30 deg/s  (loose -> more red)'),
        ('A', 80.0,  'A: Angular @ 80 deg/s  (strict -> less red)'),
        ('B', 1.10,  'B: Sinuosity @ 1.10  (loose -> more red)'),
        ('B', 1.30,  'B: Sinuosity @ 1.30  (strict -> less red)'),
    ]

    pid_df = df_full[df_full['ID'].isin(TEST_PIDS)]
    print(f'\nrendering test PDF for pids {TEST_PIDS}...')
    with PdfPages(PDF_OUT) as pdf:
        for pid in TEST_PIDS:
            sub = pid_df[pid_df['ID'] == pid]
            if len(sub) == 0:
                print(f'  pid {pid}: not in cohort, skipping')
                continue
            render_pid_page(pdf, pid, sub, x_range, y_range, threshold_specs)
            print(f'  pid {pid}: page rendered')
    print(f'wrote {PDF_OUT}')


if __name__ == '__main__':
    main()

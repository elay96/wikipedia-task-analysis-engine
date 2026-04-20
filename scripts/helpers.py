#!/usr/bin/env python3
"""
Shared helpers for pilot explore-exploit measure scripts.
=========================================================
Data loading, typing detection, and common plot utilities.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'
CLEANED_DIR = DATA_DIR / 'cleaned'
DEFAULT_DATA = CLEANED_DIR / 'Game.csv'
OUTPUT_DIR = SCRIPT_DIR / '..' / 'output'


def load_trials(csv_path=None):
    """Load Game.csv and return structured trial list."""
    if csv_path is None:
        csv_path = DEFAULT_DATA
    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    real = df[df['IsPractice'] != 1].copy()

    trials = []
    for pid in sorted(real['ID'].unique()):
        for ti in sorted(real[real['ID'] == pid]['TrialIndex'].unique()):
            t = real[(real['ID'] == pid) & (real['TrialIndex'] == ti)].sort_values('Time')
            row0 = t.iloc[0]
            starts = t[t['Action'] == 'task_start']['Time']
            ends = t[t['Action'] == 'task_end']['Time']
            if len(ends) == 0:
                continue
            t0 = starts.iloc[0] if len(starts) > 0 else t['Time'].iloc[0]
            t_end = ends.iloc[0]
            duration = (t_end - t0).total_seconds()
            if duration <= 0:
                continue

            # Articles
            articles = t[t['Action'] == 'article_open'].sort_values('Time')
            page_visits = []
            for i in range(len(articles)):
                row = articles.iloc[i]
                open_time = row['Time']
                close_time = articles.iloc[i + 1]['Time'] if i + 1 < len(articles) else t_end
                page_visits.append({
                    'title': row['ArticleTitle'],
                    'start': (open_time - t0).total_seconds(),
                    'end': (close_time - t0).total_seconds(),
                    'duration': (close_time - open_time).total_seconds(),
                    'nav_type': row['NavigationType'],
                })

            # Pastes
            pastes = t[t['Action'] == 'paste'].sort_values('Time')
            paste_times = [(pt - t0).total_seconds() for pt in pastes['Time']]

            # Typing intervals
            typing_intervals = _detect_typing(t, t0)

            trials.append({
                'pid': pid, 'trial': ti,
                'domain': row0.get('Domain', ''),
                'condition': row0.get('Condition', ''),
                'duration': duration,
                'page_visits': page_visits,
                'paste_times': paste_times,
                'typing_intervals': typing_intervals,
                'events': t, 't0': t0,
            })
    return trials


def _detect_typing(trial_events, t0):
    """Detect typing bursts from answer_snapshot events."""
    snaps = trial_events[trial_events['Action'] == 'answer_snapshot'].sort_values('Time')
    if len(snaps) < 2:
        return []
    snap_times = [(t - t0).total_seconds() for t in snaps['Time']]
    snap_lens = snaps['AnswerLength'].astype(float).fillna(0).tolist()
    bursts = []
    bs, be, prev, chg = snap_times[0], snap_times[0], snap_lens[0], False
    for k in range(1, len(snap_times)):
        if snap_times[k] - snap_times[k - 1] <= 15:
            be = snap_times[k]
            if abs(snap_lens[k] - prev) > 0:
                chg = True
        else:
            if chg and be - bs >= 1:
                bursts.append((bs, be))
            bs, be, chg = snap_times[k], snap_times[k], False
        prev = snap_lens[k]
    if chg and be - bs >= 1:
        bursts.append((bs, be))
    return bursts


def get_pids_and_trials(trials):
    """Return sorted PIDs and pid->trials dict."""
    pids = sorted(set(tr['pid'] for tr in trials))
    pid_trials = {p: sorted([t for t in trials if t['pid'] == p], key=lambda t: t['trial']) for p in pids}
    return pids, pid_trials


def setup_figure(n_participants, title_text):
    """Create standard 3-panel figure layout."""
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], hspace=0.3, wspace=0.3)
    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])
    fig.suptitle(title_text, fontsize=14, fontweight='bold')
    return fig, ax_top, ax_bl, ax_br


def finish_timeline(ax, pids, xlabel='Time within trial (seconds)'):
    """Style the timeline axis."""
    y_positions = [(len(pids) - i - 1) for i in range(len(pids))]
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"P{pid}" for pid in pids], fontsize=9, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_xlim(-10, None)
    for y in y_positions:
        ax.axhline(y=y, color='#E0E0E0', linewidth=0.3, zorder=0)


def sorted_ratio_panel(ax, pids, ratios, exploit_color, explore_color, label, xlabel):
    """Draw sorted horizontal bar chart of ratios."""
    idx = np.argsort(ratios)
    s_pids = [pids[i] for i in idx]
    s_ratios = [ratios[i] for i in idx]
    colors = [exploit_color if r > 0.5 else explore_color for r in s_ratios]
    ax.barh(range(len(s_pids)), s_ratios, color=colors, edgecolor='gray', height=0.6)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    mean_r = np.mean(ratios)
    ax.axvline(x=mean_r, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_r:.2f}')
    ax.set_yticks(range(len(s_pids)))
    ax.set_yticklabels([f"P{p}" for p in s_pids], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 1)
    ax.set_title(f'{label} Ratio per Participant (sorted)')
    ax.legend(fontsize=8)
    for i, (p, r) in enumerate(zip(s_pids, s_ratios)):
        ax.text(r + 0.015, i, f'{r:.0%}', va='center', fontsize=7, color='#333')

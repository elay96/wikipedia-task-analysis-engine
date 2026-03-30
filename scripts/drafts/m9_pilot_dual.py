#!/usr/bin/env python3
"""
M9 Pilot: Single bar — cluster colors (top 3/4) + switch indicator (bottom 1/4).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from helpers import load_trials, get_pids_and_trials, finish_timeline, OUTPUT_DIR

DATA_DIR = Path(__file__).parent / '..' / 'data'
SIM_PATH = DATA_DIR / 'similarity_matrix.json'

N_CLUSTERS = 10

CLUSTER_NAMES_ONE = [
    'Culture', 'Biodiversity', 'Brain/Cognition', 'Race',
    'Brain Bio.', 'Evolution', 'Evo. Mech.', 'Income',
    'Art Hist.', 'Art Theory',
]

CLUSTER_COLORS = [
    '#AED581', '#4FC3F7', '#CE93D8', '#EF5350',
    '#81D4FA', '#FFB74D', '#A1887F', '#F48FB1',
    '#FFD54F', '#FF8A65',
]

SWITCH_COLOR = '#222222'
NO_SWITCH_COLOR = '#E0E0E0'
FIRST_COLOR = '#CCCCCC'


def load_all():
    with open(SIM_PATH, 'r', encoding='utf-8') as f:
        sim_data = json.load(f)
    slugs = sim_data['slugs']
    n = len(slugs)
    slug_idx = {s: i for i, s in enumerate(slugs)}
    sims = sim_data['similarities']

    mat = np.zeros((n, n))
    for key, val in sims.items():
        a, b = key.split('|||')
        i, j = slug_idx[a], slug_idx[b]
        mat[i, j] = val
        mat[j, i] = val
    np.fill_diagonal(mat, 1.0)

    dist = np.sqrt(np.maximum(1.0 - mat, 0))
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    ev, evec = np.linalg.eigh(B)
    idx = np.argsort(ev)[::-1][:2]
    coords = evec[:, idx] * np.sqrt(np.maximum(ev[idx], 0))

    rng = np.random.RandomState(42)
    centers = coords[rng.choice(n, N_CLUSTERS, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(100):
        d = np.linalg.norm(coords[:, None] - centers[None, :], axis=2)
        nl = np.argmin(d, axis=1)
        if np.all(nl == labels):
            break
        labels = nl
        for j in range(N_CLUSTERS):
            m = labels == j
            if m.sum() > 0:
                centers[j] = coords[m].mean(axis=0)

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    for tr in trials:
        pvs = tr['page_visits']
        tr['cluster_seq'] = []
        tr['transitions'] = []
        for i, pv in enumerate(pvs):
            idx_a = slug_idx.get(pv['title'])
            cl = labels[idx_a] if idx_a is not None else -1
            tr['cluster_seq'].append(cl)
            if i == 0:
                tr['transitions'].append('first')
            else:
                prev_idx = slug_idx.get(pvs[i - 1]['title'])
                prev_cl = labels[prev_idx] if prev_idx is not None else -1
                tr['transitions'].append(
                    'exploit' if cl == prev_cl and cl >= 0 else 'explore')

    return slugs, slug_idx, labels, trials, pids, pid_trials


def main():
    print("[M9 Pilot] Single bar — cluster colors + switch strip")
    slugs, slug_idx, labels, trials, pids, pid_trials = load_all()
    n_pids = len(pids)

    fig, ax = plt.subplots(figsize=(22, 18))

    FULL_H = 0.32       # total bar height
    TOP_H = FULL_H * 0.75   # cluster color portion
    BOT_H = FULL_H * 0.25   # switch indicator portion

    for pi, pid in enumerate(pids):
        y_c = (n_pids - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y_mid = y_c + (0.20 if ti == 0 else -0.20)
            y_top = y_mid + BOT_H / 2   # top portion center
            y_bot = y_mid - TOP_H / 2   # bottom strip center

            # Background
            ax.barh(y_mid, tr['duration'], height=FULL_H, color='#FAFAFA',
                    edgecolor='#E8E8E8', linewidth=0.3, zorder=1)

            pvs = tr['page_visits']
            transitions = tr['transitions']
            cluster_seq = tr['cluster_seq']

            for vi, pv in enumerate(pvs):
                cl = cluster_seq[vi]
                bar_w = max(pv['duration'], 3)

                # Top 3/4: switch indicator (black/gray)
                if transitions[vi] == 'first':
                    sw_color = FIRST_COLOR
                elif transitions[vi] == 'explore':
                    sw_color = SWITCH_COLOR
                else:
                    sw_color = NO_SWITCH_COLOR

                ax.barh(y_mid + BOT_H / 2, bar_w, left=pv['start'],
                        height=TOP_H, color=sw_color, alpha=0.9,
                        edgecolor='none', zorder=3)

                # Bottom 1/4: cluster color
                cl_color = CLUSTER_COLORS[cl] if 0 <= cl < N_CLUSTERS else '#BDBDBD'
                ax.barh(y_mid - TOP_H / 2, bar_w, left=pv['start'],
                        height=BOT_H, color=cl_color, alpha=0.8,
                        edgecolor='none', zorder=3)

                # Page transition line
                ax.plot([pv['start'], pv['start']],
                        [y_mid - FULL_H / 2, y_mid + FULL_H / 2],
                        color='black', linewidth=0.5, zorder=4, alpha=0.4)

            # Right label
            n_page_trans = len(pvs) - 1
            n_switches = sum(1 for t in transitions if t == 'explore')
            ax.text(tr['duration'] + 8, y_mid,
                    f"{n_page_trans} pages, {n_switches} switches",
                    fontsize=5.5, va='center', color='#555')

    # Y-axis
    y_positions = [(n_pids - i - 1) for i in range(n_pids)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"P{pid}" for pid in pids], fontsize=9, fontweight='bold')
    ax.set_xlabel('Time within trial (seconds)', fontsize=11)
    ax.set_xlim(-10, None)
    for y in y_positions:
        ax.axhline(y=y, color='#E0E0E0', linewidth=0.3, zorder=0)

    ax.set_title(
        'M9: Semantic Similarity — Cluster-Based\n'
        'Top = dark (explore) / light (exploit) | '
        'Bottom strip = semantic cluster color',
        fontsize=13, fontweight='bold', pad=15)

    # Legend
    cl_handles = [mpatches.Patch(color=CLUSTER_COLORS[i], alpha=0.8,
                                  label=CLUSTER_NAMES_ONE[i])
                  for i in range(N_CLUSTERS)]
    switch_handles = [
        mpatches.Patch(color=SWITCH_COLOR, alpha=0.9, label='Cluster switch (explore)'),
        mpatches.Patch(color=NO_SWITCH_COLOR, alpha=0.9, label='Same cluster (exploit)'),
        mpatches.Patch(color=FIRST_COLOR, alpha=0.9, label='First page'),
        Line2D([0], [0], color='black', linewidth=0.8, alpha=0.5, label='Page transition'),
    ]

    leg1 = ax.legend(handles=switch_handles, fontsize=7, loc='lower left',
                     framealpha=0.95, title='Switch indicator', title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=cl_handles, fontsize=7, loc='lower right',
              framealpha=0.95, ncol=2, title='Semantic clusters', title_fontsize=8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm9_pilot_dual.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()

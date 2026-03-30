#!/usr/bin/env python3
"""
M8: Semantic Similarity — Cluster-Based
=========================================
Exploit = same knowledge cluster | Explore = different cluster
tf-idf cosine similarity → 10 clusters via k-means on MDS projection.
Output: m8_semantic.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
from pathlib import Path
from helpers import (load_trials, get_pids_and_trials, finish_timeline,
                     OUTPUT_DIR)

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

EXPLOIT_COLOR = '#4CAF50'
EXPLORE_COLOR = '#FF9800'


def load_all():
    with open(SIM_PATH, 'r', encoding='utf-8') as f:
        sim_data = json.load(f)
    slugs = sim_data['slugs']
    similarities = sim_data['similarities']

    n = len(slugs)
    slug_idx = {s: i for i, s in enumerate(slugs)}
    mat = np.zeros((n, n))
    for key, val in similarities.items():
        a, b = key.split('|||')
        i, j = slug_idx[a], slug_idx[b]
        mat[i, j] = val
        mat[j, i] = val
    np.fill_diagonal(mat, 1.0)

    # MDS
    dist = np.sqrt(np.maximum(1.0 - mat, 0))
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1][:2]
    coords = eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))

    # K-means
    rng = np.random.RandomState(42)
    centers = coords[rng.choice(n, N_CLUSTERS, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(100):
        dists_k = np.linalg.norm(coords[:, None] - centers[None, :], axis=2)
        new_labels = np.argmin(dists_k, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(N_CLUSTERS):
            m = labels == j
            if m.sum() > 0:
                centers[j] = coords[m].mean(axis=0)

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    # Transition matrix
    trans_matrix = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=int)
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
                if prev_cl >= 0 and cl >= 0:
                    trans_matrix[prev_cl, cl] += 1
                tr['transitions'].append(
                    'exploit' if cl == prev_cl and cl >= 0 else 'explore')

    return {
        'slugs': slugs, 'slug_idx': slug_idx, 'coords': coords,
        'cluster_labels': labels, 'trials': trials,
        'pids': pids, 'pid_trials': pid_trials,
        'trans_matrix': trans_matrix,
    }


def main():
    print("[M8] Semantic Similarity — Cluster-Based")
    data = load_all()
    pids = data['pids']
    pid_trials = data['pid_trials']
    slug_idx = data['slug_idx']
    labels = data['cluster_labels']
    coords = data['coords']
    tm = data['trans_matrix']
    trials = data['trials']
    n_pids = len(pids)

    print(f"  {len(trials)} trials, {n_pids} participants")
    print(f"  Exploit (diagonal): {int(np.trace(tm))}, "
          f"Explore: {int(tm.sum() - np.trace(tm))}")

    # ========== FIGURE ==========
    fig = plt.figure(figsize=(28, 32))

    # Layout: top = timeline (wide), bottom-left = heatmap, bottom-right = small multiples
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1.8, 1.6], hspace=0.25)

    # Top: timeline
    ax_timeline = fig.add_subplot(gs_main[0])

    # Bottom: heatmap (left) + small multiples grid (right)
    gs_bottom = gs_main[1].subgridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.25)
    ax_heatmap = fig.add_subplot(gs_bottom[0])

    # Small multiples: 4x5 grid inside the right bottom area
    gs_sm = gs_bottom[1].subgridspec(4, 5, hspace=0.3, wspace=0.15)

    fig.suptitle(
        'M8: Semantic Similarity — Cluster-Based\n'
        'Exploit = same knowledge cluster | Explore = different cluster',
        fontsize=16, fontweight='bold', y=0.985)

    # ===== TIMELINE =====
    for pi, pid in enumerate(pids):
        y_c = (n_pids - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y = y_c + (0.18 if ti == 0 else -0.18)

            ax_timeline.barh(y, tr['duration'], height=0.28, color='#F5F5F5',
                             edgecolor='#E0E0E0', linewidth=0.3, zorder=1)

            pvs = tr['page_visits']
            transitions = tr['transitions']
            cluster_seq = tr['cluster_seq']

            for vi, pv in enumerate(pvs):
                cl = cluster_seq[vi]
                color = CLUSTER_COLORS[cl] if 0 <= cl < N_CLUSTERS else '#BDBDBD'

                bar_w = max(pv['duration'], 3)
                ax_timeline.barh(y, bar_w, left=pv['start'], height=0.28,
                                 color=color, alpha=0.8, edgecolor='none',
                                 zorder=3)

                # Page transition marker (black line)
                ax_timeline.plot([pv['start'], pv['start']], [y - 0.14, y + 0.14],
                                 color='black', linewidth=0.6, zorder=4, alpha=0.6)

            # Right label: total page transitions + cluster switches
            n_page_trans = len(pvs) - 1  # total page-to-page transitions
            n_cluster_switches = sum(1 for t in transitions if t == 'explore')
            ax_timeline.text(tr['duration'] + 8, y,
                             f"{n_page_trans} pages, {n_cluster_switches} cluster switches",
                             fontsize=5.5, va='center', color='#555')

    finish_timeline(ax_timeline, pids)
    ax_timeline.set_title(
        'A. Browsing Timeline — Each Bar = Page Visit, Color = Semantic Cluster\n'
        'Same color = exploit (staying in topic) | Color change = explore (topic switch)',
        fontsize=12, fontweight='bold', pad=12)

    # Cluster legend for timeline
    cl_handles = [mpatches.Patch(color=CLUSTER_COLORS[i], alpha=0.8,
                                  label=CLUSTER_NAMES_ONE[i])
                  for i in range(N_CLUSTERS)]
    cl_handles.append(
        Line2D([0], [0], color='black', linewidth=1, alpha=0.6, label='Page transition'),
    )
    ax_timeline.legend(handles=cl_handles, fontsize=7, loc='lower right',
                       framealpha=0.95, ncol=4, fancybox=True)

    # ===== HEATMAP =====
    tm_float = tm.astype(float)
    im = ax_heatmap.imshow(tm_float, cmap='YlOrRd', aspect='equal',
                           interpolation='nearest')

    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            val = int(tm[i, j])
            if val == 0:
                continue
            color = 'white' if val > tm.max() * 0.55 else '#333'
            fw = 'bold' if i == j else 'normal'
            ax_heatmap.text(j, i, str(val), ha='center', va='center',
                            fontsize=9, fontweight=fw, color=color)

    # Highlight diagonal
    for i in range(N_CLUSTERS):
        ax_heatmap.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1, fill=False,
            edgecolor=EXPLOIT_COLOR, linewidth=2.5))

    ax_heatmap.set_xticks(range(N_CLUSTERS))
    ax_heatmap.set_yticks(range(N_CLUSTERS))
    ax_heatmap.set_xticklabels(CLUSTER_NAMES_ONE, fontsize=7.5, rotation=45, ha='right')
    ax_heatmap.set_yticklabels(CLUSTER_NAMES_ONE, fontsize=7.5)
    ax_heatmap.set_xlabel('To Cluster', fontsize=10, fontweight='bold')
    ax_heatmap.set_ylabel('From Cluster', fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.75, pad=0.02)
    cbar.set_label('Transitions', fontsize=9)

    total_exploit = int(np.trace(tm))
    total_explore = int(tm.sum() - total_exploit)
    ax_heatmap.set_title(
        f'B. Cluster Transition Matrix\n'
        f'Diagonal = Exploit ({total_exploit}) | '
        f'Off-diagonal = Explore ({total_explore})',
        fontsize=11, fontweight='bold', pad=10)

    # ===== SMALL MULTIPLES =====
    for pi, pid in enumerate(pids):
        row = pi // 5
        col = pi % 5
        ax = fig.add_subplot(gs_sm[row, col])

        # Background: faded cluster points
        for cl in range(N_CLUSTERS):
            mask = labels == cl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=CLUSTER_COLORS[cl], s=6, alpha=0.12, edgecolors='none')

        # Trajectories
        trial_colors_line = ['#1565C0', '#C62828']
        for ti, tr in enumerate(pid_trials[pid]):
            pvs = tr['page_visits']
            path_x, path_y, path_cl = [], [], []
            for pv in pvs:
                idx = slug_idx.get(pv['title'])
                if idx is not None:
                    path_x.append(coords[idx, 0])
                    path_y.append(coords[idx, 1])
                    path_cl.append(labels[idx])

            if len(path_x) < 2:
                continue

            tc = trial_colors_line[ti]
            ax.plot(path_x, path_y, '-', color=tc, linewidth=1.8,
                    alpha=0.5, zorder=3)

            # Start marker
            ax.plot(path_x[0], path_y[0], 'o', color=tc, markersize=5,
                    zorder=5, markeredgecolor='white', markeredgewidth=0.6)

            # Transition dots — exploit/explore
            for j in range(1, len(path_x)):
                mc = EXPLOIT_COLOR if path_cl[j] == path_cl[j - 1] else EXPLORE_COLOR
                ax.plot(path_x[j], path_y[j], 'o', color=mc,
                        markersize=4, zorder=4, alpha=0.85,
                        markeredgecolor='white', markeredgewidth=0.3)

        # Compute stats
        all_trans = []
        for tr in pid_trials[pid]:
            all_trans.extend(tr['transitions'])
        n_explore = sum(1 for t in all_trans if t == 'explore')
        n_exploit = sum(1 for t in all_trans if t == 'exploit')
        total = n_explore + n_exploit
        exploit_pct = n_exploit / total * 100 if total > 0 else 0

        ax.set_title(f'P{pid}  ({exploit_pct:.0f}% exploit)', fontsize=7.5,
                     fontweight='bold', pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#CCC')

    # Panel C title — add as text above the small multiples area
    fig.text(0.73, 0.47,
             'C. Per-Participant Trajectories in Semantic Space\n'
             'Blue = Trial 1, Red = Trial 2 | '
             'Green = Exploit, Orange = Explore',
             ha='center', fontsize=11, fontweight='bold')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm8_semantic.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()

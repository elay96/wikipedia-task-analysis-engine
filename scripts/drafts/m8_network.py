#!/usr/bin/env python3
"""
M8: Network-Based Explore/Exploit (Semantic Similarity)
========================================================
Based on Kedrick et al. (2026) — Creative foraging in knowledge networks.

Output: m8_network.png — 4-panel figure:
  A: Semantic map with cluster regions + transition flow (black lines)
  B: Why transitions aren't random (simplified comparison)
  E: Timeline per participant — transition points colored explore/exploit
  D: Exploit ratio per participant (cluster-based, sorted)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from pathlib import Path
from helpers import (load_trials, get_pids_and_trials, finish_timeline,
                     OUTPUT_DIR)

DATA_DIR = Path(__file__).parent / '..' / 'data'
SIM_PATH = DATA_DIR / 'cleaned' / 'similarity_matrix.json'

EXPLOIT_COLOR = '#4CAF50'
EXPLORE_COLOR = '#FF9800'
NEUTRAL_COLOR = '#BDBDBD'
N_CLUSTERS = 10

CLUSTER_NAMES = {
    0: 'Culture &\nGlobalization',
    1: 'Biodiversity &\nSpeciation',
    2: 'Brain &\nCognition',
    3: 'Race &\nDiscrimination',
    4: 'Brain Biology &\nEvolution Debates',
    5: 'Evolution\nCore',
    6: 'Evolutionary\nMechanisms',
    7: 'Income &\nInequality',
    8: 'Art\nHistory',
    9: 'Art\nTheory',
}

CLUSTER_COLORS = [
    '#AED581', '#4FC3F7', '#CE93D8', '#EF5350',
    '#81D4FA', '#FFB74D', '#A1887F', '#F48FB1',
    '#FFD54F', '#FF8A65',
]


def load_similarity():
    with open(SIM_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['slugs'], data['similarities'], data.get('stats', {})


def build_sim_matrix(slugs, similarities):
    n = len(slugs)
    slug_idx = {s: i for i, s in enumerate(slugs)}
    mat = np.zeros((n, n))
    for key, val in similarities.items():
        a, b = key.split('|||')
        i, j = slug_idx[a], slug_idx[b]
        mat[i, j] = val
        mat[j, i] = val
    np.fill_diagonal(mat, 1.0)
    return mat, slug_idx


def classical_mds(sim_matrix, n_components=2):
    dist = np.sqrt(np.maximum(1.0 - sim_matrix, 0))
    n = dist.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))


def kmeans_numpy(data, k, max_iter=100, seed=42):
    rng = np.random.RandomState(seed)
    centers = data[rng.choice(len(data), k, replace=False)].copy()
    labels = np.zeros(len(data), dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, None] - centers[None, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centers[j] = data[mask].mean(axis=0)
    return labels, centers


def convex_hull_points(points):
    """Graham scan. Returns hull vertex coordinates in order."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def expand_hull(hull_pts, factor=1.25):
    cx = np.mean([p[0] for p in hull_pts])
    cy = np.mean([p[1] for p in hull_pts])
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in hull_pts]


def get_sim_val(similarities, title_a, title_b):
    if title_a == title_b:
        return 1.0
    key1 = f"{title_a}|||{title_b}"
    key2 = f"{title_b}|||{title_a}"
    return similarities.get(key1, similarities.get(key2, 0.0))


def main():
    print("[M8] Network-based explore/exploit (semantic similarity)")
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    n_pids = len(pids)
    print(f"  {len(trials)} trials from {n_pids} participants")

    slugs, similarities, sim_stats = load_similarity()
    sim_matrix, slug_idx = build_sim_matrix(slugs, similarities)
    print(f"  {len(slugs)} articles, {sim_stats.get('n_pairs', '?')} pairs")

    print("  Computing MDS projection...")
    coords = classical_mds(sim_matrix, n_components=2)

    print(f"  Clustering into {N_CLUSTERS} clusters...")
    cluster_labels, cluster_centers = kmeans_numpy(coords, N_CLUSTERS)

    # Visit counts for node sizing
    visit_count = {}
    for tr in trials:
        for pv in tr['page_visits']:
            visit_count[pv['title']] = visit_count.get(pv['title'], 0) + 1

    node_sizes = np.array([visit_count.get(s, 0) for s in slugs])
    scaled_sizes = np.where(node_sizes > 0, 40 + node_sizes * 15, 15)
    scaled_sizes = np.minimum(scaled_sizes, 150)

    # All transitions
    all_pair_vals = np.array(list(similarities.values()))
    all_transitions = []  # (from_title, to_title, from_idx, to_idx)
    for tr in trials:
        pvs = tr['page_visits']
        for i in range(1, len(pvs)):
            fi = slug_idx.get(pvs[i - 1]['title'])
            ti = slug_idx.get(pvs[i]['title'])
            if fi is not None and ti is not None:
                all_transitions.append((fi, ti))

    trans_sims_all = np.array([
        get_sim_val(similarities, slugs[fi], slugs[ti])
        for fi, ti in all_transitions
    ])

    # Classify per trial
    pid_exploit_time = {p: [] for p in pids}
    pid_explore_time = {p: [] for p in pids}

    for tr in trials:
        pvs = tr['page_visits']
        tr['transitions'] = []
        ex_t, br_t = 0.0, 0.0
        for i, pv in enumerate(pvs):
            idx = slug_idx.get(pv['title'])
            cl = cluster_labels[idx] if idx is not None else -1
            if i == 0:
                tr['transitions'].append({'type': 'first', 'cluster': cl, 'sim': None})
            else:
                prev_idx = slug_idx.get(pvs[i - 1]['title'])
                prev_cl = cluster_labels[prev_idx] if prev_idx is not None else -1
                sim = get_sim_val(similarities, pvs[i - 1]['title'], pv['title'])
                if cl == prev_cl and cl >= 0:
                    tr['transitions'].append({'type': 'exploit', 'cluster': cl, 'sim': sim})
                    ex_t += pv['duration']
                else:
                    tr['transitions'].append({'type': 'explore', 'cluster': cl, 'sim': sim})
                    br_t += pv['duration']
        pid_exploit_time[tr['pid']].append(ex_t)
        pid_explore_time[tr['pid']].append(br_t)

    mean_all = float(all_pair_vals.mean())
    mean_trans = float(trans_sims_all.mean())
    ratio = mean_trans / mean_all

    print(f"  Transition mean sim: {mean_trans:.4f} ({ratio:.1f}x random)")

    # ========== FIGURE ==========
    fig = plt.figure(figsize=(24, 28))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 2.2, 1],
                          hspace=0.35, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, :])   # semantic map (full width)
    ax_e = fig.add_subplot(gs[1, :])   # timeline (full width)
    ax_b = fig.add_subplot(gs[2, 0])   # distribution comparison (bottom-left)
    ax_d = fig.add_subplot(gs[2, 1])   # exploit ratio (bottom-right)

    fig.suptitle(
        'M8: Semantic Network — Explore vs. Exploit\n'
        'tf-idf cosine similarity | Cluster-based classification '
        '(Kedrick et al., 2026)',
        fontsize=15, fontweight='bold', y=0.98)

    # ===== PANEL A: SEMANTIC MAP =====

    # Draw cluster hulls
    for cl in range(N_CLUSTERS):
        mask = cluster_labels == cl
        cl_coords = coords[mask]
        if len(cl_coords) < 3:
            cx, cy = cl_coords.mean(axis=0)
            circle = plt.Circle((cx, cy), 0.05, color=CLUSTER_COLORS[cl],
                                alpha=0.25, zorder=0)
            ax_a.add_patch(circle)
        else:
            pts = [(cl_coords[j, 0], cl_coords[j, 1]) for j in range(len(cl_coords))]
            hull = convex_hull_points(pts)
            hull_exp = expand_hull(hull, factor=1.3)
            poly = Polygon(hull_exp, closed=True, facecolor=CLUSTER_COLORS[cl],
                           alpha=0.25, edgecolor=CLUSTER_COLORS[cl],
                           linewidth=1.8, zorder=0)
            ax_a.add_patch(poly)

    # Draw ALL transition lines — semi-transparent black, overlaps = darker
    for fi, ti in all_transitions:
        ax_a.plot([coords[fi, 0], coords[ti, 0]],
                  [coords[fi, 1], coords[ti, 1]],
                  color='black', alpha=0.07, linewidth=1.2, zorder=1)

    # Draw article nodes
    for cl in range(N_CLUSTERS):
        mask = cluster_labels == cl
        visited = mask & (node_sizes > 0)
        unvisited = mask & (node_sizes == 0)
        if unvisited.any():
            ax_a.scatter(coords[unvisited, 0], coords[unvisited, 1],
                         c=CLUSTER_COLORS[cl], s=scaled_sizes[unvisited],
                         alpha=0.25, edgecolors='gray', linewidth=0.2, zorder=2)
        if visited.any():
            ax_a.scatter(coords[visited, 0], coords[visited, 1],
                         c=CLUSTER_COLORS[cl], s=scaled_sizes[visited],
                         alpha=0.9, edgecolors='white', linewidth=0.8, zorder=3)

    # Cluster legend at top of panel (instead of labels on the map)
    cluster_handles = []
    for cl in range(N_CLUSTERS):
        mask = cluster_labels == cl
        n_art = int(mask.sum())
        name = CLUSTER_NAMES.get(cl, f'Cluster {cl}').replace('\n', ' ')
        cluster_handles.append(
            mpatches.Patch(color=CLUSTER_COLORS[cl], alpha=0.7,
                           label=f'{name} ({n_art})')
        )

    ax_a.set_title('Semantic Map — Knowledge Clusters\n'
                    'Darker lines = more transitions between articles',
                    fontsize=10, fontweight='bold', pad=8)
    ax_a.set_xlabel('MDS Dimension 1', fontsize=9)
    ax_a.set_ylabel('MDS Dimension 2', fontsize=9)
    ax_a.grid(True, alpha=0.12, linestyle='-')

    # Pad axes
    xlim = ax_a.get_xlim()
    ylim = ax_a.get_ylim()
    xp = (xlim[1] - xlim[0]) * 0.1
    yp = (ylim[1] - ylim[0]) * 0.1
    ax_a.set_xlim(xlim[0] - xp, xlim[1] + xp)
    ax_a.set_ylim(ylim[0] - yp, ylim[1] + yp)

    # Cluster legend across the top
    cluster_handles.extend([
        Line2D([0], [0], color='black', alpha=0.3, linewidth=2,
               label='Transitions'),
        Line2D([0], [0], marker='o', color='gray', markersize=7, linestyle='',
               alpha=0.9, label='Visited'),
        Line2D([0], [0], marker='o', color='gray', markersize=4, linestyle='',
               alpha=0.25, label='Unvisited'),
    ])
    ax_a.legend(handles=cluster_handles, fontsize=7, loc='upper center',
                framealpha=0.95, ncol=5, columnspacing=1.0,
                bbox_to_anchor=(0.5, 1.0), fancybox=True)

    # ===== PANEL B: WHY TRANSITIONS AREN'T RANDOM =====

    # Simple bar comparison with clear explanation
    bar_x = [0, 1]
    bar_vals = [mean_all, mean_trans]
    bar_colors_b = ['#BDBDBD', '#42A5F5']
    bars = ax_b.bar(bar_x, bar_vals, color=bar_colors_b, edgecolor='gray',
                    width=0.5, linewidth=1.2)

    ax_b.set_xticks(bar_x)
    ax_b.set_xticklabels([
        'Random\n(any two articles)',
        'Actual navigation\n(consecutive pages)'
    ], fontsize=10, fontweight='bold')
    ax_b.set_ylabel('Mean Cosine Similarity', fontsize=10)
    ax_b.set_title('Participants Navigate Semantically,\nNot Randomly',
                    fontsize=11, fontweight='bold', pad=10)

    # Value labels on bars
    ax_b.text(0, mean_all + mean_trans * 0.02, f'{mean_all:.3f}',
              ha='center', fontsize=11, fontweight='bold', color='#555')
    ax_b.text(1, mean_trans + mean_trans * 0.02, f'{mean_trans:.3f}',
              ha='center', fontsize=11, fontweight='bold', color='#1565C0')

    # Big "x5.9" annotation with arrow
    ax_b.annotate(
        f'×{ratio:.1f}',
        xy=(1, mean_trans * 0.85), xytext=(0.5, mean_trans * 0.65),
        fontsize=28, fontweight='bold', color='#C62828',
        ha='center', va='center',
        arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                  alpha=0.95, edgecolor='#C62828', linewidth=2))

    # Explanation text
    ax_b.text(0.5, -0.18,
              'When browsing Wikipedia, participants chose pages\n'
              'that are 5.9× more semantically similar than random.\n'
              'This confirms structured, topic-driven navigation.',
              transform=ax_b.transAxes, fontsize=8.5, ha='center', va='top',
              style='italic', color='#444',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                        alpha=0.8, edgecolor='#DDD'))

    ax_b.set_ylim(0, mean_trans * 1.25)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # ===== PANEL E: TIMELINE — TRANSITION POINTS =====

    for pi, pid in enumerate(pids):
        y_c = (n_pids - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y = y_c + (0.18 if ti == 0 else -0.18)

            # Background bar (trial duration)
            ax_e.barh(y, tr['duration'], height=0.28, color='#F5F5F5',
                      edgecolor='#E0E0E0', linewidth=0.3, zorder=1)

            # Draw page visit bars in neutral light gray
            for pv in tr['page_visits']:
                ax_e.barh(y, pv['duration'], left=pv['start'], height=0.28,
                          color='#EEEEEE', edgecolor='#E0E0E0', linewidth=0.3,
                          zorder=1)

            # Draw transition POINTS at the moment of each page switch
            pvs = tr['page_visits']
            transitions = tr['transitions']
            exploit_count, explore_count = 0, 0

            for vi in range(len(pvs)):
                trans = transitions[vi]
                t_time = pvs[vi]['start']  # moment of transition

                if trans['type'] == 'first':
                    # First page — small gray diamond
                    ax_e.plot(t_time, y, 'd', color=NEUTRAL_COLOR,
                              markersize=5, zorder=5, markeredgecolor='gray',
                              markeredgewidth=0.5)
                elif trans['type'] == 'exploit':
                    ax_e.plot(t_time, y, 'o', color=EXPLOIT_COLOR,
                              markersize=7, zorder=5, markeredgecolor='#2E7D32',
                              markeredgewidth=0.8, alpha=0.85)
                    exploit_count += 1
                else:  # explore
                    ax_e.plot(t_time, y, 's', color=EXPLORE_COLOR,
                              markersize=7, zorder=5, markeredgecolor='#E65100',
                              markeredgewidth=0.8, alpha=0.85)
                    explore_count += 1

            # Connect transitions with thin line
            trans_times = [pvs[vi]['start'] for vi in range(len(pvs))]
            if len(trans_times) > 1:
                ax_e.plot(trans_times, [y] * len(trans_times),
                          color='#999', linewidth=0.5, alpha=0.4, zorder=2)

            # Right label
            total = exploit_count + explore_count
            pct = exploit_count / total * 100 if total > 0 else 0
            ax_e.text(tr['duration'] + 8, y,
                      f"{tr['domain'][:8]} ({pct:.0f}% exploit)",
                      fontsize=5.5, va='center', color='#616161')

    finish_timeline(ax_e, pids)
    ax_e.set_title(
        'Browsing Timeline — Each Point = Page Transition\n'
        'Green ● = Exploit (same cluster) | Orange ■ = Explore (new cluster) | '
        'Gray ◆ = First page',
        fontsize=11, fontweight='bold', pad=10)

    ax_e.legend(handles=[
        Line2D([0], [0], marker='o', color=EXPLOIT_COLOR, markersize=8,
               linestyle='', markeredgecolor='#2E7D32',
               label='Exploit (same cluster)'),
        Line2D([0], [0], marker='s', color=EXPLORE_COLOR, markersize=8,
               linestyle='', markeredgecolor='#E65100',
               label='Explore (new cluster)'),
        Line2D([0], [0], marker='d', color=NEUTRAL_COLOR, markersize=6,
               linestyle='', markeredgecolor='gray',
               label='First page (no transition)'),
    ], fontsize=8, loc='lower right', framealpha=0.9, ncol=3)

    # ===== PANEL D: EXPLOIT RATIO (CLUSTER-BASED, SORTED) =====
    ex_m = [np.mean(pid_exploit_time[p]) if pid_exploit_time[p] else 0 for p in pids]
    br_m = [np.mean(pid_explore_time[p]) if pid_explore_time[p] else 0 for p in pids]
    ratios = [ex_m[i] / (ex_m[i] + br_m[i]) if (ex_m[i] + br_m[i]) > 0 else 0.5
              for i in range(n_pids)]

    idx_sort = np.argsort(ratios)
    s_pids = [pids[i] for i in idx_sort]
    s_ratios = [ratios[i] for i in idx_sort]

    colors = [EXPLOIT_COLOR if r > 0.5 else EXPLORE_COLOR for r in s_ratios]
    ax_d.barh(range(len(s_pids)), s_ratios, color=colors, edgecolor='gray',
              height=0.6, linewidth=0.5)
    ax_d.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    mean_r = np.mean(ratios)
    ax_d.axvline(x=mean_r, color='red', linestyle='--', linewidth=1.5)

    ax_d.set_yticks(range(len(s_pids)))
    ax_d.set_yticklabels([f"P{p}" for p in s_pids], fontsize=7)
    ax_d.set_xlabel('Exploit Ratio (% time in same semantic cluster)', fontsize=10)
    ax_d.set_xlim(0, 1)
    ax_d.set_title('Exploit Ratio per Participant — Cluster-Based (sorted)',
                    fontsize=11, fontweight='bold')

    ax_d.legend(handles=[
        mpatches.Patch(color=EXPLOIT_COLOR, label='Exploit (same cluster)'),
        mpatches.Patch(color=EXPLORE_COLOR, label='Explore (cross-cluster)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_r:.0%}'),
    ], fontsize=8, loc='lower right', framealpha=0.9)

    for i, (p, r) in enumerate(zip(s_pids, s_ratios)):
        ax_d.text(r + 0.012, i, f'{r:.0%}', va='center', fontsize=6.5, color='#333')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm8_network.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

    print(f"\n  === Summary ===")
    print(f"  Articles: {len(slugs)}, Clusters: {N_CLUSTERS}")
    print(f"  Transitions: {len(all_transitions)}")
    print(f"  All-pairs mean sim: {mean_all:.4f}")
    print(f"  Transition mean sim: {mean_trans:.4f} ({ratio:.1f}x random)")
    print(f"  Mean exploit ratio (cluster): {mean_r:.2%}")


if __name__ == '__main__':
    main()

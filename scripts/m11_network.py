#!/usr/bin/env python3
"""
M11: Semantic Network — The Actual Knowledge Graph
====================================================
134 articles as nodes, colored by cluster.
Edges = participant transitions between articles.
Node size = visit frequency.
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
from helpers import load_trials, OUTPUT_DIR

DATA_DIR = Path(__file__).parent / '..' / 'data'
SIM_PATH = DATA_DIR / 'similarity_matrix.json'

N_CLUSTERS = 10

CLUSTER_NAMES = [
    'Culture & Globalization', 'Biodiversity & Speciation',
    'Brain & Cognition', 'Race & Discrimination',
    'Brain Biology', 'Evolution Core',
    'Evolutionary Mechanisms', 'Income & Inequality',
    'Art History', 'Art Theory',
]

CLUSTER_COLORS = [
    '#AED581', '#4FC3F7', '#CE93D8', '#EF5350',
    '#81D4FA', '#FFB74D', '#A1887F', '#F48FB1',
    '#FFD54F', '#FF8A65',
]


def convex_hull_points(points):
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


def main():
    # Load similarity matrix
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

    # MDS projection
    dist = np.sqrt(np.maximum(1.0 - mat, 0))
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1][:2]
    coords = eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))

    # K-means clustering
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

    # Load trials for transitions and visit counts
    trials = load_trials()

    visit_count = {}
    for tr in trials:
        for pv in tr['page_visits']:
            visit_count[pv['title']] = visit_count.get(pv['title'], 0) + 1

    node_sizes = np.array([visit_count.get(s, 0) for s in slugs])
    scaled_sizes = np.where(node_sizes > 0, 50 + node_sizes * 18, 18)
    scaled_sizes = np.minimum(scaled_sizes, 180)

    # Collect transitions
    edge_counts = {}
    for tr in trials:
        pvs = tr['page_visits']
        for i in range(1, len(pvs)):
            fi = slug_idx.get(pvs[i - 1]['title'])
            ti_idx = slug_idx.get(pvs[i]['title'])
            if fi is not None and ti_idx is not None:
                key = (min(fi, ti_idx), max(fi, ti_idx))
                edge_counts[key] = edge_counts.get(key, 0) + 1

    # ========== FIGURE ==========
    fig, ax = plt.subplots(figsize=(18, 14))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    # Draw cluster hulls
    for cl in range(N_CLUSTERS):
        mask = labels == cl
        cl_coords = coords[mask]
        if len(cl_coords) < 3:
            cx, cy = cl_coords.mean(axis=0)
            circle = plt.Circle((cx, cy), 0.05, color=CLUSTER_COLORS[cl],
                                alpha=0.18, zorder=0)
            ax.add_patch(circle)
        else:
            pts = [(cl_coords[j, 0], cl_coords[j, 1]) for j in range(len(cl_coords))]
            hull = convex_hull_points(pts)
            hull_exp = expand_hull(hull, factor=1.3)
            poly = Polygon(hull_exp, closed=True, facecolor=CLUSTER_COLORS[cl],
                           alpha=0.18, edgecolor=CLUSTER_COLORS[cl],
                           linewidth=1.5, linestyle='--', zorder=0)
            ax.add_patch(poly)

        # Cluster label at centroid
        cx, cy = cl_coords.mean(axis=0)
        ax.text(cx, cy, CLUSTER_NAMES[cl], fontsize=7, ha='center', va='center',
                fontweight='bold', color='#333', alpha=0.6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5,
                          edgecolor='none'))

    # Draw transition edges
    max_count = max(edge_counts.values()) if edge_counts else 1
    for (fi, ti_idx), count in edge_counts.items():
        alpha = 0.08 + 0.5 * (count / max_count)
        lw = 0.5 + 2.5 * (count / max_count)
        ax.plot([coords[fi, 0], coords[ti_idx, 0]],
                [coords[fi, 1], coords[ti_idx, 1]],
                color='#333', alpha=alpha, linewidth=lw, zorder=1)

    # Draw nodes
    for cl in range(N_CLUSTERS):
        mask = labels == cl
        visited = mask & (node_sizes > 0)
        unvisited = mask & (node_sizes == 0)
        if unvisited.any():
            ax.scatter(coords[unvisited, 0], coords[unvisited, 1],
                       c=CLUSTER_COLORS[cl], s=scaled_sizes[unvisited],
                       alpha=0.25, edgecolors='gray', linewidth=0.3, zorder=2)
        if visited.any():
            ax.scatter(coords[visited, 0], coords[visited, 1],
                       c=CLUSTER_COLORS[cl], s=scaled_sizes[visited],
                       alpha=0.9, edgecolors='white', linewidth=0.8, zorder=3)

    # Label most-visited nodes
    top_indices = np.argsort(node_sizes)[-15:]
    for idx_node in top_indices:
        if node_sizes[idx_node] > 0:
            label = slugs[idx_node].replace('_', ' ')
            if len(label) > 25:
                label = label[:22] + '...'
            ax.annotate(label, (coords[idx_node, 0], coords[idx_node, 1]),
                        fontsize=5.5, fontweight='bold', color='#222',
                        textcoords='offset points', xytext=(6, 6),
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  alpha=0.7, edgecolor='none'))

    # Title and axes
    ax.set_title(
        f'M11: Semantic Knowledge Network\n'
        f'{n} articles · {N_CLUSTERS} clusters · {len(edge_counts)} unique edges · '
        f'{sum(edge_counts.values())} total transitions',
        fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('MDS Dimension 1', fontsize=10, color='#555')
    ax.set_ylabel('MDS Dimension 2', fontsize=10, color='#555')
    ax.grid(True, alpha=0.1, linestyle='-')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xp = (xlim[1] - xlim[0]) * 0.08
    yp = (ylim[1] - ylim[0]) * 0.08
    ax.set_xlim(xlim[0] - xp, xlim[1] + xp)
    ax.set_ylim(ylim[0] - yp, ylim[1] + yp)

    for spine in ax.spines.values():
        spine.set_color('#ccc')

    # Legend
    handles = [mpatches.Patch(color=CLUSTER_COLORS[i], alpha=0.7,
                               label=CLUSTER_NAMES[i])
               for i in range(N_CLUSTERS)]
    handles.extend([
        Line2D([0], [0], marker='o', color='gray', markersize=9, linestyle='',
               alpha=0.9, markeredgecolor='white', label='Visited article'),
        Line2D([0], [0], marker='o', color='gray', markersize=5, linestyle='',
               alpha=0.25, label='Unvisited article'),
        Line2D([0], [0], color='#333', linewidth=2, alpha=0.4,
               label='Navigation edge'),
    ])
    ax.legend(handles=handles, fontsize=7.5, loc='upper left',
              framealpha=0.9, ncol=2, fancybox=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm11_network.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()

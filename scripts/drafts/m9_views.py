#!/usr/bin/env python3
"""
M9: Six Alternative Visualizations for Semantic Explore/Exploit
================================================================
Generates 6 separate figures, each showing the same data differently:
  1. Chord diagram (cluster-to-cluster transitions)
  2. Heatmap (transition matrix)
  3. Small multiples (per-participant trajectories)
  4. Timeline with cluster color bands
  5. Sankey-style flow
  6. Compact network (clusters as nodes)

Output: m9_1_chord.png ... m9_6_compact_network.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrowPatch, Arc, Wedge
from matplotlib.path import Path as MPath
import matplotlib.patheffects as pe
from pathlib import Path
from collections import Counter
from helpers import (load_trials, get_pids_and_trials, finish_timeline,
                     OUTPUT_DIR)

DATA_DIR = Path(__file__).parent / '..' / 'data'
SIM_PATH = DATA_DIR / 'cleaned' / 'similarity_matrix.json'

EXPLOIT_COLOR = '#4CAF50'
EXPLORE_COLOR = '#FF9800'
N_CLUSTERS = 10

CLUSTER_NAMES_SHORT = [
    'Culture &\nGlobalization',
    'Biodiversity &\nSpeciation',
    'Brain &\nCognition',
    'Race &\nDiscrimination',
    'Brain Biology &\nEvo. Debates',
    'Evolution\nCore',
    'Evolutionary\nMechanisms',
    'Income &\nInequality',
    'Art\nHistory',
    'Art\nTheory',
]

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


# ========== DATA LOADING ==========

def load_all():
    """Load and precompute everything needed for all visualizations."""
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
        dists = np.linalg.norm(coords[:, None] - centers[None, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(N_CLUSTERS):
            m = labels == j
            if m.sum() > 0:
                centers[j] = coords[m].mean(axis=0)

    # Trials
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    # Visit counts
    visit_count = {}
    for tr in trials:
        for pv in tr['page_visits']:
            visit_count[pv['title']] = visit_count.get(pv['title'], 0) + 1

    # Classify transitions
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
                tr['transitions'].append('exploit' if cl == prev_cl and cl >= 0 else 'explore')

    return {
        'slugs': slugs, 'slug_idx': slug_idx, 'coords': coords,
        'cluster_labels': labels, 'cluster_centers': centers,
        'trials': trials, 'pids': pids, 'pid_trials': pid_trials,
        'visit_count': visit_count, 'trans_matrix': trans_matrix,
    }


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


# ========== 1. CHORD DIAGRAM ==========

def draw_chord(data):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')
    ax.axis('off')

    tm = data['trans_matrix']
    total_per_cluster = tm.sum(axis=0) + tm.sum(axis=1)
    # Avoid division by zero
    total_per_cluster = np.maximum(total_per_cluster, 1)

    n = N_CLUSTERS
    gap = 2  # degrees gap between clusters
    total_deg = 360 - n * gap
    cluster_deg = total_deg * (total_per_cluster / total_per_cluster.sum())

    # Compute start/end angles for each cluster arc
    angles = []
    current = 90  # start at top
    for i in range(n):
        start = current
        end = current + cluster_deg[i]
        angles.append((start, end))
        current = end + gap

    R = 1.0  # radius

    # Draw outer arcs
    for i in range(n):
        start, end = angles[i]
        theta = np.linspace(np.radians(start), np.radians(end), 50)
        x_out = R * np.cos(theta)
        y_out = R * np.sin(theta)
        x_in = 0.92 * R * np.cos(theta)
        y_in = 0.92 * R * np.sin(theta)

        verts = list(zip(x_out, y_out)) + list(zip(x_in[::-1], y_in[::-1]))
        verts.append(verts[0])
        poly = Polygon(verts, closed=True, facecolor=CLUSTER_COLORS[i],
                        edgecolor='white', linewidth=1, alpha=0.85)
        ax.add_patch(poly)

        # Label
        mid_angle = np.radians((start + end) / 2)
        lx = 1.12 * R * np.cos(mid_angle)
        ly = 1.12 * R * np.sin(mid_angle)
        rotation = np.degrees(mid_angle)
        if 90 < rotation < 270:
            rotation += 180
        ax.text(lx, ly, CLUSTER_NAMES_ONE[i], fontsize=8, fontweight='bold',
                ha='center', va='center', rotation=rotation - 90,
                color='#333')

    # Draw chords (bezier curves between cluster arcs)
    max_val = tm.max()
    for i in range(n):
        for j in range(n):
            val = tm[i, j]
            if val == 0:
                continue

            # Source point: midpoint of cluster i arc
            si, ei = angles[i]
            mid_i = np.radians((si + ei) / 2)

            # Target point: midpoint of cluster j arc
            sj, ej = angles[j]
            mid_j = np.radians((sj + ej) / 2)

            if i == j:
                # Self-loop: draw a small bump
                a1 = np.radians(si + (ei - si) * 0.3)
                a2 = np.radians(si + (ei - si) * 0.7)
                x1 = 0.92 * R * np.cos(a1)
                y1 = 0.92 * R * np.sin(a1)
                x2 = 0.92 * R * np.cos(a2)
                y2 = 0.92 * R * np.sin(a2)
                cx = 0.6 * R * np.cos(mid_i)
                cy = 0.6 * R * np.sin(mid_i)
            else:
                x1 = 0.92 * R * np.cos(mid_i)
                y1 = 0.92 * R * np.sin(mid_i)
                x2 = 0.92 * R * np.cos(mid_j)
                y2 = 0.92 * R * np.sin(mid_j)
                cx, cy = 0, 0  # control point at center

            verts = [(x1, y1), (cx, cy), (x2, y2)]
            codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
            path = MPath(verts, codes)

            lw = 1 + (val / max(max_val, 1)) * 8
            alpha = 0.3 + 0.5 * (val / max(max_val, 1))
            patch = mpatches.FancyArrowPatch(
                path=path, color=CLUSTER_COLORS[i],
                linewidth=lw, alpha=alpha,
                arrowstyle='-', mutation_scale=1)
            ax.add_patch(patch)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('Cluster-to-Cluster Transition Flow\n'
                 'Thicker chords = more transitions | Self-loops = exploit',
                 fontsize=13, fontweight='bold', pad=20)

    # Legend with counts
    total_exploit = int(np.trace(tm))
    total_explore = int(tm.sum() - total_exploit)
    ax.text(0, -1.38, f'Total: {int(tm.sum())} transitions | '
            f'Exploit (within): {total_exploit} | Explore (between): {total_explore}',
            ha='center', fontsize=10, color='#555')

    outpath = OUTPUT_DIR / 'm9_1_chord.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ========== 2. HEATMAP ==========

def draw_heatmap(data):
    fig, ax = plt.subplots(figsize=(12, 10))

    tm = data['trans_matrix'].astype(float)
    im = ax.imshow(tm, cmap='YlOrRd', aspect='equal', interpolation='nearest')

    # Annotate cells
    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            val = int(tm[i, j])
            if val == 0:
                continue
            color = 'white' if val > tm.max() * 0.6 else '#333'
            fontweight = 'bold' if i == j else 'normal'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=10, fontweight=fontweight, color=color)

    ax.set_xticks(range(N_CLUSTERS))
    ax.set_yticks(range(N_CLUSTERS))
    ax.set_xticklabels(CLUSTER_NAMES_ONE, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(CLUSTER_NAMES_ONE, fontsize=8)
    ax.set_xlabel('To Cluster', fontsize=11, fontweight='bold')
    ax.set_ylabel('From Cluster', fontsize=11, fontweight='bold')

    # Highlight diagonal
    for i in range(N_CLUSTERS):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor=EXPLOIT_COLOR, linewidth=2.5))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Transitions', fontsize=10)

    total_exploit = int(np.trace(tm))
    total_explore = int(tm.sum() - total_exploit)
    ax.set_title(
        'Cluster Transition Matrix\n'
        f'Diagonal (green border) = Exploit ({total_exploit}) | '
        f'Off-diagonal = Explore ({total_explore})',
        fontsize=13, fontweight='bold', pad=15)

    outpath = OUTPUT_DIR / 'm9_2_heatmap.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ========== 3. SMALL MULTIPLES ==========

def draw_small_multiples(data):
    coords = data['coords']
    labels = data['cluster_labels']
    slug_idx = data['slug_idx']
    pids = data['pids']
    pid_trials = data['pid_trials']

    n_pids = len(pids)
    ncols = 5
    nrows = (n_pids + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 4))
    axes = axes.flatten()

    for pi, pid in enumerate(pids):
        ax = axes[pi]

        # Background: faded cluster points
        for cl in range(N_CLUSTERS):
            mask = labels == cl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=CLUSTER_COLORS[cl], s=8, alpha=0.15, edgecolors='none')

        # Draw trajectories for both trials
        trial_colors = ['#1565C0', '#C62828']
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

            # Draw path
            tc = trial_colors[ti]
            ax.plot(path_x, path_y, '-', color=tc, linewidth=1.5, alpha=0.6, zorder=3)

            # Start and end markers
            ax.plot(path_x[0], path_y[0], 'o', color=tc, markersize=6,
                    zorder=5, markeredgecolor='white', markeredgewidth=0.8)
            ax.plot(path_x[-1], path_y[-1], 's', color=tc, markersize=5,
                    zorder=5, markeredgecolor='white', markeredgewidth=0.8)

            # Color transition points
            for j in range(1, len(path_x)):
                marker_color = EXPLOIT_COLOR if path_cl[j] == path_cl[j - 1] else EXPLORE_COLOR
                ax.plot(path_x[j], path_y[j], 'o', color=marker_color,
                        markersize=4, zorder=4, alpha=0.8)

        ax.set_title(f'P{pid}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#CCC')

    # Hide unused axes
    for i in range(n_pids, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Per-Participant Trajectories in Semantic Space\n'
                 'Blue = Trial 1, Red = Trial 2 | Green dots = Exploit, Orange = Explore',
                 fontsize=14, fontweight='bold', y=1.01)

    fig.legend(handles=[
        Line2D([0], [0], color='#1565C0', linewidth=2, label='Trial 1'),
        Line2D([0], [0], color='#C62828', linewidth=2, label='Trial 2'),
        Line2D([0], [0], marker='o', color=EXPLOIT_COLOR, markersize=6,
               linestyle='', label='Exploit'),
        Line2D([0], [0], marker='o', color=EXPLORE_COLOR, markersize=6,
               linestyle='', label='Explore'),
    ], fontsize=9, loc='lower center', ncol=4, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02))

    outpath = OUTPUT_DIR / 'm9_3_small_multiples.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ========== 4. TIMELINE WITH CLUSTER COLOR BANDS ==========

def draw_timeline_bands(data):
    pids = data['pids']
    pid_trials = data['pid_trials']
    labels = data['cluster_labels']
    slug_idx = data['slug_idx']
    n_pids = len(pids)

    fig, ax = plt.subplots(figsize=(22, 16))

    for pi, pid in enumerate(pids):
        y_c = (n_pids - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y = y_c + (0.18 if ti == 0 else -0.18)

            # Background
            ax.barh(y, tr['duration'], height=0.28, color='#F5F5F5',
                    edgecolor='#E0E0E0', linewidth=0.3, zorder=1)

            pvs = tr['page_visits']
            transitions = tr['transitions']

            for vi, pv in enumerate(pvs):
                idx = slug_idx.get(pv['title'])
                cl = labels[idx] if idx is not None else -1

                if cl >= 0:
                    color = CLUSTER_COLORS[cl]
                else:
                    color = '#BDBDBD'

                # Border color based on transition type
                if transitions[vi] == 'exploit':
                    ec = '#2E7D32'
                    lw = 0.8
                elif transitions[vi] == 'explore':
                    ec = '#E65100'
                    lw = 0.8
                else:
                    ec = '#999'
                    lw = 0.4

                bar_w = max(pv['duration'], 3)
                ax.barh(y, bar_w, left=pv['start'], height=0.28,
                        color=color, alpha=0.75, edgecolor=ec,
                        linewidth=lw, zorder=3)

                # Page transition marker
                ax.plot([pv['start'], pv['start']], [y - 0.14, y + 0.14],
                        color='black', linewidth=0.4, zorder=4, alpha=0.4)

            # Right label: domain + clusters visited
            cluster_seq = [labels[slug_idx[pv['title']]]
                           for pv in pvs if pv['title'] in slug_idx]
            n_clusters = len(set(cluster_seq))
            ax.text(tr['duration'] + 8, y,
                    f"{tr['domain'][:8]} ({n_clusters} clusters)",
                    fontsize=5.5, va='center', color='#616161')

    finish_timeline(ax, pids)
    ax.set_title('Browsing Timeline — Pages Colored by Semantic Cluster\n'
                 'Same color = same topic area | Color change = topic switch',
                 fontsize=13, fontweight='bold', pad=15)

    # Cluster legend
    handles = [mpatches.Patch(color=CLUSTER_COLORS[i], alpha=0.75,
                               label=CLUSTER_NAMES_ONE[i])
               for i in range(N_CLUSTERS)]
    handles.extend([
        Line2D([0], [0], color='black', linewidth=0.5, label='Page transition'),
    ])
    ax.legend(handles=handles, fontsize=7, loc='lower right',
              framealpha=0.95, ncol=4)

    outpath = OUTPUT_DIR / 'm9_4_timeline_bands.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ========== 5. SANKEY-STYLE FLOW ==========

def draw_sankey(data):
    """Cluster flow over step number — aggregated across all participants."""
    trials = data['trials']

    # Find max steps
    max_steps = max(len(tr['cluster_seq']) for tr in trials)
    max_steps = min(max_steps, 15)  # cap for readability

    # Count participants in each cluster at each step
    step_counts = np.zeros((max_steps, N_CLUSTERS))
    flow_counts = np.zeros((max_steps - 1, N_CLUSTERS, N_CLUSTERS))

    for tr in trials:
        seq = tr['cluster_seq']
        for s in range(min(len(seq), max_steps)):
            cl = seq[s]
            if 0 <= cl < N_CLUSTERS:
                step_counts[s, cl] += 1
            if s > 0:
                prev_cl = seq[s - 1]
                if 0 <= prev_cl < N_CLUSTERS and 0 <= cl < N_CLUSTERS:
                    flow_counts[s - 1, prev_cl, cl] += 1

    fig, ax = plt.subplots(figsize=(20, 10))

    col_width = 0.3
    x_positions = np.arange(max_steps) * 1.5

    # Draw stacked bars at each step
    for s in range(max_steps):
        bottom = 0
        total = step_counts[s].sum()
        if total == 0:
            continue
        for cl in range(N_CLUSTERS):
            h = step_counts[s, cl]
            if h == 0:
                continue
            ax.bar(x_positions[s], h, bottom=bottom, width=col_width,
                   color=CLUSTER_COLORS[cl], edgecolor='white', linewidth=0.5,
                   alpha=0.85)
            if h >= 2:
                ax.text(x_positions[s], bottom + h / 2, f'{int(h)}',
                        ha='center', va='center', fontsize=6, color='#333')
            bottom += h

    # Draw flow curves between steps
    for s in range(max_steps - 1):
        # Compute y positions for source and target
        src_bottom = np.zeros(N_CLUSTERS)
        dst_bottom = np.zeros(N_CLUSTERS)
        cum_src = np.cumsum(np.concatenate([[0], step_counts[s, :]]))
        cum_dst = np.cumsum(np.concatenate([[0], step_counts[s + 1, :]]))

        src_offsets = np.zeros(N_CLUSTERS)
        dst_offsets = np.zeros(N_CLUSTERS)

        for cl_from in range(N_CLUSTERS):
            for cl_to in range(N_CLUSTERS):
                count = flow_counts[s, cl_from, cl_to]
                if count == 0:
                    continue

                # Source y range
                y_src_bottom = cum_src[cl_from] + src_offsets[cl_from]
                y_src_top = y_src_bottom + count
                src_offsets[cl_from] += count

                # Dest y range
                y_dst_bottom = cum_dst[cl_to] + dst_offsets[cl_to]
                y_dst_top = y_dst_bottom + count
                dst_offsets[cl_to] += count

                # Draw bezier ribbon
                x0 = x_positions[s] + col_width / 2
                x1 = x_positions[s + 1] - col_width / 2
                xm = (x0 + x1) / 2

                n_pts = 30
                t = np.linspace(0, 1, n_pts)
                # Cubic bezier x
                bx = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * xm + 3 * (1 - t) * t ** 2 * xm + t ** 3 * x1
                # Bottom edge
                by_bot = (1 - t) * y_src_bottom + t * y_dst_bottom
                # Top edge
                by_top = (1 - t) * y_src_top + t * y_dst_top

                color = CLUSTER_COLORS[cl_from]
                alpha = 0.4 if cl_from != cl_to else 0.6
                ax.fill_between(bx, by_bot, by_top, color=color, alpha=alpha,
                                edgecolor='none')

    ax.set_xticks(x_positions[:max_steps])
    ax.set_xticklabels([f'Page {i + 1}' for i in range(max_steps)], fontsize=9)
    ax.set_ylabel('Number of Participants', fontsize=11)
    ax.set_title('Cluster Flow Across Browsing Steps\n'
                 'Each column = page visit step | Flows show cluster-to-cluster transitions',
                 fontsize=13, fontweight='bold', pad=15)

    handles = [mpatches.Patch(color=CLUSTER_COLORS[i], alpha=0.75,
                               label=CLUSTER_NAMES_ONE[i])
               for i in range(N_CLUSTERS)]
    ax.legend(handles=handles, fontsize=7, loc='upper right',
              framealpha=0.95, ncol=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    outpath = OUTPUT_DIR / 'm9_5_sankey.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ========== 6. COMPACT NETWORK ==========

def draw_compact_network(data):
    coords = data['coords']
    labels = data['cluster_labels']
    tm = data['trans_matrix']
    visit_count = data['visit_count']
    slugs = data['slugs']

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')

    # Cluster positions = centroids in MDS space
    cl_pos = np.zeros((N_CLUSTERS, 2))
    for cl in range(N_CLUSTERS):
        mask = labels == cl
        cl_pos[cl] = coords[mask].mean(axis=0)

    # Visited articles per cluster
    cl_visits = np.zeros(N_CLUSTERS)
    for slug, count in visit_count.items():
        from helpers import load_trials  # slug_idx already available
        idx = data['slug_idx'].get(slug)
        if idx is not None:
            cl_visits[labels[idx]] += count

    # Node sizes
    min_size = 800
    max_size = 5000
    if cl_visits.max() > 0:
        norm_visits = cl_visits / cl_visits.max()
    else:
        norm_visits = np.ones(N_CLUSTERS) / N_CLUSTERS
    node_sizes = min_size + norm_visits * (max_size - min_size)

    # Draw edges (between clusters)
    max_val = max(tm.max(), 1)
    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            val = tm[i, j]
            if val == 0 or i == j:
                continue
            lw = 1 + (val / max_val) * 8
            alpha = 0.2 + 0.6 * (val / max_val)
            ax.annotate('', xy=cl_pos[j], xytext=cl_pos[i],
                        arrowprops=dict(arrowstyle='->', color=CLUSTER_COLORS[i],
                                        lw=lw, alpha=alpha,
                                        connectionstyle='arc3,rad=0.15'))

    # Draw self-loops (exploit) as circular arcs
    for i in range(N_CLUSTERS):
        val = tm[i, i]
        if val == 0:
            continue
        size = node_sizes[i]
        r = np.sqrt(size) / 200  # approximate node radius in data coords
        # Draw a small loop above the node
        angle = np.linspace(0, 2 * np.pi * 0.75, 30)
        loop_r = r * 1.5
        cx = cl_pos[i, 0] + r * 0.8
        cy = cl_pos[i, 1] + r * 0.8
        lx = cx + loop_r * np.cos(angle)
        ly = cy + loop_r * np.sin(angle)
        lw = 1 + (val / max_val) * 6
        ax.plot(lx, ly, color=CLUSTER_COLORS[i], linewidth=lw,
                alpha=0.5, zorder=2)
        ax.text(cx, cy + loop_r * 1.3, str(val), fontsize=8,
                fontweight='bold', ha='center', va='bottom',
                color=CLUSTER_COLORS[i])

    # Draw nodes
    for cl in range(N_CLUSTERS):
        ax.scatter(cl_pos[cl, 0], cl_pos[cl, 1], s=node_sizes[cl],
                   c=CLUSTER_COLORS[cl], edgecolors='white', linewidth=2,
                   zorder=4, alpha=0.9)

    # Node labels
    for cl in range(N_CLUSTERS):
        mask = labels == cl
        n_art = int(mask.sum())
        label = f'{CLUSTER_NAMES_ONE[cl]}\n({n_art} articles)'
        ax.text(cl_pos[cl, 0], cl_pos[cl, 1], label,
                fontsize=7.5, fontweight='bold', ha='center', va='center',
                zorder=5,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    ax.set_title('Compact Network — Clusters as Nodes\n'
                 'Node size = visit frequency | Arrow width = transition count | '
                 'Loops = exploit (within-cluster)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('MDS Dimension 1', fontsize=10)
    ax.set_ylabel('MDS Dimension 2', fontsize=10)
    ax.grid(True, alpha=0.1)

    # Edge count legend
    total_exploit = int(np.trace(tm))
    total_explore = int(tm.sum() - total_exploit)
    ax.text(0.02, 0.02,
            f'Exploit (loops): {total_exploit} | Explore (arrows): {total_explore}\n'
            f'Total transitions: {int(tm.sum())}',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.9, edgecolor='gray'))

    outpath = OUTPUT_DIR / 'm9_6_compact_network.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ========== MAIN ==========

def main():
    print("[M9] Generating 6 alternative visualizations...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all()
    print(f"  {len(data['pids'])} participants, {len(data['trials'])} trials")
    print(f"  Transition matrix total: {data['trans_matrix'].sum()}")
    print(f"  Exploit (diagonal): {np.trace(data['trans_matrix'])}")

    print("\n  1/6 Chord diagram...")
    draw_chord(data)

    print("  2/6 Heatmap...")
    draw_heatmap(data)

    print("  3/6 Small multiples...")
    draw_small_multiples(data)

    print("  4/6 Timeline with cluster bands...")
    draw_timeline_bands(data)

    print("  5/6 Sankey flow...")
    draw_sankey(data)

    print("  6/6 Compact network...")
    draw_compact_network(data)

    print("\n  Done! 6 images saved to output/")


if __name__ == '__main__':
    main()

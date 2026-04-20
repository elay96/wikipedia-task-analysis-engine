#!/usr/bin/env python3
"""
M13: Unified Explore/Exploit — Pages (M10) + Transitions (M12)
================================================================
Single line per trial combining:
  - Integer x (1, 2, 3...) = pages, classified by time vs session mean
  - Half x (1.5, 2.5...) = transitions, classified by cosine distance vs session mean
Y axis: binary Exploit (+0.5) / Explore (-0.5)
PCA: 7 dimensions.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

DATA_DIR = Path(__file__).parent / '..' / 'data'
SIM_PATH = DATA_DIR / 'cleaned' / 'similarity_matrix.json'
N_COMPONENTS = 7


def cos_dist(a, b):
    d = np.dot(a, b)
    nm = np.linalg.norm(a) * np.linalg.norm(b)
    return 1 - d / nm if nm > 0 else 1.0


def build_pca(n_components):
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

    X = mat - mat.mean(axis=0)
    cov = np.cov(X, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx_sort]
    eigenvalues = eigenvalues[idx_sort]
    pc = X @ eigenvectors[:, :n_components]

    var_explained = eigenvalues[:n_components].sum() / eigenvalues.sum() * 100
    return slugs, slug_idx, pc, var_explained


def build_sequences(pids, pid_trials, slug_idx, pc):
    """Build interleaved page+transition sequences per participant."""
    pid_data = {}

    for pid in pids:
        pid_data[pid] = []
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            if len(pvs) < 2:
                continue

            # Raw values
            page_times = [pv['duration'] for pv in pvs]

            trans_dists = []
            for i in range(1, len(pvs)):
                fi = slug_idx.get(pvs[i - 1]['title'])
                ti = slug_idx.get(pvs[i]['title'])
                if fi is not None and ti is not None:
                    trans_dists.append(cos_dist(pc[fi], pc[ti]))
                else:
                    trans_dists.append(np.nan)

            # Session means
            time_mean = np.mean(page_times)
            valid_dists = [d for d in trans_dists if not np.isnan(d)]
            dist_mean = np.mean(valid_dists) if valid_dists else 0.5

            # Build interleaved sequence: page1, trans1.5, page2, trans2.5, ...
            points = []
            for i in range(len(pvs)):
                # Page point (integer x)
                is_exploit = page_times[i] >= time_mean
                points.append({
                    'x': i + 1,
                    'y': 0.5 if is_exploit else -0.5,
                    'type': 'page',
                    'raw': page_times[i],
                    'threshold': time_mean,
                })

                # Transition point (half x) — between page i and page i+1
                if i < len(pvs) - 1:
                    d = trans_dists[i]
                    # For distance: LOWER distance = more similar = EXPLOIT
                    is_exploit_d = d <= dist_mean if not np.isnan(d) else False
                    points.append({
                        'x': i + 1.5,
                        'y': 0.5 if is_exploit_d else -0.5,
                        'type': 'transition',
                        'raw': d,
                        'threshold': dist_mean,
                    })

            pid_data[pid].append({
                'trial': tr['trial'],
                'condition': tr['condition'],
                'points': points,
                'time_mean': time_mean,
                'dist_mean': dist_mean,
            })

    return pid_data


def plot_trial_grid(pid_data, pids, trial_idx, var_explained, n_components, outname,
                    measure_label='M13', method_label='PCA', threshold_label='mean'):
    """Plot a single grid for one trial (trial_idx=0 or 1)."""
    n_pids = len(pids)
    cols = 4
    rows = int(np.ceil(n_pids / cols))

    EXPLOIT_BG = '#4CAF50'
    EXPLORE_BG = '#FF9800'
    LINE_COLOR = '#4FC3F7'

    trial_num = trial_idx + 1
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.2))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f'{measure_label}: Explore / Exploit — Trial {trial_num}  '
        f'({method_label} {n_components}D, {var_explained:.1f}% var.)\n'
        f'Integer X = page (time vs session {threshold_label}) · '
        f'Half X = transition (cosine distance vs session {threshold_label}) · '
        f'Green = Exploit · Orange = Explore',
        fontsize=13, color='#e6edf3', fontweight='bold', y=0.99)

    axes_flat = axes.flatten()

    for pi, pid in enumerate(pids):
        ax = axes_flat[pi]
        ax.set_facecolor('#0d1117')

        trial_list = pid_data[pid]
        td = trial_list[trial_idx] if trial_idx < len(trial_list) else None

        if not td or not td['points']:
            ax.set_title(f'User {pid}', fontsize=10, color='#e6edf3', fontweight='bold')
            ax.set_visible(True)
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            continue

        pts = td['points']
        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]

        # Background shading
        ax.axhspan(0, 0.85, facecolor=EXPLOIT_BG, alpha=0.05, zorder=0)
        ax.axhspan(-0.85, 0, facecolor=EXPLORE_BG, alpha=0.05, zorder=0)
        ax.axhline(y=0, color='#8b949e', linewidth=0.6, zorder=1)

        # Single connected line
        ax.plot(xs, ys, color=LINE_COLOR, linewidth=1.8, alpha=0.85, zorder=2)

        # Dots — same marker for all, color by exploit/explore
        for p in pts:
            c = EXPLOIT_BG if p['y'] > 0 else EXPLORE_BG
            ax.plot(p['x'], p['y'], 'o', color=c, markersize=7,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=4)

        # X ticks: only integers (page numbers)
        max_page = max(p['x'] for p in pts if p['type'] == 'page')
        ax.set_xticks(range(1, int(max_page) + 1))
        ax.set_xlim(0.5, max_page + 0.5)

        ax.set_ylim(-0.85, 0.85)
        ax.set_yticks([0.5, -0.5])
        ax.set_yticklabels(['Exploit', 'Explore'], fontsize=8, color='#c9d1d9')
        ax.set_xlabel('Page #', fontsize=8, color='#8b949e')
        ax.tick_params(axis='x', colors='#8b949e', labelsize=7)

        ax.set_title(f'User {pid}', fontsize=10, color='#e6edf3', fontweight='bold', pad=5)

        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(False)

    for k in range(n_pids, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    outpath = OUTPUT_DIR / outname
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


def main():
    slugs, slug_idx, pc, var_explained = build_pca(N_COMPONENTS)

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    pid_data = build_sequences(pids, pid_trials, slug_idx, pc)

    # Two separate grids: Trial 1 and Trial 2
    plot_trial_grid(pid_data, pids, 0, var_explained, N_COMPONENTS, 'm13_trial1.png')
    plot_trial_grid(pid_data, pids, 1, var_explained, N_COMPONENTS, 'm13_trial2.png')

    # Summary
    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M13 {t_name} (PCA {N_COMPONENTS}D) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


if __name__ == '__main__':
    main()

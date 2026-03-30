#!/usr/bin/env python3
"""
M12: Cosine Distance Between Consecutive Pages (7-dim PCA)
============================================================
Per-participant subplots showing cosine distance between
each consecutive page pair in the reduced 7-dimensional space.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

DATA_DIR = Path(__file__).parent / '..' / 'data'
SIM_PATH = DATA_DIR / 'similarity_matrix.json'
N_COMPONENTS = 7


def cosine_distance(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return 1.0 - dot / norm


def main():
    # Load and build similarity matrix
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

    # PCA → 7 dimensions
    X = mat.copy()
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    pc_scores = X_centered @ eigenvectors[:, :N_COMPONENTS]

    total_var = eigenvalues.sum()
    var_explained = eigenvalues[:N_COMPONENTS].sum() / total_var * 100

    # Load trials
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    # Compute per-participant transition distances
    pid_data = {}
    for pid in pids:
        transitions = []
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            for i in range(1, len(pvs)):
                fi = slug_idx.get(pvs[i - 1]['title'])
                ti = slug_idx.get(pvs[i]['title'])
                if fi is not None and ti is not None:
                    d = cosine_distance(pc_scores[fi], pc_scores[ti])
                    transitions.append({
                        'step': i,
                        'trial': tr['trial'],
                        'dist': d,
                        'from': pvs[i - 1]['title'],
                        'to': pvs[i]['title'],
                    })
        pid_data[pid] = transitions

    # ========== FIGURE ==========
    n_pids = len(pids)
    cols = 4
    rows = int(np.ceil(n_pids / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(22, rows * 3.5),
                             sharex=False, sharey=True)
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f'M12: Cosine Distance Between Consecutive Pages (PCA → {N_COMPONENTS}D, {var_explained:.1f}% variance)\n'
        f'Lower = more similar pages · Higher = topic switch',
        fontsize=15, color='#e6edf3', fontweight='bold', y=0.98)

    axes_flat = axes.flatten()
    trial_markers = ['o', 's', 'D', '^', 'v']
    trial_colors = ['#4FC3F7', '#FF8A65', '#AED581', '#CE93D8', '#FFD54F']

    for pi, pid in enumerate(pids):
        ax = axes_flat[pi]
        ax.set_facecolor('#0d1117')

        trans = pid_data[pid]
        if not trans:
            ax.set_title(f'User {pid}', fontsize=10, color='#e6edf3', fontweight='bold')
            continue

        trial_ids = sorted(set(t['trial'] for t in trans))

        for ji, trial_id in enumerate(trial_ids):
            trial_trans = [t for t in trans if t['trial'] == trial_id]
            steps = list(range(1, len(trial_trans) + 1))
            dists = [t['dist'] for t in trial_trans]
            color = trial_colors[ji % len(trial_colors)]
            marker = trial_markers[ji % len(trial_markers)]

            ax.plot(steps, dists, marker=marker, markersize=5,
                    linewidth=1.5, alpha=0.85, color=color,
                    markeredgecolor='white', markeredgewidth=0.4,
                    label=f'Trial {trial_id}', zorder=3)

            # Mean line
            if len(dists) > 1:
                mean_d = np.mean(dists)
                ax.axhline(y=mean_d, color=color, linestyle='--',
                            linewidth=0.8, alpha=0.5)

        ax.set_title(f'User {pid}', fontsize=10, color='#e6edf3', fontweight='bold', pad=5)
        ax.set_xlabel('Transition #', fontsize=8, color='#8b949e')
        ax.set_ylabel('Cosine Dist.', fontsize=8, color='#8b949e')
        ax.tick_params(colors='#8b949e', labelsize=7)
        ax.grid(True, color='#21262d', linewidth=0.5, zorder=0)
        ax.set_ylim(-0.05, 1.05)

        for spine in ax.spines.values():
            spine.set_color('#30363d')

        if len(trial_ids) > 1:
            ax.legend(fontsize=6, facecolor='#161b22', edgecolor='#30363d',
                      labelcolor='#c9d1d9')

    for k in range(n_pids, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    outpath = OUTPUT_DIR / 'm12_pca7_distance.png'
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')

    # Summary
    all_dists = [t['dist'] for trans in pid_data.values() for t in trans]
    print(f'\n=== M12 Summary ===')
    print(f'PCA dimensions: {N_COMPONENTS} ({var_explained:.1f}% variance)')
    print(f'Total transitions: {len(all_dists)}')
    print(f'Mean cosine distance: {np.mean(all_dists):.4f}')
    print(f'SD: {np.std(all_dists):.4f}')
    print(f'Range: [{np.min(all_dists):.4f}, {np.max(all_dists):.4f}]')


if __name__ == '__main__':
    main()

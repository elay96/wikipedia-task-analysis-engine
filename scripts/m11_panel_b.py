#!/usr/bin/env python3
"""Extract Panel B (Cumulative Variance Explained) from M11 as standalone."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from helpers import OUTPUT_DIR

DATA_DIR = Path(__file__).parent / '..' / 'data'
SIM_PATH = DATA_DIR / 'similarity_matrix.json'


def main():
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
    eigenvalues = eigenvalues[idx_sort]

    total_var = eigenvalues.sum()
    var_explained = eigenvalues / total_var
    cumulative = np.cumsum(var_explained)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n + 1), cumulative * 100, color='#283593', linewidth=2.5)
    ax.fill_between(range(1, n + 1), cumulative * 100, alpha=0.1, color='#5C6BC0')

    for thresh, style, color in [(90, '--', '#E53935'), (95, ':', '#E53935')]:
        k = np.searchsorted(cumulative, thresh / 100) + 1
        ax.axhline(y=thresh, color=color, linestyle=style, linewidth=0.8, alpha=0.6)
        ax.axvline(x=k, color=color, linestyle=style, linewidth=0.8, alpha=0.6)
        ax.plot(k, thresh, 'o', color=color, markersize=7, zorder=5)
        ax.annotate(f'{thresh}% → {k} PCs',
                    xy=(k, thresh), xytext=(k + 8, thresh - 4),
                    fontsize=10, fontweight='bold', color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # Mark 7D
    k7_var = cumulative[6] * 100
    ax.plot(7, k7_var, 's', color='#FF9800', markersize=9, zorder=5)
    ax.annotate(f'7 PCs → {k7_var:.1f}%',
                xy=(7, k7_var), xytext=(15, k7_var + 5),
                fontsize=10, fontweight='bold', color='#FF9800',
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.2))

    ax.set_xlabel('Number of Principal Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'PCA — Cumulative Variance Explained\n'
                 f'{n} articles · tf-idf cosine similarity matrix',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(1, n)
    ax.set_ylim(0, 102)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    outpath = OUTPUT_DIR / 'm11_panel_b_cumulative_variance.png'
    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()

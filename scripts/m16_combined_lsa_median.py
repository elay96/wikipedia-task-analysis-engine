#!/usr/bin/env python3
"""
M16: Unified Explore/Exploit — LSA + Median Threshold
======================================================
Same as M15 (median threshold, binary classification) but uses
LSA (Truncated SVD) instead of PCA for dimensionality reduction.

LSA applies SVD directly to the centered similarity matrix:
  U, S, Vt = svd(X_centered)
  components = U[:, :k] * S[:k]

This is more appropriate than PCA for similarity/co-occurrence
matrices because it doesn't go through the covariance matrix.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import json
import numpy as np
from pathlib import Path

from m13_combined_binary import cos_dist, plot_trial_grid
from m15_combined_binary_median import build_sequences_median
from helpers import load_trials, get_pids_and_trials

DATA_DIR = Path(__file__).parent / '..' / 'data'
SIM_PATH = DATA_DIR / 'similarity_matrix.json'
TARGET_VARIANCE = 0.90


def build_lsa(target_var=TARGET_VARIANCE):
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

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Find number of components for target variance
    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
    n_components = int(np.searchsorted(cumvar, target_var) + 1)

    components = U[:, :n_components] * S[:n_components]
    var_explained = cumvar[n_components - 1] * 100

    print(f'LSA: {n_components} components needed for >= {target_var*100:.0f}% variance (actual: {var_explained:.1f}%)')
    return slugs, slug_idx, components, var_explained, n_components


def main():
    slugs, slug_idx, lsa, var_explained, n_components = build_lsa()

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    pid_data = build_sequences_median(pids, pid_trials, slug_idx, lsa)

    plot_trial_grid(pid_data, pids, 0, var_explained, n_components, 'm16_trial1.png',
                    measure_label='M16', method_label='LSA', threshold_label='median')
    plot_trial_grid(pid_data, pids, 1, var_explained, n_components, 'm16_trial2.png',
                    measure_label='M16', method_label='LSA', threshold_label='median')

    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M16 {t_name} (LSA {n_components}D, MEDIAN threshold) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
M14: Unified Explore/Exploit — Pages (M10) + Transitions (M12) — PCA 71D
==========================================================================
Same as M13 but with 71 PCA dimensions (90% variance).
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from m13_combined_binary import build_pca, build_sequences, plot_trial_grid
from helpers import load_trials, get_pids_and_trials

N_COMPONENTS = 71


def main():
    slugs, slug_idx, pc, var_explained = build_pca(N_COMPONENTS)

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    pid_data = build_sequences(pids, pid_trials, slug_idx, pc)

    plot_trial_grid(pid_data, pids, 0, var_explained, N_COMPONENTS, 'm14_trial1.png')
    plot_trial_grid(pid_data, pids, 1, var_explained, N_COMPONENTS, 'm14_trial2.png')

    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M14 {t_name} (PCA {N_COMPONENTS}D) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


if __name__ == '__main__':
    main()

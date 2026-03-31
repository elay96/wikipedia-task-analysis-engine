#!/usr/bin/env python3
"""
M28: Explore/Exploit — LSA Transitions + 60s Pages (4.ב)
========================================================
Pages: Exploit if page duration >= 60 seconds (M2 logic).
Transitions: Exploit if cosine distance <= cross-subject domain median (M20 logic).
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')

from m13_combined_binary import cos_dist
from m16_combined_lsa_median import build_lsa
from m20_cross_subject_median import (
    compute_domain_medians,
    count_switches,
    plot_trial_grid_with_switches,
)
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

TIME_THRESHOLD = 60


def build_sequences_60s(pids, pid_trials, slug_idx, pc, domain_medians):
    pid_data = {}

    for pid in pids:
        pid_data[pid] = []
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            if len(pvs) < 2:
                continue

            domain = tr['domain']
            dist_threshold = domain_medians[domain]

            trans_dists = []
            for i in range(1, len(pvs)):
                fi = slug_idx.get(pvs[i - 1]['title'])
                ti = slug_idx.get(pvs[i]['title'])
                if fi is not None and ti is not None:
                    trans_dists.append(cos_dist(pc[fi], pc[ti]))
                else:
                    trans_dists.append(np.nan)

            points = []
            for i in range(len(pvs)):
                is_exploit = pvs[i]['duration'] >= TIME_THRESHOLD
                points.append({
                    'x': i + 1,
                    'y': 0.5 if is_exploit else -0.5,
                    'type': 'page',
                })

                if i < len(pvs) - 1:
                    d = trans_dists[i]
                    is_exploit_d = d <= dist_threshold if not np.isnan(d) else False
                    points.append({
                        'x': i + 1.5,
                        'y': 0.5 if is_exploit_d else -0.5,
                        'type': 'transition',
                        'raw': d,
                        'threshold': dist_threshold,
                    })

            pid_data[pid].append({
                'trial': tr['trial'],
                'condition': tr['condition'],
                'points': points,
                'dist_threshold': dist_threshold,
                'domain': domain,
            })

    return pid_data


def main():
    slugs, slug_idx, lsa, var_explained, n_components = build_lsa()

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    print('\nCross-subject median cosine distance per domain:')
    domain_medians = compute_domain_medians(pid_trials, slug_idx, lsa)

    pid_data = build_sequences_60s(pids, pid_trials, slug_idx, lsa, domain_medians)

    plot_trial_grid_with_switches(pid_data, pids, 0, var_explained, n_components, 'm28_trial1.png')
    plot_trial_grid_with_switches(pid_data, pids, 1, var_explained, n_components, 'm28_trial2.png')

    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M28 {t_name} (LSA {n_components}D, 60s pages + cross-subject median transitions) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


if __name__ == '__main__':
    main()

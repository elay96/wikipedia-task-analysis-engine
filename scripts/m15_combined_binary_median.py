#!/usr/bin/env python3
"""
M15: Unified Explore/Exploit — Median Threshold — PCA 71D
==========================================================
Same as M14 but uses session MEDIAN instead of MEAN for classification.
Rationale: if page-time or transition-distance distributions are skewed,
the mean is pulled toward outliers and the median is more robust.
Prints distribution diagnostics (skewness, mean vs median) per participant.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
from scipy import stats as sp_stats
from m13_combined_binary import build_pca, cos_dist, plot_trial_grid
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

N_COMPONENTS = 71


def build_sequences_median(pids, pid_trials, slug_idx, pc):
    """Like M13 build_sequences but threshold = session median."""
    pid_data = {}

    for pid in pids:
        pid_data[pid] = []
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            if len(pvs) < 2:
                continue

            page_times = [pv['duration'] for pv in pvs]

            trans_dists = []
            for i in range(1, len(pvs)):
                fi = slug_idx.get(pvs[i - 1]['title'])
                ti = slug_idx.get(pvs[i]['title'])
                if fi is not None and ti is not None:
                    trans_dists.append(cos_dist(pc[fi], pc[ti]))
                else:
                    trans_dists.append(np.nan)

            time_median = np.median(page_times)
            valid_dists = [d for d in trans_dists if not np.isnan(d)]
            dist_median = np.median(valid_dists) if valid_dists else 0.5

            points = []
            for i in range(len(pvs)):
                is_exploit = page_times[i] >= time_median
                points.append({
                    'x': i + 1,
                    'y': 0.5 if is_exploit else -0.5,
                    'type': 'page',
                    'raw': page_times[i],
                    'threshold': time_median,
                })

                if i < len(pvs) - 1:
                    d = trans_dists[i]
                    is_exploit_d = d <= dist_median if not np.isnan(d) else False
                    points.append({
                        'x': i + 1.5,
                        'y': 0.5 if is_exploit_d else -0.5,
                        'type': 'transition',
                        'raw': d,
                        'threshold': dist_median,
                    })

            pid_data[pid].append({
                'trial': tr['trial'],
                'condition': tr['condition'],
                'points': points,
                'time_median': time_median,
                'dist_median': dist_median,
            })

    return pid_data


def print_distribution_diagnostics(pids, pid_trials, slug_idx, pc):
    """Print skewness, mean, median for page-time and transition-distance."""
    all_page_times = []
    all_trans_dists = []

    for pid in pids:
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            if len(pvs) < 2:
                continue
            all_page_times.extend(pv['duration'] for pv in pvs)
            for i in range(1, len(pvs)):
                fi = slug_idx.get(pvs[i - 1]['title'])
                ti = slug_idx.get(pvs[i]['title'])
                if fi is not None and ti is not None:
                    all_trans_dists.append(cos_dist(pc[fi], pc[ti]))

    pt = np.array(all_page_times)
    td = np.array(all_trans_dists)

    print('\n' + '=' * 60)
    print('DISTRIBUTION DIAGNOSTICS — Mean vs Median reliability')
    print('=' * 60)

    for name, arr in [('Page durations (sec)', pt), ('Transition distances', td)]:
        mean = np.mean(arr)
        median = np.median(arr)
        skew = sp_stats.skew(arr)
        std = np.std(arr)
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        ratio = abs(mean - median) / std if std > 0 else 0

        print(f'\n--- {name} (n={len(arr)}) ---')
        print(f'  Mean:    {mean:.4f}')
        print(f'  Median:  {median:.4f}')
        print(f'  Std:     {std:.4f}')
        print(f'  IQR:     {iqr:.4f}')
        skew_label = "(right-skewed)" if skew > 0.5 else "(left-skewed)" if skew < -0.5 else "(roughly symmetric)"
        verdict = "=> median recommended" if ratio > 0.1 or abs(skew) > 0.5 else "=> mean is fine"
        print(f'  Skewness:{skew:+.3f}  {skew_label}')
        print(f'  |Mean-Median|/Std: {ratio:.3f}  {verdict}')

    print('\n' + '=' * 60)


def main():
    slugs, slug_idx, pc, var_explained = build_pca(N_COMPONENTS)

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)

    print_distribution_diagnostics(pids, pid_trials, slug_idx, pc)

    pid_data = build_sequences_median(pids, pid_trials, slug_idx, pc)

    plot_trial_grid(pid_data, pids, 0, var_explained, N_COMPONENTS, 'm15_trial1.png',
                    measure_label='M15', method_label='PCA', threshold_label='median')
    plot_trial_grid(pid_data, pids, 1, var_explained, N_COMPONENTS, 'm15_trial2.png',
                    measure_label='M15', method_label='PCA', threshold_label='median')

    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M15 {t_name} (PCA {N_COMPONENTS}D, MEDIAN threshold) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


if __name__ == '__main__':
    main()

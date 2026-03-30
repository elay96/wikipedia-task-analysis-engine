#!/usr/bin/env python3
"""
M20: Explore/Exploit — Typing (M18) + Cross-Subject Transition Median
=====================================================================
Pages: Exploit = typing/pasting on page (M18 logic)
Transitions: threshold = median cosine distance across ALL subjects
who answered the same question (domain). LSA for dim reduction.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from m13_combined_binary import cos_dist
from m16_combined_lsa_median import build_lsa
from m18_typing_binary import page_had_typing_or_paste
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR


def compute_domain_medians(pid_trials, slug_idx, pc):
    """Compute median cosine distance per domain across all subjects."""
    domain_dists = defaultdict(list)

    for pid, trials in pid_trials.items():
        for tr in trials:
            pvs = tr['page_visits']
            domain = tr['domain']
            for i in range(1, len(pvs)):
                fi = slug_idx.get(pvs[i - 1]['title'])
                ti = slug_idx.get(pvs[i]['title'])
                if fi is not None and ti is not None:
                    domain_dists[domain].append(cos_dist(pc[fi], pc[ti]))

    domain_medians = {}
    for domain, dists in domain_dists.items():
        domain_medians[domain] = np.median(dists)
        print(f'  {domain}: n={len(dists)}, median={domain_medians[domain]:.4f}')

    return domain_medians


def count_switches(points):
    """Count number of transitions between exploit and explore."""
    switches = 0
    for i in range(1, len(points)):
        if points[i]['y'] != points[i - 1]['y']:
            switches += 1
    return switches


def plot_trial_grid_with_switches(pid_data, pids, trial_idx, var_explained,
                                  n_components, outname):
    n_pids = len(pids)
    cols = 4
    rows = int(np.ceil(n_pids / cols))
    trial_num = trial_idx + 1

    EXPLOIT_BG = '#4CAF50'
    EXPLORE_BG = '#FF9800'
    LINE_COLOR = '#4FC3F7'

    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.2))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f'M20: Explore / Exploit — Trial {trial_num}  '
        f'(LSA {n_components}D, {var_explained:.1f}% var.)\n'
        f'Page = typing/paste · Transition = cosine dist vs cross-subj median · '
        f'SR (Switch Rate) = % of strategy changes out of all steps',
        fontsize=13, color='#e6edf3', fontweight='bold', y=0.99)

    axes_flat = axes.flatten()

    for pi, pid in enumerate(pids):
        ax = axes_flat[pi]
        ax.set_facecolor('#0d1117')

        trial_list = pid_data[pid]
        td = trial_list[trial_idx] if trial_idx < len(trial_list) else None

        if not td or not td['points']:
            ax.set_title(f'User {pid}', fontsize=10, color='#e6edf3', fontweight='bold')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            continue

        pts = td['points']
        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]
        sw = count_switches(pts)
        sw_rate = sw / (len(pts) - 1) * 100 if len(pts) > 1 else 0

        ax.axhspan(0, 0.85, facecolor=EXPLOIT_BG, alpha=0.05, zorder=0)
        ax.axhspan(-0.85, 0, facecolor=EXPLORE_BG, alpha=0.05, zorder=0)
        ax.axhline(y=0, color='#8b949e', linewidth=0.6, zorder=1)

        ax.plot(xs, ys, color=LINE_COLOR, linewidth=1.8, alpha=0.85, zorder=2)

        for p in pts:
            c = EXPLOIT_BG if p['y'] > 0 else EXPLORE_BG
            ax.plot(p['x'], p['y'], 'o', color=c, markersize=7,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=4)

        max_page = max(p['x'] for p in pts if p['type'] == 'page')
        ax.set_xticks(range(1, int(max_page) + 1))
        ax.set_xlim(0.5, max_page + 0.5)
        ax.set_ylim(-0.85, 0.85)
        ax.set_yticks([0.5, -0.5])
        ax.set_yticklabels(['Exploit', 'Explore'], fontsize=8, color='#c9d1d9')
        ax.set_xlabel('Page #', fontsize=8, color='#8b949e')
        ax.tick_params(axis='x', colors='#8b949e', labelsize=7)

        ax.set_title(f'User {pid} — SR: {sw_rate:.0f}%',
                     fontsize=10, color='#e6edf3', fontweight='bold', pad=5)

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


def build_sequences(pids, pid_trials, slug_idx, pc, domain_medians):
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
                is_exploit = page_had_typing_or_paste(
                    pvs[i], tr['typing_intervals'], tr['paste_times'])
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

    pid_data = build_sequences(pids, pid_trials, slug_idx, lsa, domain_medians)

    plot_trial_grid_with_switches(pid_data, pids, 0, var_explained, n_components, 'm20_trial1.png')
    plot_trial_grid_with_switches(pid_data, pids, 1, var_explained, n_components, 'm20_trial2.png')

    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M20 {t_name} (LSA {n_components}D, cross-subject median) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


if __name__ == '__main__':
    main()

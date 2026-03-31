#!/usr/bin/env python3
"""
M26: Explore/Exploit — Topic Modeling (LDA JSD Distance)
=========================================================
Pages: Exploit = typing/pasting on page (M18 logic)
Transitions: threshold = median JSD topic distance across ALL subjects
who answered the same question (domain).
"""

import sys
import json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from m18_typing_binary import page_had_typing_or_paste
from m20_cross_subject_median import plot_trial_grid_with_switches, count_switches
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

DATA_DIR = __import__('pathlib').Path(__file__).parent.parent / 'data'
TOPIC_MODEL_PATH = DATA_DIR / 'topic_model.json'

N_TOPIC_WORDS = 8


def load_topic_model():
    with open(TOPIC_MODEL_PATH) as f:
        tm = json.load(f)
    slugs = tm['slugs']
    slug_idx = {s: i for i, s in enumerate(slugs)}
    return tm, slugs, slug_idx


def topic_dist(tm, slug_a, slug_b):
    key_ab = f'{slug_a}|||{slug_b}'
    key_ba = f'{slug_b}|||{slug_a}'
    distances = tm['distances']
    if key_ab in distances:
        return distances[key_ab]
    if key_ba in distances:
        return distances[key_ba]
    return np.nan


def compute_domain_medians_topic(pid_trials, tm):
    domain_dists = defaultdict(list)

    for pid, trials in pid_trials.items():
        for tr in trials:
            pvs = tr['page_visits']
            domain = tr['domain']
            for i in range(1, len(pvs)):
                d = topic_dist(tm, pvs[i - 1]['title'], pvs[i]['title'])
                if not np.isnan(d):
                    domain_dists[domain].append(d)

    domain_medians = {}
    for domain, dists in domain_dists.items():
        domain_medians[domain] = np.median(dists)
        print(f'  {domain}: n={len(dists)}, median={domain_medians[domain]:.4f}')

    return domain_medians


def build_sequences_topic(pids, pid_trials, tm, domain_medians):
    pid_data = {}

    for pid in pids:
        pid_data[pid] = []
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            if len(pvs) < 2:
                continue

            domain = tr['domain']
            dist_threshold = domain_medians.get(domain, np.nan)

            trans_dists = []
            for i in range(1, len(pvs)):
                d = topic_dist(tm, pvs[i - 1]['title'], pvs[i]['title'])
                trans_dists.append(d)

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
                    if np.isnan(dist_threshold):
                        is_exploit_d = False
                    else:
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


def plot_topic_overview(tm, pid_trials, pids):
    domain_dists = defaultdict(list)
    for pid in pids:
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            domain = tr['domain']
            for i in range(1, len(pvs)):
                d = topic_dist(tm, pvs[i - 1]['title'], pvs[i]['title'])
                if not np.isnan(d):
                    domain_dists[domain].append(d)

    domains = sorted(domain_dists.keys())
    topic_words = tm['topic_words']
    n_topics = tm['n_topics']

    DOMAIN_COLORS = [
        '#4FC3F7', '#81C784', '#FFB74D', '#F06292',
        '#CE93D8', '#80DEEA', '#FFCC80', '#A5D6A7',
    ]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')

    # Left panel: histogram of JSD distances per domain
    ax_left.set_facecolor('#0d1117')
    bins = np.linspace(0, 1, 30)
    for di, domain in enumerate(domains):
        color = DOMAIN_COLORS[di % len(DOMAIN_COLORS)]
        dists = domain_dists[domain]
        ax_left.hist(dists, bins=bins, alpha=0.55, color=color,
                     label=f'{domain} (n={len(dists)})', density=True)
        med = np.median(dists)
        ax_left.axvline(med, color=color, linewidth=1.5, linestyle='--', alpha=0.9)

    ax_left.set_xlabel('JSD Topic Distance', fontsize=11, color='#c9d1d9')
    ax_left.set_ylabel('Density', fontsize=11, color='#c9d1d9')
    ax_left.set_title('Topic Distance Distribution per Domain\n(dashed = cross-subject median)',
                       fontsize=12, color='#e6edf3', fontweight='bold')
    ax_left.tick_params(colors='#8b949e')
    for spine in ax_left.spines.values():
        spine.set_color('#30363d')
    ax_left.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d',
                   labelcolor='#c9d1d9')

    # Right panel: top words per topic
    ax_right.set_facecolor('#0d1117')
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(-0.5, n_topics - 0.5)
    ax_right.axis('off')

    ax_right.set_title('Top Words per LDA Topic', fontsize=12,
                        color='#e6edf3', fontweight='bold')

    topic_colors = plt.cm.tab10(np.linspace(0, 1, n_topics))

    for t in range(n_topics):
        y = n_topics - 1 - t
        words = topic_words[str(t)][:N_TOPIC_WORDS]
        label = f'T{t}: {", ".join(words)}'
        ax_right.text(0.02, y, label, fontsize=9.5, color=topic_colors[t],
                      va='center', ha='left',
                      bbox=dict(facecolor='#161b22', edgecolor='#30363d',
                                boxstyle='round,pad=0.3', alpha=0.8))

    fig.suptitle('M26: Topic Modeling Overview — LDA JSD Distances',
                 fontsize=14, color='#e6edf3', fontweight='bold', y=1.01)

    plt.tight_layout()
    outpath = OUTPUT_DIR / 'm26_topic_overview.png'
    plt.savefig(outpath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {outpath}')


def main():
    print('Loading topic model...')
    tm, slugs, slug_idx = load_topic_model()
    print(f'  {len(slugs)} slugs, {tm["n_topics"]} topics')

    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    print(f'  {len(trials)} trials from {len(pids)} participants')

    print('\nCross-subject median JSD topic distance per domain:')
    domain_medians = compute_domain_medians_topic(pid_trials, tm)

    pid_data = build_sequences_topic(pids, pid_trials, tm, domain_medians)

    plot_topic_overview(tm, pid_trials, pids)

    # Reuse m20's grid plotter — pass dummy var_explained / n_components
    # Override suptitle by patching — just pass a sentinel value
    _plot_grid_m26(pid_data, pids, 0, 'm26_trial1.png')
    _plot_grid_m26(pid_data, pids, 1, 'm26_trial2.png')

    for t_idx, t_name in [(0, 'Trial 1'), (1, 'Trial 2')]:
        pts = [p for pid in pids if t_idx < len(pid_data[pid])
               for p in pid_data[pid][t_idx]['points']]
        pages = [p for p in pts if p['type'] == 'page']
        trans = [p for p in pts if p['type'] == 'transition']
        page_ex = sum(1 for p in pages if p['y'] > 0) / len(pages) * 100 if pages else 0
        trans_ex = sum(1 for p in trans if p['y'] > 0) / len(trans) * 100 if trans else 0
        print(f'\n=== M26 {t_name} (LDA topic JSD, cross-subject median) ===')
        print(f'Pages: {len(pages)}, exploit rate: {page_ex:.1f}%')
        print(f'Transitions: {len(trans)}, exploit rate: {trans_ex:.1f}%')


def _plot_grid_m26(pid_data, pids, trial_idx, outname):
    """Grid plot variant with M26-specific title."""
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
        f'M26: Explore / Exploit — Trial {trial_num}  (LDA Topic Modeling)\n'
        f'Page = typing/paste · Transition = JSD topic dist vs cross-subj median · '
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

        page_pts = [p for p in pts if p['type'] == 'page']
        if page_pts:
            max_page = max(p['x'] for p in page_pts)
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


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
M5: Navigation-Based Explore-Exploit
======================================
Exploit = link click (staying in topic) | Explore = search (new topic)
Output: m5_navigation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from helpers import load_trials, get_pids_and_trials, setup_figure, finish_timeline, sorted_ratio_panel, OUTPUT_DIR


def main():
    print("[M5] Navigation type")
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    n = len(pids)
    print(f"  {len(trials)} trials from {n} participants")

    fig, ax_top, ax_bl, ax_br = setup_figure(n,
        'Explore vs. Exploit — Navigation Type\n'
        'Green = Exploit (link click) | Orange = Explore (search)')

    pid_exploit = {p: [] for p in pids}
    pid_explore = {p: [] for p in pids}

    for pi, pid in enumerate(pids):
        y_c = (n - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y = y_c + (0.18 if ti == 0 else -0.18)
            ax_top.barh(y, tr['duration'], height=0.28, color='#FAFAFA',
                        edgecolor='#E0E0E0', linewidth=0.3, zorder=1)
            ex_t, br_t = 0, 0
            for pv in tr['page_visits']:
                if pv['nav_type'] == 'link_click':
                    c, ec = '#4CAF50', '#2E7D32'
                    ex_t += pv['duration']
                else:
                    c, ec = '#FF9800', '#E65100'
                    br_t += pv['duration']
                ax_top.barh(y, pv['duration'], left=pv['start'], height=0.28,
                            color=c, alpha=0.7, edgecolor=ec, linewidth=0.5, zorder=3)
                ax_top.plot([pv['start'], pv['start']], [y - 0.14, y + 0.14],
                            color='black', linewidth=0.4, zorder=4, alpha=0.5)
            pct = ex_t / tr['duration'] * 100 if tr['duration'] > 0 else 0
            ax_top.text(tr['duration'] + 8, y, f"{tr['domain'][:6]} ({pct:.0f}%)",
                        fontsize=6, va='center', color='#616161')
            pid_exploit[pid].append(ex_t)
            pid_explore[pid].append(br_t)

    finish_timeline(ax_top, pids)
    ax_top.legend(handles=[
        mpatches.Patch(color='#4CAF50', alpha=0.7, label='Exploit (link click)'),
        mpatches.Patch(color='#FF9800', alpha=0.7, label='Explore (search)'),
        Line2D([0], [0], color='black', linewidth=0.5, label='Page transition'),
    ], fontsize=8, loc='lower right', framealpha=0.9, ncol=3)

    ex_m = [np.mean(pid_exploit[p]) for p in pids]
    br_m = [np.mean(pid_explore[p]) for p in pids]
    x = np.arange(n)
    ax_bl.bar(x, ex_m, color='#4CAF50', label='Exploit (link)', edgecolor='gray', width=0.6)
    ax_bl.bar(x, br_m, bottom=ex_m, color='#FF9800', label='Explore (search)', edgecolor='gray', width=0.6)
    ax_bl.set_xticks(x)
    ax_bl.set_xticklabels([f"P{p}" for p in pids], fontsize=8, rotation=45, ha='right')
    ax_bl.set_ylabel('Mean time (s)')
    ax_bl.set_title('Avg Exploit (link) vs Explore (search)')
    ax_bl.legend(fontsize=7)

    ratios = [ex_m[i] / (ex_m[i] + br_m[i]) if (ex_m[i] + br_m[i]) > 0 else 0.5 for i in range(n)]
    sorted_ratio_panel(ax_br, pids, ratios, '#4CAF50', '#FF9800', 'Exploit', 'Link-click ratio')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm5_navigation.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()

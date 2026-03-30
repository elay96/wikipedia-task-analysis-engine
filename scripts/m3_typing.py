#!/usr/bin/env python3
"""
M3: Typing-Based Explore-Exploit
==================================
Exploit = typing/editing answer | Explore = browsing Wikipedia
Paste events shown as red triangles
Output: m3_typing.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from helpers import load_trials, get_pids_and_trials, setup_figure, finish_timeline, sorted_ratio_panel, OUTPUT_DIR


def main():
    print("[M3] Typing vs browsing")
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    n = len(pids)
    print(f"  {len(trials)} trials from {n} participants")

    fig, ax_top, ax_bl, ax_br = setup_figure(n,
        'Explore vs. Exploit — Typing-Based\n'
        'Blue = Exploit (typing answer) | Green = Explore (browsing)')

    pid_exploit = {p: [] for p in pids}
    pid_explore = {p: [] for p in pids}

    for pi, pid in enumerate(pids):
        y_c = (n - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y = y_c + (0.18 if ti == 0 else -0.18)
            # Full bar = explore
            ax_top.barh(y, tr['duration'], height=0.28, color='#4CAF50', alpha=0.5,
                        edgecolor='#2E7D32', linewidth=0.3, zorder=2)
            # Typing = exploit (blue)
            typing_total = 0
            for bs, be in tr['typing_intervals']:
                ax_top.barh(y, be - bs, left=bs, height=0.28,
                            color='#2196F3', alpha=0.85, edgecolor='#0D47A1', linewidth=0.5, zorder=4)
                typing_total += be - bs
            # Page ticks
            for pv in tr['page_visits']:
                ax_top.plot([pv['start'], pv['start']], [y - 0.14, y + 0.14],
                            color='black', linewidth=0.4, zorder=5, alpha=0.5)
            # Paste triangles
            for pt in tr['paste_times']:
                ax_top.scatter(pt, y, marker='v', s=30, color='#F44336',
                               zorder=6, edgecolors='black', linewidths=0.3, alpha=0.8)

            pct = typing_total / tr['duration'] * 100 if tr['duration'] > 0 else 0
            ax_top.text(tr['duration'] + 8, y, f"{tr['domain'][:6]} ({pct:.0f}%)",
                        fontsize=6, va='center', color='#616161')
            pid_exploit[pid].append(typing_total)
            pid_explore[pid].append(tr['duration'] - typing_total)

    finish_timeline(ax_top, pids)
    ax_top.legend(handles=[
        mpatches.Patch(color='#4CAF50', alpha=0.5, label='Explore (browsing)'),
        mpatches.Patch(color='#2196F3', alpha=0.85, label='Exploit (typing)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#F44336', markersize=8, label='Paste'),
        Line2D([0], [0], color='black', linewidth=0.5, label='Page open'),
    ], fontsize=8, loc='lower right', framealpha=0.9, ncol=4)

    ex_m = [np.mean(pid_exploit[p]) for p in pids]
    br_m = [np.mean(pid_explore[p]) for p in pids]
    x = np.arange(n)
    ax_bl.bar(x, ex_m, color='#2196F3', label='Exploit (typing)', edgecolor='gray', width=0.6)
    ax_bl.bar(x, br_m, bottom=ex_m, color='#4CAF50', label='Explore (browsing)', edgecolor='gray', width=0.6)
    ax_bl.set_xticks(x)
    ax_bl.set_xticklabels([f"P{p}" for p in pids], fontsize=8, rotation=45, ha='right')
    ax_bl.set_ylabel('Mean time (s)')
    ax_bl.set_title('Avg Exploit (typing) vs Explore (browsing)')
    ax_bl.legend(fontsize=7)

    ratios = [ex_m[i] / (ex_m[i] + br_m[i]) if (ex_m[i] + br_m[i]) > 0 else 0 for i in range(n)]
    sorted_ratio_panel(ax_br, pids, ratios, '#2196F3', '#4CAF50', 'Exploit', 'Typing ratio')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm3_typing.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()

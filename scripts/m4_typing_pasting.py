#!/usr/bin/env python3
"""
M4: Typing + Pasting Combined Exploit
=======================================
Exploit = typing answer OR paste window (±10s around paste event)
Explore = everything else (browsing Wikipedia)

Difference from M3: M3 only counts typing as exploit.
M7 also counts paste windows — the period where the participant
reads, selects, copies, and pastes content into the answer box.

Output: m4_typing_pasting.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from helpers import load_trials, get_pids_and_trials, setup_figure, finish_timeline, sorted_ratio_panel, OUTPUT_DIR

PASTE_WINDOW = 10  # seconds before and after paste event


def get_exploit_intervals(tr):
    """Merge typing intervals and paste windows into unified exploit intervals."""
    intervals = []

    # Typing intervals (already detected)
    for bs, be in tr['typing_intervals']:
        intervals.append((bs, be))

    # Paste windows: ±PASTE_WINDOW seconds around each paste
    for pt in tr['paste_times']:
        ps = max(0, pt - PASTE_WINDOW)
        pe = min(tr['duration'], pt + PASTE_WINDOW)
        intervals.append((ps, pe))

    if not intervals:
        return []

    # Merge overlapping intervals
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def main():
    print("[M4] Typing + Pasting (combined exploit)")
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    n = len(pids)
    print(f"  {len(trials)} trials from {n} participants")

    fig, ax_top, ax_bl, ax_br = setup_figure(n,
        f'Explore vs. Exploit — Typing + Pasting (window ±{PASTE_WINDOW}s)\n'
        'Purple = Exploit (typing + paste window) | Green = Explore (browsing)')

    pid_exploit = {p: [] for p in pids}
    pid_explore = {p: [] for p in pids}

    for pi, pid in enumerate(pids):
        y_c = (n - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y = y_c + (0.18 if ti == 0 else -0.18)

            # Full bar = explore (green)
            ax_top.barh(y, tr['duration'], height=0.28, color='#4CAF50', alpha=0.5,
                        edgecolor='#2E7D32', linewidth=0.3, zorder=2)

            # Merged exploit intervals (purple)
            exploit_intervals = get_exploit_intervals(tr)
            exploit_total = 0
            for s, e in exploit_intervals:
                ax_top.barh(y, e - s, left=s, height=0.28,
                            color='#7B1FA2', alpha=0.8, edgecolor='#4A148C',
                            linewidth=0.5, zorder=4)
                exploit_total += e - s

            # Typing-only segments (blue, thinner, on top for visibility)
            for bs, be in tr['typing_intervals']:
                ax_top.barh(y, be - bs, left=bs, height=0.16,
                            color='#2196F3', alpha=0.9, edgecolor='none',
                            linewidth=0, zorder=5)

            # Page ticks
            for pv in tr['page_visits']:
                ax_top.plot([pv['start'], pv['start']], [y - 0.14, y + 0.14],
                            color='black', linewidth=0.4, zorder=6, alpha=0.5)

            # Paste triangles
            for pt in tr['paste_times']:
                ax_top.scatter(pt, y, marker='v', s=40, color='#F44336',
                               zorder=7, edgecolors='black', linewidths=0.4, alpha=0.9)

            pct = exploit_total / tr['duration'] * 100 if tr['duration'] > 0 else 0
            ax_top.text(tr['duration'] + 8, y, f"{tr['domain'][:6]} ({pct:.0f}%)",
                        fontsize=6, va='center', color='#616161')
            pid_exploit[pid].append(exploit_total)
            pid_explore[pid].append(tr['duration'] - exploit_total)

    finish_timeline(ax_top, pids)
    ax_top.legend(handles=[
        mpatches.Patch(color='#4CAF50', alpha=0.5, label='Explore (browsing)'),
        mpatches.Patch(color='#7B1FA2', alpha=0.8, label=f'Exploit (typing + paste ±{PASTE_WINDOW}s)'),
        mpatches.Patch(color='#2196F3', alpha=0.9, label='Typing only (within exploit)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#F44336',
               markersize=8, label='Paste event'),
        Line2D([0], [0], color='black', linewidth=0.5, label='Page open'),
    ], fontsize=8, loc='lower right', framealpha=0.9, ncol=5)

    # Bottom left: stacked bars
    ex_m = [np.mean(pid_exploit[p]) for p in pids]
    br_m = [np.mean(pid_explore[p]) for p in pids]
    x = np.arange(n)
    ax_bl.bar(x, ex_m, color='#7B1FA2', label='Exploit (type+paste)', edgecolor='gray', width=0.6)
    ax_bl.bar(x, br_m, bottom=ex_m, color='#4CAF50', label='Explore (browsing)', edgecolor='gray', width=0.6)
    ax_bl.set_xticks(x)
    ax_bl.set_xticklabels([f"P{p}" for p in pids], fontsize=8, rotation=45, ha='right')
    ax_bl.set_ylabel('Mean time (s)')
    ax_bl.set_title('Avg Exploit (type+paste) vs Explore')
    ax_bl.legend(fontsize=7)

    # Bottom right: sorted ratio
    ratios = [ex_m[i] / (ex_m[i] + br_m[i]) if (ex_m[i] + br_m[i]) > 0 else 0 for i in range(n)]
    sorted_ratio_panel(ax_br, pids, ratios, '#7B1FA2', '#4CAF50', 'Exploit', 'Typing+Paste ratio')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm4_typing_pasting.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()

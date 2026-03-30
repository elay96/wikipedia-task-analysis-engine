#!/usr/bin/env python3
"""
M18: Explore/Exploit — Typing/Pasting Per-Page Binary
=====================================================
Like M3 layout but each page visit is colored as a single block:
  - Exploit (blue) = page had typing or pasting
  - Explore (green) = browsing only
Output: m18_typing_binary.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from helpers import load_trials, get_pids_and_trials, setup_figure, finish_timeline, sorted_ratio_panel, OUTPUT_DIR


EXPLOIT_COLOR = '#2196F3'
EXPLORE_COLOR = '#4CAF50'


def page_had_typing_or_paste(pv, typing_intervals, paste_times):
    ps, pe = pv['start'], pv['end']
    for bs, be in typing_intervals:
        if bs < pe and be > ps:
            return True
    for pt in paste_times:
        if ps <= pt <= pe:
            return True
    return False


def main():
    print("[M17] Typing/Pasting per-page binary")
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    n = len(pids)
    print(f"  {len(trials)} trials from {n} participants")

    fig, ax_top, ax_bl, ax_br = setup_figure(n,
        'Explore vs. Exploit — Typing/Pasting Per Page\n'
        'Blue = Exploit (page with typing/paste) | Green = Explore (browsing only)')

    pid_exploit = {p: [] for p in pids}
    pid_explore = {p: [] for p in pids}

    for pi, pid in enumerate(pids):
        y_c = (n - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y = y_c + (0.18 if ti == 0 else -0.18)
            exploit_time = 0
            exploit_pages = 0
            explore_pages = 0

            for pv in tr['page_visits']:
                is_exploit = page_had_typing_or_paste(
                    pv, tr['typing_intervals'], tr['paste_times'])
                color = EXPLOIT_COLOR if is_exploit else EXPLORE_COLOR
                alpha = 0.85 if is_exploit else 0.5
                edge = '#0D47A1' if is_exploit else '#2E7D32'

                ax_top.barh(y, pv['duration'], left=pv['start'], height=0.28,
                            color=color, alpha=alpha, edgecolor=edge,
                            linewidth=0.5, zorder=3)

                if is_exploit:
                    exploit_time += pv['duration']
                    exploit_pages += 1
                else:
                    explore_pages += 1

                # Page tick
                ax_top.plot([pv['start'], pv['start']], [y - 0.14, y + 0.14],
                            color='black', linewidth=0.4, zorder=5, alpha=0.5)


            explore_time = tr['duration'] - exploit_time
            ax_top.text(tr['duration'] + 8, y,
                        f"{exploit_pages} exploit, {explore_pages} explore",
                        fontsize=6, va='center', color='#616161')
            pid_exploit[pid].append(exploit_time)
            pid_explore[pid].append(explore_time)

    finish_timeline(ax_top, pids)
    ax_top.legend(handles=[
        mpatches.Patch(color=EXPLORE_COLOR, alpha=0.5, label='Explore (browsing only)'),
        mpatches.Patch(color=EXPLOIT_COLOR, alpha=0.85, label='Exploit (typing/paste)'),

        Line2D([0], [0], color='black', linewidth=0.5, label='Page open'),
    ], fontsize=8, loc='lower right', framealpha=0.9, ncol=4)

    ex_m = [np.mean(pid_exploit[p]) for p in pids]
    br_m = [np.mean(pid_explore[p]) for p in pids]
    x = np.arange(n)
    ax_bl.bar(x, ex_m, color=EXPLOIT_COLOR, label='Exploit (typing/paste)', edgecolor='gray', width=0.6)
    ax_bl.bar(x, br_m, bottom=ex_m, color=EXPLORE_COLOR, label='Explore (browsing)', edgecolor='gray', width=0.6)
    ax_bl.set_xticks(x)
    ax_bl.set_xticklabels([f"P{p}" for p in pids], fontsize=8, rotation=45, ha='right')
    ax_bl.set_ylabel('Mean time (s)')
    ax_bl.set_title('Avg Exploit (typing/paste) vs Explore (browsing)')
    ax_bl.legend(fontsize=7)

    ratios = [ex_m[i] / (ex_m[i] + br_m[i]) if (ex_m[i] + br_m[i]) > 0 else 0 for i in range(n)]
    sorted_ratio_panel(ax_br, pids, ratios, EXPLOIT_COLOR, EXPLORE_COLOR, 'Exploit', 'Typing/paste page ratio')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm18_typing_binary.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()

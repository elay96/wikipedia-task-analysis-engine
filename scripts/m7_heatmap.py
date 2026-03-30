#!/usr/bin/env python3
"""
M7: 3-Measure Agreement Heatmap
=================================
Per-second scoring from 3 independent measures:
  M1: Time on page > 20s → exploit
  M2: Typing answer → exploit
  M3: Link-click navigation → exploit

Score 0 = all explore (blue), 3 = all exploit (orange)
Output: m7_heatmap.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from helpers import load_trials, get_pids_and_trials, finish_timeline, OUTPUT_DIR


def main():
    print("[M7] Heatmap (3-measure agreement)")
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    n = len(pids)
    print(f"  {len(trials)} trials from {n} participants")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'ee', ['#1565C0', '#64B5F6', '#E0E0E0', '#FFB74D', '#E65100'], N=256)

    fig = plt.figure(figsize=(22, 20))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 0.15, 1.2], hspace=0.35, wspace=0.3)
    ax_top = fig.add_subplot(gs[0, :])
    ax_cbar = fig.add_subplot(gs[1, :])
    ax_bl = fig.add_subplot(gs[2, 0])
    ax_br = fig.add_subplot(gs[2, 1])

    fig.suptitle(
        'Explore-Exploit Heatmap — Agreement of 3 Measures\n'
        'M1: Time >20s = exploit | M2: Typing = exploit | M3: Link-click = exploit',
        fontsize=14, fontweight='bold')

    pid_scores = {p: [] for p in pids}

    for pi, pid in enumerate(pids):
        y_c = (n - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y = y_c + (0.20 if ti == 0 else -0.20)
            dur = int(tr['duration'])
            m1 = np.zeros(dur)
            m2 = np.zeros(dur)
            m3 = np.zeros(dur)
            for pv in tr['page_visits']:
                s, e = max(0, int(pv['start'])), min(dur, int(pv['end']))
                if pv['duration'] > 20:
                    m1[s:e] = 1
                if pv['nav_type'] == 'link_click':
                    m3[s:e] = 1
            for bs, be in tr['typing_intervals']:
                s, e = max(0, int(bs)), min(dur, int(be) + 1)
                m2[s:e] = 1
            score = m1 + m2 + m3
            pid_scores[pid].append(np.mean(score))

            chunk = 3
            for s in range(0, dur, chunk):
                e = min(s + chunk, dur)
                val = np.mean(score[s:e]) / 3.0
                ax_top.barh(y, e - s, left=s, height=0.32, color=cmap(val),
                            edgecolor='none', zorder=2)
            ax_top.barh(y, dur, height=0.32, color='none',
                        edgecolor='#9E9E9E', linewidth=0.4, zorder=3)
            for pv in tr['page_visits']:
                ax_top.plot([pv['start'], pv['start']], [y - 0.16, y + 0.16],
                            color='black', linewidth=0.4, zorder=5, alpha=0.5)
            for pt in tr['paste_times']:
                ax_top.scatter(pt, y, marker='v', s=30, color='#F44336',
                               zorder=6, edgecolors='black', linewidths=0.3, alpha=0.8)
            ms = np.mean(score)
            ax_top.text(dur + 8, y, f"{tr['domain'][:6]} ({ms:.1f}/3)",
                        fontsize=6, va='center', color='#616161')

    finish_timeline(ax_top, pids)
    ax_top.set_title('Per-Second Agreement Heatmap', fontsize=12)
    ax_top.legend(handles=[
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#F44336', markersize=8, label='Paste'),
        Line2D([0], [0], color='black', linewidth=0.5, label='Page open'),
    ], fontsize=8, loc='lower right', framealpha=0.9)

    # Colorbar
    norm = mcolors.Normalize(vmin=0, vmax=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
    cb.set_ticks([0, 1, 2, 3])
    cb.set_ticklabels(['0/3 All Explore', '1/3 Mostly Explore',
                        '2/3 Mostly Exploit', '3/3 All Exploit'])
    cb.ax.tick_params(labelsize=9)

    # Bottom left: per-user mean score
    means = [np.mean(pid_scores[p]) for p in pids]
    x = np.arange(n)
    ax_bl.bar(x, means, color=[cmap(m / 3) for m in means], edgecolor='gray', width=0.7)
    ax_bl.axhline(np.mean(means), color='red', linestyle='--', linewidth=1.2,
                  label=f'Mean: {np.mean(means):.2f}/3')
    ax_bl.set_xticks(x)
    ax_bl.set_xticklabels([f"P{p}" for p in pids], fontsize=8, rotation=45, ha='right')
    ax_bl.set_ylabel('Mean score (0-3)')
    ax_bl.set_ylim(0, 3)
    ax_bl.set_title('Per-Participant Mean Exploit Score')
    ax_bl.legend(fontsize=8)

    # Bottom right: stacked distribution
    level_pcts = {l: [] for l in range(4)}
    for pid in pids:
        all_s = []
        for tr in pid_trials[pid]:
            dur = int(tr['duration'])
            m1 = np.zeros(dur); m2 = np.zeros(dur); m3 = np.zeros(dur)
            for pv in tr['page_visits']:
                s, e = max(0, int(pv['start'])), min(dur, int(pv['end']))
                if pv['duration'] > 20: m1[s:e] = 1
                if pv['nav_type'] == 'link_click': m3[s:e] = 1
            for bs, be in tr['typing_intervals']:
                s, e = max(0, int(bs)), min(dur, int(be) + 1)
                m2[s:e] = 1
            all_s.extend(m1 + m2 + m3)
        total = len(all_s)
        for l in range(4):
            level_pcts[l].append(sum(1 for v in all_s if v == l) / total * 100)

    bottom = np.zeros(n)
    for l in range(4):
        ax_br.bar(x, level_pcts[l], bottom=bottom, color=cmap(l / 3.0),
                  edgecolor='gray', width=0.7, label=f'{l}/3')
        bottom += np.array(level_pcts[l])
    ax_br.set_xticks(x)
    ax_br.set_xticklabels([f"P{p}" for p in pids], fontsize=8, rotation=45, ha='right')
    ax_br.set_ylabel('% of time')
    ax_br.set_title('Agreement Level Distribution')
    ax_br.set_ylim(0, 100)
    ax_br.legend(fontsize=7, title='Exploit votes')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm7_heatmap.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()

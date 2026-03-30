#!/usr/bin/env python3
"""
M6: Page Visits Timeline + Distribution
=========================================
Timeline of page visits (color-coded, numbered) per participant.
Bottom panels: histogram, per-participant mean, box plot.
No scipy dependency — normality test uses numpy-only approximation.
Output: m6_pages.png
"""

import numpy as np
import matplotlib.pyplot as plt
from helpers import load_trials, get_pids_and_trials, finish_timeline, OUTPUT_DIR


def shapiro_wilk_approx(data):
    """Simple normality check using skewness and kurtosis (no scipy needed)."""
    n = len(data)
    if n < 8:
        return None, None
    x = np.array(data, dtype=float)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return None, None
    z = (x - m) / s
    skew = np.mean(z ** 3)
    kurt = np.mean(z ** 4) - 3
    # D'Agostino-Pearson-like: rough p-value based on skew+kurt
    k2 = (skew ** 2) + (kurt ** 2) / 4
    # Approximate p from chi2(2) — use exponential approximation
    p_approx = np.exp(-k2 / 2)
    return k2, min(p_approx, 1.0)


def main():
    print("[M6] Page visits")
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    n = len(pids)
    print(f"  {len(trials)} trials from {n} participants")

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1], hspace=0.35, wspace=0.3)
    ax_top = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_box = fig.add_subplot(gs[1, 2])

    fig.suptitle('Page Visits per Participant — Timeline & Distribution',
                 fontsize=14, fontweight='bold')

    pid_page_counts = {p: [] for p in pids}
    all_page_counts = []

    for pi, pid in enumerate(pids):
        y_c = (n - pi - 1)
        for ti, tr in enumerate(pid_trials[pid]):
            y = y_c + (0.18 if ti == 0 else -0.18)
            pvs = tr['page_visits']
            n_pages = len(pvs)
            pid_page_counts[pid].append(n_pages)
            all_page_counts.append(n_pages)

            ax_top.barh(y, tr['duration'], height=0.28, color='#FAFAFA',
                        edgecolor='#E0E0E0', linewidth=0.3, zorder=1)

            page_palette = ['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC',
                            '#EF5350', '#26C6DA', '#8D6E63', '#78909C']
            for i, pv in enumerate(pvs):
                c = page_palette[i % len(page_palette)]
                ax_top.barh(y, pv['duration'], left=pv['start'], height=0.28,
                            color=c, alpha=0.6, edgecolor='gray', linewidth=0.3, zorder=3)
                mid = pv['start'] + pv['duration'] / 2
                if pv['duration'] > 15:
                    ax_top.text(mid, y, str(i + 1), ha='center', va='center',
                                fontsize=6, fontweight='bold', color='white', zorder=4)

            ax_top.text(tr['duration'] + 8, y,
                        f"{tr['domain'][:6]} ({n_pages} pp)",
                        fontsize=6, va='center', color='#616161')

    finish_timeline(ax_top, pids)
    ax_top.set_title('Page Visits Timeline (numbered, color-coded)', fontsize=12)

    # Bottom left: histogram
    ax_hist.hist(all_page_counts, bins=range(1, max(all_page_counts) + 2),
                 color='#42A5F5', edgecolor='gray', alpha=0.8, align='left')
    mean_p = np.mean(all_page_counts)
    median_p = np.median(all_page_counts)
    ax_hist.axvline(mean_p, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_p:.1f}')
    ax_hist.axvline(median_p, color='blue', linestyle=':', linewidth=1.5, label=f'Median: {median_p:.1f}')
    ax_hist.set_xlabel('Pages per trial')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Distribution of Page Counts')
    ax_hist.legend(fontsize=8)

    # Normality test (no scipy)
    if len(all_page_counts) >= 8:
        k2, p = shapiro_wilk_approx(all_page_counts)
        if k2 is not None:
            norm_text = f'Skew+Kurt test: K2={k2:.3f}, p~{p:.3f}\n'
            norm_text += 'Normal' if p > 0.05 else 'Non-normal'
            ax_hist.text(0.95, 0.95, norm_text, transform=ax_hist.transAxes,
                         fontsize=8, va='top', ha='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Bottom middle: per-participant mean
    means = [np.mean(pid_page_counts[p]) for p in pids]
    x = np.arange(n)
    ax_bar.bar(x, means, color='#42A5F5', edgecolor='gray', width=0.6)
    ax_bar.axhline(mean_p, color='red', linestyle='--', linewidth=1, label=f'Grand mean: {mean_p:.1f}')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"P{p}" for p in pids], fontsize=8, rotation=45, ha='right')
    ax_bar.set_ylabel('Mean pages per trial')
    ax_bar.set_title('Per-Participant Mean Page Count')
    ax_bar.legend(fontsize=8)

    # Bottom right: box plot
    data_for_box = [pid_page_counts[p] for p in pids]
    bp = ax_box.boxplot(data_for_box, labels=[f"P{p}" for p in pids],
                        patch_artist=True, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor('#42A5F5')
        patch.set_alpha(0.6)
    ax_box.set_ylabel('Pages per trial')
    ax_box.set_title('Page Count Distribution per Participant')
    ax_box.tick_params(axis='x', labelsize=8, labelrotation=45)
    for i, p in enumerate(pids):
        jitter = np.random.uniform(-0.15, 0.15, len(pid_page_counts[p]))
        ax_box.scatter(np.array([i + 1] * len(pid_page_counts[p])) + jitter,
                       pid_page_counts[p], color='#EF5350', s=30, zorder=5, alpha=0.8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm6_pages.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    main()

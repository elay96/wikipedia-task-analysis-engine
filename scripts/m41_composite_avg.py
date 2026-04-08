#!/usr/bin/env python3
"""
M41: Composite Switching Score (Simple Average)
================================================
Composite = mean(SR Time, SR Topic, SR Typing), averaged across domains first.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from helpers import load_trials, OUTPUT_DIR

BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'
BORDER_COLOR = '#30363d'
MUTED_COLOR = '#8b949e'

SR_TIME_COLOR = '#FF9800'
SR_TOPIC_COLOR = '#4FC3F7'
SR_TYPING_COLOR = '#81C784'
COMPOSITE_COLOR = 'white'


def load_conditions():
    trials = load_trials()
    return {tr['pid']: tr['condition'] for tr in trials}


def load_and_average(path, col_name):
    df = pd.read_csv(path)
    return df.groupby('participant_id')['switch_rate'].mean().rename(col_name)


def build_composite():
    sr_time = load_and_average(OUTPUT_DIR / 'm34_switch_time.csv', 'sr_time')
    sr_topic = load_and_average(OUTPUT_DIR / 'm35_switch_lda.csv', 'sr_topic')
    sr_typing = load_and_average(OUTPUT_DIR / 'm36_switch_typing.csv', 'sr_typing')

    df = pd.concat([sr_time, sr_topic, sr_typing], axis=1).reset_index()
    df['composite'] = df[['sr_time', 'sr_topic', 'sr_typing']].mean(axis=1)

    conditions = load_conditions()
    df['condition'] = df['participant_id'].map(conditions)

    return df[['participant_id', 'condition', 'sr_time', 'sr_topic', 'sr_typing', 'composite']]


def apply_dark_theme(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=LABEL_COLOR)
    ax.xaxis.label.set_color(LABEL_COLOR)
    ax.yaxis.label.set_color(LABEL_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_COLOR)
    ax.grid(color=GRID_COLOR, linestyle='--', linewidth=0.5)


def plot_individual_panel(ax, df_sorted):
    pids = df_sorted['participant_id'].astype(str).tolist()
    x = np.arange(len(pids))

    ax.scatter(x, df_sorted['sr_time'], color=SR_TIME_COLOR, marker='o', s=60, zorder=3, label='SR Time')
    ax.scatter(x, df_sorted['sr_topic'], color=SR_TOPIC_COLOR, marker='s', s=60, zorder=3, label='SR Topic')
    ax.scatter(x, df_sorted['sr_typing'], color=SR_TYPING_COLOR, marker='D', s=60, zorder=3, label='SR Typing')
    ax.scatter(x, df_sorted['composite'], color=COMPOSITE_COLOR, marker='*', s=120, zorder=4, label='Composite')

    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Switch Rate', color=LABEL_COLOR)
    ax.set_title('Switch Rates per Participant (sorted by composite)', color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    apply_dark_theme(ax)


def plot_distribution_panel(ax, df):
    composites = df['composite'].dropna()
    mean_val = composites.mean()
    median_val = composites.median()

    ax.hist(composites, bins=10, color=SR_TOPIC_COLOR, edgecolor=BORDER_COLOR, alpha=0.85)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.3f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=1.5, label=f'Median = {median_val:.3f}')

    ax.text(0.97, 0.95, f'mean = {mean_val:.3f}\nstd = {composites.std():.3f}',
            transform=ax.transAxes, ha='right', va='top',
            color=TEXT_COLOR, fontsize=9,
            bbox=dict(facecolor=BG_COLOR, edgecolor=BORDER_COLOR, boxstyle='round,pad=0.3'))

    ax.set_xlabel('Composite Score', color=LABEL_COLOR)
    ax.set_ylabel('Count', color=LABEL_COLOR)
    ax.set_title('Composite Score Distribution', color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    apply_dark_theme(ax)


def print_summary(df):
    print("\nPer-participant composite scores:")
    print(df[['participant_id', 'condition', 'sr_time', 'sr_topic', 'sr_typing', 'composite']].to_string(index=False, float_format='{:.3f}'.format))

    c = df['composite']
    print(f"\nSummary stats:")
    print(f"  N         = {len(c)}")
    print(f"  Mean      = {c.mean():.4f}")
    print(f"  Std       = {c.std():.4f}")
    print(f"  Min       = {c.min():.4f}")
    print(f"  Max       = {c.max():.4f}")
    print(f"  Range     = {c.max() - c.min():.4f}")


def main():
    df = build_composite()
    df_sorted = df.sort_values('composite').reset_index(drop=True)

    print_summary(df_sorted)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        'M41: Composite Switching Score = Mean(SR Time, SR Topic, SR Typing)',
        color=TEXT_COLOR, fontsize=13, fontweight='bold'
    )

    plot_individual_panel(ax_left, df_sorted)
    plot_distribution_panel(ax_right, df)

    plt.tight_layout()
    out_img = OUTPUT_DIR / 'm41_composite_avg.png'
    plt.savefig(out_img, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"\nPlot saved: {out_img}")

    out_csv = OUTPUT_DIR / 'm41_composite_avg.csv'
    df_sorted.to_csv(out_csv, index=False)
    print(f"CSV saved:  {out_csv}")


if __name__ == '__main__':
    main()

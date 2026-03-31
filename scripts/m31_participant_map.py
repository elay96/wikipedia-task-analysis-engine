#!/usr/bin/env python3
"""
m31_participant_map.py — Per-participant behavior map in PCA space.

Each participant is one point (mean PC1, PC2) with a 1-std error ellipse.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from m29_pca_raw import (
    load_topic_distances, build_feature_matrix,
    FEATURE_NAMES, BG_COLOR, TEXT_COLOR, LABEL_COLOR, BORDER_COLOR,
)
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

QUADRANT_COLOR = '#555555'
QUADRANT_FONTSIZE = 9
CROSSHAIR_COLOR = '#444444'

QUADRANT_LABELS = [
    (1,  1,  "Deep engagement\ndifferent topics",          'right', 'top'),
    (-1,  1, "Quick browsing\ndifferent topics\n= EXPLORATION", 'left',  'top'),
    (1, -1,  "Deep engagement\nsimilar topics\n= EXPLOITATION", 'right', 'bottom'),
    (-1, -1, "Quick browsing\nsimilar topics",              'left',  'bottom'),
]


def group_scores_by_participant(scores, meta):
    pids = sorted(set(m['pid'] for m in meta))
    pid_scores = {p: [] for p in pids}
    for m, row in zip(meta, scores):
        pid_scores[m['pid']].append(row[:2])
    return pids, {p: np.array(v) for p, v in pid_scores.items()}


def make_participant_map(pids, pid_score_map, var_explained, output_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Crosshairs
    ax.axhline(0, color=CROSSHAIR_COLOR, linestyle='--', linewidth=1, zorder=1)
    ax.axvline(0, color=CROSSHAIR_COLOR, linestyle='--', linewidth=1, zorder=1)

    cmap = plt.get_cmap('tab20')
    n = max(len(pids), 1)

    for i, pid in enumerate(pids):
        pts = pid_score_map[pid]
        color = cmap(i / n)

        mean_pc1 = pts[:, 0].mean()
        mean_pc2 = pts[:, 1].mean()
        std_pc1 = pts[:, 0].std() if len(pts) > 1 else 0.0
        std_pc2 = pts[:, 1].std() if len(pts) > 1 else 0.0

        # Error ellipse (1 std)
        ellipse = Ellipse(
            xy=(mean_pc1, mean_pc2),
            width=2 * std_pc1,
            height=2 * std_pc2,
            facecolor=color,
            edgecolor='none',
            alpha=0.15,
            zorder=2,
        )
        ax.add_patch(ellipse)

        # Participant point
        ax.scatter(
            mean_pc1, mean_pc2,
            s=150,
            color=color,
            edgecolors='white',
            linewidths=1,
            zorder=3,
        )

        # Label offset slightly up-right
        ax.text(
            mean_pc1 + 0.05, mean_pc2 + 0.05,
            f'P{pid}',
            color=color,
            fontsize=8,
            zorder=4,
        )

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Use data coordinates padded from axis limits after autoscale
    ax.autoscale()
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    x_pad = (xl[1] - xl[0]) * 0.04
    y_pad = (yl[1] - yl[0]) * 0.04

    for xsign, ysign, label, ha, va in QUADRANT_LABELS:
        x = xl[1] - x_pad if xsign > 0 else xl[0] + x_pad
        y = yl[1] - y_pad if ysign > 0 else yl[0] + y_pad
        ax.text(
            x, y, label,
            color=QUADRANT_COLOR,
            fontsize=QUADRANT_FONTSIZE,
            style='italic',
            ha=ha, va=va,
            zorder=1,
        )

    pc1_pct = var_explained[0] * 100
    pc2_pct = var_explained[1] * 100

    ax.set_xlabel(f'PC1: Engagement ({pc1_pct:.1f}%)', color=LABEL_COLOR, fontsize=12)
    ax.set_ylabel(f'PC2: Semantic Breadth ({pc2_pct:.1f}%)', color=LABEL_COLOR, fontsize=12)
    ax.set_title(
        'Individual Foraging Strategies — Mean Position in PCA Space',
        color=TEXT_COLOR, fontsize=14,
    )

    ax.tick_params(colors=LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    trials = load_trials()
    pids, pid_trials = get_pids_and_trials(trials)
    tm = load_topic_distances()

    print("Building feature matrix...")
    X, meta = build_feature_matrix(pids, pid_trials, tm)
    print(f"  Observations: {X.shape[0]}, Features: {X.shape[1]}")

    print("Running PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X_scaled)

    print("Variance explained:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {v*100:.1f}%")

    pids_sorted, pid_score_map = group_scores_by_participant(scores, meta)

    print("Plotting participant map...")
    make_participant_map(
        pids_sorted,
        pid_score_map,
        pca.explained_variance_ratio_,
        OUTPUT_DIR / 'm31_participant_map.png',
    )
    print("Done.")


if __name__ == '__main__':
    main()

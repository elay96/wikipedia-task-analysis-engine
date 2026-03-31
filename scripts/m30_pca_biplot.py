#!/usr/bin/env python3
"""
m30_pca_biplot.py — Enhanced PCA biplot colored by domain with quadrant labels.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from m29_pca_raw import (
    load_topic_distances, build_feature_matrix,
    FEATURE_NAMES, BG_COLOR, TEXT_COLOR, LABEL_COLOR, BORDER_COLOR,
)
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN_COLORS = {
    'art_history': '#E91E63',
    'ecology':     '#4CAF50',
    'psychology':  '#9C27B0',
    'economics':   '#FF9800',
}
ARROW_COLOR = '#FF9800'

QUADRANT_LABELS = {
    'top_right':    "Deep engagement\ndifferent topics",
    'top_left':     "Quick browsing\ndifferent topics\n= EXPLORATION",
    'bottom_right': "Deep engagement\nsimilar topics\n= EXPLOITATION",
    'bottom_left':  "Quick browsing\nsimilar topics",
}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_enhanced_biplot(scores, loadings, explained_var, meta, output_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # --- Dots colored by domain ---
    for m, (x, y) in zip(meta, scores[:, :2]):
        color = DOMAIN_COLORS.get(m['domain'], '#888888')
        ax.scatter(x, y, color=color, alpha=0.6, s=35,
                   edgecolors='white', linewidths=0.3, zorder=2)

    # --- Feature loading arrows ---
    x_range = scores[:, 0].max() - scores[:, 0].min()
    y_range = scores[:, 1].max() - scores[:, 1].min()
    scale = 0.4 * max(x_range, y_range)

    for j, name in enumerate(FEATURE_NAMES):
        lx = loadings[0, j] * scale
        ly = loadings[1, j] * scale
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=3))
        ax.text(lx * 1.12, ly * 1.12, name,
                color=ARROW_COLOR, fontsize=12, fontweight='bold',
                ha='center', va='center')

    # --- Zero lines ---
    ax.axhline(0, color=BORDER_COLOR, linewidth=0.8, zorder=1)
    ax.axvline(0, color=BORDER_COLOR, linewidth=0.8, zorder=1)

    # --- Quadrant labels ---
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Recompute from data to ensure accuracy before axis limits are set
    xs = scores[:, 0]
    ys = scores[:, 1]
    pad_x = (xs.max() - xs.min()) * 0.04
    pad_y = (ys.max() - ys.min()) * 0.04

    quadrant_positions = {
        'top_right':    (xs.max() - pad_x, ys.max() - pad_y, 'right', 'top'),
        'top_left':     (xs.min() + pad_x, ys.max() - pad_y, 'left',  'top'),
        'bottom_right': (xs.max() - pad_x, ys.min() + pad_y, 'right', 'bottom'),
        'bottom_left':  (xs.min() + pad_x, ys.min() + pad_y, 'left',  'bottom'),
    }

    for key, (qx, qy, ha, va) in quadrant_positions.items():
        ax.text(qx, qy, QUADRANT_LABELS[key],
                color='#888888', fontsize=10, fontstyle='italic',
                ha=ha, va=va, zorder=3,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_COLOR,
                          edgecolor='none', alpha=0.7))

    # --- Axis labels with variance explained ---
    pct1 = explained_var[0] * 100
    pct2 = explained_var[1] * 100
    ax.set_xlabel(f'PC1: Engagement ({pct1:.1f}%)', color=LABEL_COLOR, fontsize=13)
    ax.set_ylabel(f'PC2: Semantic Breadth ({pct2:.1f}%)', color=LABEL_COLOR, fontsize=13)
    ax.set_title('Behavioral Space — PCA on Foraging Signals',
                 color=TEXT_COLOR, fontsize=15, fontweight='bold', pad=16)

    # --- Styling ---
    ax.tick_params(colors=LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_COLOR)

    # --- Domain legend ---
    handles = [
        mpatches.Patch(color=color, label=domain.replace('_', ' ').title())
        for domain, color in DOMAIN_COLORS.items()
    ]
    legend = ax.legend(
        handles=handles,
        title='Domain',
        title_fontsize=11,
        fontsize=10,
        facecolor=BG_COLOR,
        edgecolor=BORDER_COLOR,
        labelcolor=TEXT_COLOR,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        framealpha=0.9,
    )
    legend.get_title().set_color(TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    print("Explained variance:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {v*100:.1f}%")

    print("Loadings:")
    for i, comp in enumerate(pca.components_):
        parts = ', '.join(f'{FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(len(FEATURE_NAMES)))
        print(f"  PC{i+1}: {parts}")

    print("Generating plot...")
    plot_enhanced_biplot(
        scores, pca.components_, pca.explained_variance_ratio_,
        meta, OUTPUT_DIR / 'm30_pca_biplot.png'
    )

    print("Done.")


if __name__ == '__main__':
    main()

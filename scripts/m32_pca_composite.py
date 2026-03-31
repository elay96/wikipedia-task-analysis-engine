#!/usr/bin/env python3
"""
m32_pca_composite.py — 3-panel composite PCA figure for paper/presentation.

Panels:
  A (left):   Scree plot
  B (center): Biplot with domain coloring and quadrant labels
  C (right):  Per-participant mean positions with ellipses
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
    FEATURE_NAMES,
)
from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

# Light theme overrides (replaces dark constants from m29)
BG_COLOR     = '#ffffff'
TEXT_COLOR   = '#1a1a2e'
LABEL_COLOR  = '#4a4a4a'
BORDER_COLOR = '#cccccc'

DOMAIN_COLORS = {
    'art_history': '#E91E63',
    'ecology':     '#4CAF50',
    'psychology':  '#9C27B0',
    'economics':   '#FF9800',
}

QUADRANT_COLOR = '#999999'
QUADRANT_FS = 9

# ---------------------------------------------------------------------------
# Theme helper
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=LABEL_COLOR)
    ax.xaxis.label.set_color(LABEL_COLOR)
    ax.yaxis.label.set_color(LABEL_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_COLOR)


# ---------------------------------------------------------------------------
# Panel A: Scree
# ---------------------------------------------------------------------------

def draw_scree(ax, evr):
    pcs = [f'PC{i+1}' for i in range(len(evr))]
    pct = evr * 100
    cumulative = np.cumsum(pct)

    bars = ax.bar(pcs, pct, color='#4FC3F7', zorder=2)
    ax.plot(pcs, cumulative, color='#FF9800', marker='o', linewidth=2,
            zorder=3, label='Cumulative')

    for bar, val in zip(bars, pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.1f}%',
            ha='center', va='bottom', color=TEXT_COLOR, fontsize=10,
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel('Variance Explained (%)', color=LABEL_COLOR)
    ax.set_title('A. Variance Explained', color=TEXT_COLOR, fontsize=12)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, edgecolor=BORDER_COLOR)
    _style_ax(ax)


# ---------------------------------------------------------------------------
# Shared quadrant helpers
# ---------------------------------------------------------------------------

def _draw_crosshairs(ax):
    ax.axhline(0, color='#cccccc', linewidth=0.8, zorder=1)
    ax.axvline(0, color='#cccccc', linewidth=0.8, zorder=1)


def _draw_quadrant_labels(ax, pc1_positive_is_engagement, pc2_positive_is_breadth):
    """
    Place quadrant labels based on actual loading signs.

    pc1_positive_is_engagement: True if +PC1 = more engagement (time/writing)
    pc2_positive_is_breadth:    True if +PC2 = more topic breadth (distance)
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # offsets: 10% from centre toward corner
    x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.97
    x_neg = xlim[0] + (xlim[1] - xlim[0]) * 0.03
    y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.97
    y_neg = ylim[0] + (ylim[1] - ylim[0]) * 0.03

    def _label(x, y, ha, va, txt):
        ax.text(x, y, txt, ha=ha, va=va, color=QUADRANT_COLOR,
                fontsize=QUADRANT_FS, style='italic', zorder=5,
                multialignment='center')

    if pc1_positive_is_engagement and pc2_positive_is_breadth:
        _label(x_pos, y_pos, 'right', 'top',    "Deep engagement\ndifferent topics")
        _label(x_neg, y_pos, 'left',  'top',    "Quick browsing\ndifferent topics")
        _label(x_pos, y_neg, 'right', 'bottom', "Deep engagement\nsimilar topics")
        _label(x_neg, y_neg, 'left',  'bottom', "Quick browsing\nsimilar topics")
    else:
        # fallback: swap as needed
        eng_pos = "Deep engagement" if pc1_positive_is_engagement else "Quick browsing"
        eng_neg = "Quick browsing"  if pc1_positive_is_engagement else "Deep engagement"
        br_pos  = "different topics" if pc2_positive_is_breadth else "similar topics"
        br_neg  = "similar topics"   if pc2_positive_is_breadth else "different topics"
        _label(x_pos, y_pos, 'right', 'top',    f"{eng_pos}\n{br_pos}")
        _label(x_neg, y_pos, 'left',  'top',    f"{eng_neg}\n{br_pos}")
        _label(x_pos, y_neg, 'right', 'bottom', f"{eng_pos}\n{br_neg}")
        _label(x_neg, y_neg, 'left',  'bottom', f"{eng_neg}\n{br_neg}")


# ---------------------------------------------------------------------------
# Panel B: Biplot
# ---------------------------------------------------------------------------

def draw_biplot(ax, scores, loadings, meta, evr, loading_signs):
    pc1_pos_eng, pc2_pos_breadth = loading_signs

    # Scatter by domain
    for domain, color in DOMAIN_COLORS.items():
        mask = [m['domain'] == domain for m in meta]
        xs = scores[mask, 0]
        ys = scores[mask, 1]
        ax.scatter(xs, ys, color=color, alpha=0.5, s=25, zorder=2,
                   label=domain.replace('_', ' ').title())

    # Feature loading arrows
    x_range = scores[:, 0].max() - scores[:, 0].min()
    y_range = scores[:, 1].max() - scores[:, 1].min()
    scale = 0.4 * max(x_range, y_range)

    for j, name in enumerate(FEATURE_NAMES):
        lx = loadings[0, j] * scale
        ly = loadings[1, j] * scale
        ax.annotate(
            '', xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2.5),
            zorder=4,
        )
        ax.text(lx * 1.12, ly * 1.12, name, color='#FF9800', fontsize=8,
                ha='center', va='center', zorder=5)

    _draw_crosshairs(ax)

    var1, var2 = evr[0] * 100, evr[1] * 100
    ax.set_xlabel(f'PC1: Engagement ({var1:.1f}%)', color=LABEL_COLOR)
    ax.set_ylabel(f'PC2: Semantic Breadth ({var2:.1f}%)', color=LABEL_COLOR)
    ax.set_title('B. Behavioral Space', color=TEXT_COLOR, fontsize=12)

    legend = ax.legend(
        facecolor=BG_COLOR, labelcolor=TEXT_COLOR,
        edgecolor=BORDER_COLOR, fontsize=8, loc='upper left',
    )

    _style_ax(ax)
    _draw_quadrant_labels(ax, pc1_pos_eng, pc2_pos_breadth)


# ---------------------------------------------------------------------------
# Panel C: Participant map
# ---------------------------------------------------------------------------

def draw_participant_map(ax, scores, meta, evr, loading_signs):
    pc1_pos_eng, pc2_pos_breadth = loading_signs

    pids = sorted(set(m['pid'] for m in meta))
    cmap = plt.get_cmap('tab20')
    n = max(len(pids), 1)

    meta_arr = np.array([(m['pid'],) for m in meta],
                        dtype=[('pid', object)])

    for idx, pid in enumerate(pids):
        mask = np.array([m['pid'] == pid for m in meta])
        pid_scores = scores[mask, :2]
        if len(pid_scores) < 2:
            continue

        color = cmap(idx / n)
        mean_x, mean_y = pid_scores[:, 0].mean(), pid_scores[:, 1].mean()
        std_x,  std_y  = pid_scores[:, 0].std(),  pid_scores[:, 1].std()

        # Ellipse (1 std)
        ell = Ellipse(
            xy=(mean_x, mean_y),
            width=2 * std_x,
            height=2 * std_y,
            angle=0,
            facecolor=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=0.8,
            zorder=2,
        )
        ax.add_patch(ell)

        ax.scatter(mean_x, mean_y, color=color, s=120,
                   edgecolors='#666666', linewidths=0.8, zorder=4)
        ax.text(mean_x + 0.05, mean_y + 0.05, f'P{pid}',
                color=TEXT_COLOR, fontsize=7, zorder=5)

    _draw_crosshairs(ax)

    var1, var2 = evr[0] * 100, evr[1] * 100
    ax.set_xlabel(f'PC1: Engagement ({var1:.1f}%)', color=LABEL_COLOR)
    ax.set_ylabel(f'PC2: Semantic Breadth ({var2:.1f}%)', color=LABEL_COLOR)
    ax.set_title('C. Individual Strategies', color=TEXT_COLOR, fontsize=12)

    _style_ax(ax)
    _draw_quadrant_labels(ax, pc1_pos_eng, pc2_pos_breadth)


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
    evr = pca.explained_variance_ratio_

    print("Explained variance:", [f"{v*100:.1f}%" for v in evr])
    print("Loadings (PC1, PC2):")
    for j, name in enumerate(FEATURE_NAMES):
        print(f"  {name}: PC1={pca.components_[0, j]:+.3f}  PC2={pca.components_[1, j]:+.3f}")

    # Determine sign conventions from actual loadings
    # PC1: positive direction = engagement if time_on_page and writing load positively
    time_idx    = 0  # 'Time on page (s)'
    writing_idx = 2  # 'Writing amount (s)'
    dist_idx    = 1  # 'Topic distance (JSD)'

    pc1_time_writing = (pca.components_[0, time_idx] + pca.components_[0, writing_idx])
    pc2_dist = pca.components_[1, dist_idx]

    pc1_positive_is_engagement = pc1_time_writing > 0
    pc2_positive_is_breadth    = pc2_dist > 0

    print(f"PC1 positive = engagement: {pc1_positive_is_engagement}")
    print(f"PC2 positive = breadth:    {pc2_positive_is_breadth}")

    loading_signs = (pc1_positive_is_engagement, pc2_positive_is_breadth)

    # Convert meta to numpy boolean masks friendly form
    meta_domains = np.array([m['domain'] for m in meta])

    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        "PCA Analysis — Behavioral Foraging Signals",
        color=TEXT_COLOR, fontweight='bold', fontsize=16, y=1.01,
    )

    draw_scree(axes[0], evr)
    draw_biplot(axes[1], scores, pca.components_, meta, evr, loading_signs)
    draw_participant_map(axes[2], scores, meta, evr, loading_signs)

    fig.tight_layout()

    out_path = OUTPUT_DIR / 'm32_pca_composite.png'
    fig.savefig(out_path, dpi=200, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()

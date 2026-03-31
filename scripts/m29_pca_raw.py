#!/usr/bin/env python3
"""
m29_pca_raw.py — PCA on 3 raw continuous features per page visit.

Features:
  1. Time on page (seconds)
  2. Topic distance from previous page (JSD)
  3. Writing amount (seconds)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
TOPIC_MODEL_PATH = SCRIPT_DIR / '..' / 'data' / 'topic_model.json'
PASTE_WEIGHT = 5.0
FEATURE_NAMES = ['Time on page (s)', 'Topic distance (JSD)', 'Writing amount (s)']

BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
BORDER_COLOR = '#30363d'
BAR_COLOR = '#4FC3F7'
LINE_COLOR = '#FF9800'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_topic_distances():
    with open(TOPIC_MODEL_PATH, 'r') as f:
        tm = json.load(f)
    return tm


def get_topic_dist(tm, slug_a, slug_b):
    key_ab = f"{slug_a}|||{slug_b}"
    key_ba = f"{slug_b}|||{slug_a}"
    d = tm['distances']
    if key_ab in d:
        return d[key_ab]
    if key_ba in d:
        return d[key_ba]
    return np.nan


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_writing_amount(pv, typing_intervals, paste_times):
    typing_dur = 0.0
    for bs, be in typing_intervals:
        start = max(bs, pv['start'])
        end = min(be, pv['end'])
        if end > start:
            typing_dur += end - start

    n_pastes = sum(1 for pt in paste_times if pv['start'] <= pt <= pv['end'])
    return typing_dur + n_pastes * PASTE_WEIGHT


def build_feature_matrix(pids, pid_trials, tm):
    rows = []
    meta = []

    for pid in pids:
        for tr in pid_trials[pid]:
            pvs = tr['page_visits']
            for i in range(1, len(pvs)):
                pv = pvs[i]
                prev = pvs[i - 1]

                time_on_page = pv['duration']
                topic_dist = get_topic_dist(tm, prev['title'], pv['title'])
                if np.isnan(topic_dist):
                    continue
                writing = compute_writing_amount(pv, tr['typing_intervals'], tr['paste_times'])

                rows.append([time_on_page, topic_dist, writing])
                meta.append({
                    'pid': pid,
                    'trial': tr['trial'],
                    'domain': tr['domain'],
                    'page_idx': i,
                    'title': pv['title'],
                })

    X = np.array(rows, dtype=float)
    return X, meta


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _apply_dark_theme(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=LABEL_COLOR)
        ax.xaxis.label.set_color(LABEL_COLOR)
        ax.yaxis.label.set_color(LABEL_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)


def plot_scree(explained_variance_ratio, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    pcs = [f'PC{i+1}' for i in range(len(explained_variance_ratio))]
    pct = explained_variance_ratio * 100
    cumulative = np.cumsum(pct)

    bars = ax.bar(pcs, pct, color=BAR_COLOR, zorder=2)
    ax.plot(pcs, cumulative, color=LINE_COLOR, marker='o', linewidth=2, zorder=3, label='Cumulative')

    for bar, val in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', color=TEXT_COLOR, fontsize=10)

    ax.set_ylabel('Variance Explained (%)', color=LABEL_COLOR)
    ax.set_title('Scree Plot — PCA on Raw Continuous Signals', color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, edgecolor=BORDER_COLOR)
    ax.set_ylim(0, 110)

    _apply_dark_theme(fig, [ax])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_biplot(scores, loadings, meta, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    pids = sorted(set(m['pid'] for m in meta))
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    cmap = plt.get_cmap('tab20')

    for m, (x, y) in zip(meta, scores[:, :2]):
        color = cmap(pid_to_idx[m['pid']] / max(len(pids), 1))
        ax.scatter(x, y, color=color, alpha=0.5, s=20, zorder=2)

    # Scale arrows to data range
    x_range = scores[:, 0].max() - scores[:, 0].min()
    y_range = scores[:, 1].max() - scores[:, 1].min()
    scale = 0.4 * max(x_range, y_range)

    for j, name in enumerate(FEATURE_NAMES):
        lx = loadings[0, j] * scale
        ly = loadings[1, j] * scale
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=LINE_COLOR, lw=2.5))
        ax.text(lx * 1.1, ly * 1.1, name, color=LINE_COLOR, fontsize=9,
                ha='center', va='center')

    ax.axhline(0, color=BORDER_COLOR, linewidth=0.5)
    ax.axvline(0, color=BORDER_COLOR, linewidth=0.5)
    ax.set_xlabel('PC1', color=LABEL_COLOR)
    ax.set_ylabel('PC2', color=LABEL_COLOR)
    ax.set_title('Biplot — PC1 vs PC2 with Feature Loadings', color=TEXT_COLOR)

    _apply_dark_theme(fig, [ax])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pc1_scores(scores, meta, output_path):
    pids = sorted(set(m['pid'] for m in meta))
    pid_scores = {p: [] for p in pids}
    for m, s in zip(meta, scores[:, 0]):
        pid_scores[m['pid']].append(s)

    fig, ax = plt.subplots(figsize=(14, 6))

    data_for_box = [pid_scores[p] for p in pids]
    bp = ax.boxplot(data_for_box,
                    patch_artist=True,
                    medianprops=dict(color=LINE_COLOR, linewidth=2),
                    whiskerprops=dict(color=LABEL_COLOR),
                    capprops=dict(color=LABEL_COLOR),
                    flierprops=dict(markerfacecolor=LABEL_COLOR, marker='o', markersize=3, alpha=0.5),
                    boxprops=dict(facecolor='#4FC3F7', alpha=0.6))

    for patch in bp['boxes']:
        patch.set_facecolor(BAR_COLOR)
        patch.set_alpha(0.6)

    ax.axhline(0, color=LABEL_COLOR, linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xticks(range(1, len(pids) + 1))
    ax.set_xticklabels([f'P{p}' for p in pids], rotation=45, ha='right', color=LABEL_COLOR)
    ax.set_ylabel('PC1 Score (Exploit \u2190 \u2192 Explore)', color=LABEL_COLOR)
    ax.set_title('PC1 Score Distribution per Participant', color=TEXT_COLOR)

    _apply_dark_theme(fig, [ax])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOR)
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

    print("\nRaw feature stats:")
    for j, name in enumerate(FEATURE_NAMES):
        col = X[:, j]
        print(f"  {name}: mean={col.mean():.3f}, std={col.std():.3f}, "
              f"min={col.min():.3f}, max={col.max():.3f}")

    print("\nRunning PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    scores = pca.fit_transform(X_scaled)

    print("\nExplained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)")

    print("\nFeature loadings (components):")
    for i, comp in enumerate(pca.components_):
        parts = ', '.join(f'{FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(len(FEATURE_NAMES)))
        print(f"  PC{i+1}: {parts}")

    print("\nGenerating plots...")
    plot_scree(pca.explained_variance_ratio_, OUTPUT_DIR / 'm29_pca_scree.png')
    plot_biplot(scores, pca.components_, meta, OUTPUT_DIR / 'm29_pca_biplot.png')
    plot_pc1_scores(scores, meta, OUTPUT_DIR / 'm29_pca_scores.png')

    print("\nDone.")


if __name__ == '__main__':
    main()

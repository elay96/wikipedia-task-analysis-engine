#!/usr/bin/env python3
"""
M48: PCA on Raw Switch Counts
==============================
Computes raw switch counts (not divided by N-1) for time, topic, and typing,
then runs PCA with and without z-scoring. Compares with rate-based M38/M40.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR
from m18_typing_binary import page_had_typing_or_paste

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

THRESHOLD_S = 60.0

BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'
BORDER_COLOR = '#30363d'
MUTED_COLOR = '#8b949e'
BAR_COLOR = '#4FC3F7'
LINE_COLOR = '#FF9800'
ARROW_COLOR = '#FF9800'
FEATURE_NAMES = ['Count Time', 'Count Topic', 'Count Typing']
CONDITION_COLORS = {'high-creativity': '#4FC3F7', 'low-creativity': '#F06292'}
DEFAULT_DOT_COLOR = '#4FC3F7'


def compute_switch_count(labels):
    if len(labels) < 2:
        return np.nan
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def load_lda_assignments():
    with open(DATA_DIR / 'topic_model.json') as f:
        tm = json.load(f)
    return {slug: int(np.argmax(dist)) for slug, dist in tm['topic_distributions'].items()}


def build_counts_df(trials):
    pids, pid_trials = get_pids_and_trials(trials)
    lda_assignments = load_lda_assignments()

    rows = []
    for pid in pids:
        for tr in pid_trials[pid]:
            domain = tr['domain']
            if domain == 'practice':
                continue

            pvs = tr['page_visits']

            # count_time: exploit/explore via 60s threshold
            time_labels = ['exploit' if pv['duration'] > THRESHOLD_S else 'explore' for pv in pvs]
            count_time = compute_switch_count(time_labels)

            # count_topic: LDA argmax topic changes
            topic_labels = [lda_assignments.get(pv['title'], -1) for pv in pvs]
            count_topic = compute_switch_count(topic_labels)

            # count_typing: typing/no-typing binary
            typing_labels = [
                page_had_typing_or_paste(pv, tr['typing_intervals'], tr['paste_times'])
                for pv in pvs
            ]
            count_typing = compute_switch_count(typing_labels)

            rows.append({
                'participant_id': pid,
                'domain': domain,
                'count_time': count_time,
                'count_topic': count_topic,
                'count_typing': count_typing,
            })

    return pd.DataFrame(rows)


def plot_row(fig, axes, pca, scores, pids, conditions, title, pct):
    ax_scree, ax_biplot, ax_table = axes
    cumulative = np.cumsum(pct)
    loadings = pca.components_
    pc_labels = [f'PC{i+1}' for i in range(len(pct))]

    # Scree plot
    ax_scree.set_facecolor(BG_COLOR)
    bars = ax_scree.bar(pc_labels, pct, color=BAR_COLOR, zorder=2)
    ax_scree.plot(pc_labels, cumulative, color=LINE_COLOR, marker='o', linewidth=2, zorder=3)
    for bar, val in zip(bars, pct):
        ax_scree.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f'{val:.1f}%', ha='center', va='bottom', color=TEXT_COLOR, fontsize=10)
    ax_scree.set_ylabel('Variance Explained (%)', color=LABEL_COLOR)
    ax_scree.set_title(title + '\nScree Plot', color=TEXT_COLOR, fontweight='bold')
    ax_scree.set_ylim(0, 110)
    ax_scree.tick_params(colors=MUTED_COLOR)
    ax_scree.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax_scree.spines.values():
        spine.set_color(BORDER_COLOR)

    # Biplot
    ax_biplot.set_facecolor(BG_COLOR)
    for i, pid in enumerate(pids):
        cond = conditions.get(pid, '')
        color = CONDITION_COLORS.get(cond, DEFAULT_DOT_COLOR)
        ax_biplot.scatter(scores[i, 0], scores[i, 1], color=color, s=50, alpha=0.8,
                          edgecolors='white', linewidth=0.5, zorder=3)
        ax_biplot.annotate(f'P{pid}', (scores[i, 0], scores[i, 1]),
                           fontsize=7, color=LABEL_COLOR, ha='left', va='bottom',
                           xytext=(4, 4), textcoords='offset points')

    x_range = scores[:, 0].max() - scores[:, 0].min() if len(scores) > 1 else 1
    y_range = scores[:, 1].max() - scores[:, 1].min() if len(scores) > 1 else 1
    scale = 0.4 * max(x_range, y_range)

    for j, name in enumerate(FEATURE_NAMES):
        lx = loadings[0, j] * scale
        ly = loadings[1, j] * scale
        ax_biplot.annotate('', xy=(lx, ly), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2.5))
        ax_biplot.text(lx * 1.15, ly * 1.15, name, color=ARROW_COLOR, fontsize=10,
                       ha='center', va='center', fontweight='bold')

    ax_biplot.axhline(0, color=BORDER_COLOR, linewidth=0.5)
    ax_biplot.axvline(0, color=BORDER_COLOR, linewidth=0.5)
    ax_biplot.set_xlabel(f'PC1 ({pct[0]:.1f}%)', color=LABEL_COLOR)
    ax_biplot.set_ylabel(f'PC2 ({pct[1]:.1f}%)', color=LABEL_COLOR)
    ax_biplot.set_title('Biplot', color=TEXT_COLOR, fontweight='bold')
    ax_biplot.tick_params(colors=MUTED_COLOR)
    for spine in ax_biplot.spines.values():
        spine.set_color(BORDER_COLOR)

    # Loadings table
    ax_table.set_facecolor(BG_COLOR)
    ax_table.axis('off')
    ax_table.set_title('Loadings', color=TEXT_COLOR, fontweight='bold')

    col_labels = [f'PC{i+1}' for i in range(3)]
    cell_text = []
    for j in range(3):
        row = [f'{loadings[i, j]:+.3f}' for i in range(min(3, len(loadings)))]
        cell_text.append(row)

    table = ax_table.table(
        cellText=cell_text,
        rowLabels=FEATURE_NAMES,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for key, cell in table.get_celld().items():
        cell.set_facecolor(BG_COLOR)
        cell.set_edgecolor(BORDER_COLOR)
        cell.set_text_props(color=TEXT_COLOR)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M48] PCA on Switch Counts')

    trials = load_trials()
    conditions = {tr['pid']: tr['condition'] for tr in trials}

    counts_df = build_counts_df(trials)

    # Average across domains per participant
    avg_df = counts_df.groupby('participant_id')[['count_time', 'count_topic', 'count_typing']].mean().reset_index()
    avg_df = avg_df.dropna()
    pids = avg_df['participant_id'].values

    print(f'  N={len(pids)} participants')

    # Correlation matrix
    print('\n  Correlation matrix (raw counts, before PCA):')
    corr = avg_df[['count_time', 'count_topic', 'count_typing']].corr()
    print(corr.to_string())

    X = avg_df[['count_time', 'count_topic', 'count_typing']].values

    # PCA without z-score
    pca_raw = PCA(n_components=3)
    scores_raw = pca_raw.fit_transform(X)
    pct_raw = pca_raw.explained_variance_ratio_ * 100

    print('\n  PCA (no z-score) explained variance:')
    for i, v in enumerate(pct_raw):
        print(f'    PC{i+1}: {v:.1f}%')
    print('  Loadings:')
    for i, comp in enumerate(pca_raw.components_):
        parts = ', '.join(f'{FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(3))
        print(f'    PC{i+1}: {parts}')

    # PCA with z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_z = PCA(n_components=3)
    scores_z = pca_z.fit_transform(X_scaled)
    pct_z = pca_z.explained_variance_ratio_ * 100

    print('\n  PCA (z-scored) explained variance:')
    for i, v in enumerate(pct_z):
        print(f'    PC{i+1}: {v:.1f}%')
    print('  Loadings:')
    for i, comp in enumerate(pca_z.components_):
        parts = ', '.join(f'{FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(3))
        print(f'    PC{i+1}: {parts}')

    # Visualization: 2 rows x 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(22, 14),
                             gridspec_kw={'width_ratios': [1, 1.3, 0.7]})
    fig.patch.set_facecolor(BG_COLOR)

    plot_row(fig, axes[0], pca_raw, scores_raw, pids, conditions,
             'M48: PCA on Switch Counts (No Z-Score)', pct_raw)
    plot_row(fig, axes[1], pca_z, scores_z, pids, conditions,
             'M48: PCA on Switch Counts (Z-Scored)', pct_z)

    plt.tight_layout()
    out_png = OUTPUT_DIR / 'm48_pca_counts.png'
    plt.savefig(out_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out_png}')

    # Save CSV
    scores_df = pd.DataFrame({
        'participant_id': pids,
        'condition': [conditions.get(p, '') for p in pids],
        'count_time': avg_df['count_time'].values,
        'count_topic': avg_df['count_topic'].values,
        'count_typing': avg_df['count_typing'].values,
        'PC1_raw': scores_raw[:, 0],
        'PC2_raw': scores_raw[:, 1],
        'PC3_raw': scores_raw[:, 2],
        'PC1_zscore': scores_z[:, 0],
        'PC2_zscore': scores_z[:, 1],
        'PC3_zscore': scores_z[:, 2],
    })
    csv_path = OUTPUT_DIR / 'm48_scores_counts.csv'
    scores_df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()

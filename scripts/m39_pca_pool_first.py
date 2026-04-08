#!/usr/bin/env python3
"""
M39: PCA on Switch Rates - Pool First (Approach B)
===================================================
Runs PCA on all rows pooled (participant x domain), then averages PC scores
per participant. Biplot shows per-participant averaged scores.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from helpers import load_trials, OUTPUT_DIR

BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'
BORDER_COLOR = '#30363d'
MUTED_COLOR = '#8b949e'
BAR_COLOR = '#4FC3F7'
LINE_COLOR = '#FF9800'
ARROW_COLOR = '#FF9800'

FEATURE_NAMES = ['SR Time', 'SR Topic', 'SR Typing']

CONDITION_COLORS = {
    'high-creativity': '#4FC3F7',
    'low-creativity': '#F06292',
}
DEFAULT_DOT_COLOR = '#4FC3F7'


def load_merged():
    time_df = pd.read_csv(OUTPUT_DIR / 'm34_switch_time.csv').rename(columns={'switch_rate': 'sr_time'})
    topic_df = pd.read_csv(OUTPUT_DIR / 'm35_switch_lda.csv').rename(columns={'switch_rate': 'sr_topic'})
    typing_df = pd.read_csv(OUTPUT_DIR / 'm36_switch_typing.csv').rename(columns={'switch_rate': 'sr_typing'})
    return time_df.merge(topic_df, on=['participant_id', 'domain']).merge(typing_df, on=['participant_id', 'domain'])


def plot_results(avg_scores_df, pca, conditions, output_path):
    pids = avg_scores_df['participant_id'].values
    scores = avg_scores_df[['PC1', 'PC2', 'PC3']].values
    pct = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(pct)
    loadings = pca.components_
    pc_labels = [f'PC{i+1}' for i in range(len(pct))]

    fig, (ax_scree, ax_biplot, ax_table) = plt.subplots(
        1, 3, figsize=(22, 7),
        gridspec_kw={'width_ratios': [1, 1.3, 0.7]},
    )
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        'M39: PCA on Switch Rates - Approach B (Pool First, then Average)',
        fontsize=14, color=TEXT_COLOR, fontweight='bold', y=0.99,
    )

    # Panel 1: Scree plot
    ax_scree.set_facecolor(BG_COLOR)
    bars = ax_scree.bar(pc_labels, pct, color=BAR_COLOR, zorder=2)
    ax_scree.plot(pc_labels, cumulative, color=LINE_COLOR, marker='o', linewidth=2, zorder=3)
    for bar, val in zip(bars, pct):
        ax_scree.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f'{val:.1f}%', ha='center', va='bottom', color=TEXT_COLOR, fontsize=10)
    ax_scree.set_ylabel('Variance Explained (%)', color=LABEL_COLOR)
    ax_scree.set_title('Scree Plot (pooled rows)', color=TEXT_COLOR, fontweight='bold')
    ax_scree.set_ylim(0, 110)
    ax_scree.tick_params(colors=MUTED_COLOR)
    ax_scree.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax_scree.spines.values():
        spine.set_color(BORDER_COLOR)

    # Panel 2: Biplot (averaged per-participant scores)
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
    ax_biplot.set_title('Biplot (averaged PC scores per participant)', color=TEXT_COLOR, fontweight='bold')
    ax_biplot.tick_params(colors=MUTED_COLOR)
    for spine in ax_biplot.spines.values():
        spine.set_color(BORDER_COLOR)

    # Panel 3: Loadings table
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

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M39] PCA on Switch Rates - Approach B (Pool First)')

    trials = load_trials()
    conditions = {tr['pid']: tr['condition'] for tr in trials}

    merged = load_merged().dropna()

    # Step 1: PCA on all pooled rows (~40 rows x 3 features)
    X = merged[['sr_time', 'sr_topic', 'sr_typing']].values
    n_components = min(3, X.shape[0], X.shape[1])

    pca = PCA(n_components=n_components)
    scores_pooled = pca.fit_transform(X)

    print(f'  Pooled rows: {len(X)}')
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f'  PC{i+1}: {var*100:.1f}%')
    print('  Loadings:')
    for i, comp in enumerate(pca.components_):
        parts = ', '.join(f'{FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(len(FEATURE_NAMES)))
        print(f'    PC{i+1}: {parts}')

    # Step 2: attach scores back, average per participant
    scores_df = pd.DataFrame(scores_pooled, columns=[f'PC{i+1}' for i in range(n_components)])
    scores_df.insert(0, 'participant_id', merged['participant_id'].values)
    scores_df.insert(1, 'domain', merged['domain'].values)

    avg_scores_df = scores_df.groupby('participant_id')[[f'PC{i+1}' for i in range(n_components)]].mean().reset_index()
    avg_scores_df.insert(1, 'condition', [conditions.get(p, '') for p in avg_scores_df['participant_id'].values])

    print(f'  Averaged to {len(avg_scores_df)} participants')

    # Visualization (biplot shows averaged scores)
    plot_results(avg_scores_df, pca, conditions, OUTPUT_DIR / 'm39_pca_pool_first.png')

    # Save averaged scores CSV
    csv_path = OUTPUT_DIR / 'm39_scores_pool_first.csv'
    avg_scores_df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
M48: PCA Comparison - Switch Rates vs Switch Counts
=====================================================
Computes both rates (transitions / N-1) and raw counts for time, topic,
and typing signals, then runs PCA on each (both z-scored).
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
CONDITION_COLORS = {'high-creativity': '#4FC3F7', 'low-creativity': '#F06292'}
DEFAULT_DOT_COLOR = '#4FC3F7'

RATE_FEATURE_NAMES = ['SR Time', 'SR Topic', 'SR Typing']
COUNT_FEATURE_NAMES = ['Count Time', 'Count Topic', 'Count Typing']


def load_lda_assignments():
    with open(DATA_DIR / 'topic_model.json') as f:
        tm = json.load(f)
    return {slug: int(np.argmax(dist)) for slug, dist in tm['topic_distributions'].items()}


def compute_switches(labels):
    """Return (count, rate) tuple."""
    if len(labels) < 2:
        return np.nan, np.nan
    count = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    rate = count / (len(labels) - 1)
    return count, rate


def build_features_df(trials):
    pids, pid_trials = get_pids_and_trials(trials)
    lda_assignments = load_lda_assignments()

    rows = []
    for pid in pids:
        for tr in pid_trials[pid]:
            domain = tr['domain']
            if domain == 'practice':
                continue

            pvs = tr['page_visits']

            time_labels = ['exploit' if pv['duration'] > THRESHOLD_S else 'explore' for pv in pvs]
            count_time, rate_time = compute_switches(time_labels)

            topic_labels = [lda_assignments.get(pv['title'], -1) for pv in pvs]
            count_topic, rate_topic = compute_switches(topic_labels)

            typing_labels = [
                page_had_typing_or_paste(pv, tr['typing_intervals'], tr['paste_times'])
                for pv in pvs
            ]
            count_typing, rate_typing = compute_switches(typing_labels)

            rows.append({
                'participant_id': pid,
                'domain': domain,
                'rate_time': rate_time,
                'rate_topic': rate_topic,
                'rate_typing': rate_typing,
                'count_time': count_time,
                'count_topic': count_topic,
                'count_typing': count_typing,
            })

    return pd.DataFrame(rows)


def plot_row(fig, gs, pca, scores, pids, conditions, row_title, pct, feature_names):
    ax_scree = fig.add_subplot(gs[0])
    ax_biplot = fig.add_subplot(gs[1])
    ax_table = fig.add_subplot(gs[2])

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
    ax_scree.set_title('Scree Plot', color=TEXT_COLOR, fontweight='bold')
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

    for j, name in enumerate(feature_names):
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
        rowLabels=feature_names,
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

    # Row title annotation - bold and prominent
    ax_scree.set_title(row_title + '\nScree Plot', color=TEXT_COLOR, fontweight='bold', fontsize=13)

    return ax_scree, ax_biplot, ax_table


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M48] PCA Comparison: Switch Rates vs Switch Counts')

    trials = load_trials()
    conditions = {tr['pid']: tr['condition'] for tr in trials}

    features_df = build_features_df(trials)

    rate_cols = ['rate_time', 'rate_topic', 'rate_typing']
    count_cols = ['count_time', 'count_topic', 'count_typing']
    all_cols = rate_cols + count_cols

    avg_df = features_df.groupby('participant_id')[all_cols].mean().reset_index()
    avg_df = avg_df.dropna()
    pids = avg_df['participant_id'].values

    print(f'  N={len(pids)} participants')

    # Correlation matrices
    corr_rates = avg_df[rate_cols].corr()
    corr_counts = avg_df[count_cols].corr()

    print('\n  Correlation matrix (rates):')
    print(corr_rates.to_string())
    print('\n  Correlation matrix (counts):')
    print(corr_counts.to_string())

    # PCA on rates (z-scored)
    scaler = StandardScaler()
    X_rates = scaler.fit_transform(avg_df[rate_cols].values)
    pca_rates = PCA(n_components=3)
    scores_rates = pca_rates.fit_transform(X_rates)
    pct_rates = pca_rates.explained_variance_ratio_ * 100

    # PCA on counts (z-scored)
    X_counts = scaler.fit_transform(avg_df[count_cols].values)
    pca_counts = PCA(n_components=3)
    scores_counts = pca_counts.fit_transform(X_counts)
    pct_counts = pca_counts.explained_variance_ratio_ * 100

    print('\n  Explained variance (rates, z-scored):')
    for i, v in enumerate(pct_rates):
        print(f'    PC{i+1}: {v:.1f}%')
    print('  Loadings:')
    for i, comp in enumerate(pca_rates.components_):
        parts = ', '.join(f'{RATE_FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(3))
        print(f'    PC{i+1}: {parts}')

    print('\n  Explained variance (counts, z-scored):')
    for i, v in enumerate(pct_counts):
        print(f'    PC{i+1}: {v:.1f}%')
    print('  Loadings:')
    for i, comp in enumerate(pca_counts.components_):
        parts = ', '.join(f'{COUNT_FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(3))
        print(f'    PC{i+1}: {parts}')

    print('\n  Comparison:')
    print(f'    Rates  PC1: {pct_rates[0]:.1f}%  |  Counts PC1: {pct_counts[0]:.1f}%')
    print(f'    Rates  PC2: {pct_rates[1]:.1f}%  |  Counts PC2: {pct_counts[1]:.1f}%')
    print(f'    Rates  PC3: {pct_rates[2]:.1f}%  |  Counts PC3: {pct_counts[2]:.1f}%')

    # Figure layout
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG_COLOR)

    gs_top = fig.add_gridspec(1, 3, top=0.90, bottom=0.52,
                              width_ratios=[1, 1.3, 0.7], wspace=0.3)
    gs_bot = fig.add_gridspec(1, 3, top=0.43, bottom=0.05,
                              width_ratios=[1, 1.3, 0.7], wspace=0.3)

    plot_row(fig, gs_top, pca_rates, scores_rates, pids, conditions,
             'PCA on Switch Rates (transitions / N-1, LDA topics)', pct_rates, RATE_FEATURE_NAMES)
    plot_row(fig, gs_bot, pca_counts, scores_counts, pids, conditions,
             'PCA on Switch Counts (raw transitions, LDA topics)', pct_counts, COUNT_FEATURE_NAMES)

    fig.suptitle('M48: PCA Comparison - Switch Rates vs Switch Counts (both z-scored, LDA topics)',
                 color=TEXT_COLOR, fontsize=16, fontweight='bold', y=0.99)

    out_png = OUTPUT_DIR / 'm48_pca_counts.png'
    plt.savefig(out_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out_png}')

    # Save CSV
    scores_df = pd.DataFrame({
        'participant_id': pids,
        'condition': [conditions.get(p, '') for p in pids],
        'PC1_rates': scores_rates[:, 0],
        'PC2_rates': scores_rates[:, 1],
        'PC3_rates': scores_rates[:, 2],
        'PC1_counts': scores_counts[:, 0],
        'PC2_counts': scores_counts[:, 1],
        'PC3_counts': scores_counts[:, 2],
    })
    csv_path = OUTPUT_DIR / 'm48_scores_counts.csv'
    scores_df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()

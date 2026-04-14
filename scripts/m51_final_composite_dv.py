#!/usr/bin/env python3
"""
M51: Final Composite DV - Switch Counts + PCA (No Z-Score)
==========================================================
Same pipeline as M50 but on updated dataset (Game_new.csv).
  1. Exclusion criteria (idle >=50%, <3 pages, 3 SD outliers)
  2. Raw switch counts for time, topic (LDA), and typing
  3. PCA on raw counts (no standardization)

Outputs: PNG, CSV, PDF report.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.decomposition import PCA

from scipy import stats as sp_stats

from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR
from m18_typing_binary import page_had_typing_or_paste

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

THRESHOLD_S = 60.0
IDLE_THRESHOLD_PCT = 50.0
MIN_PAGE_VISITS = 3
OUTLIER_SD = 3

FEATURE_NAMES = ['Count Time', 'Count Topic', 'Count Typing']

# --- Light mode palette ---
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1a1a1a'
LABEL_COLOR = '#333333'
GRID_COLOR = '#E0E0E0'
BORDER_COLOR = '#CCCCCC'
MUTED_COLOR = '#666666'
BAR_COLOR = '#1976D2'
LINE_COLOR = '#E65100'
ARROW_COLOR = '#E65100'
CONDITION_COLORS = {
    'diffuse': '#1976D2', 'clumpy': '#C62828',
    'high-creativity': '#1976D2', 'low-creativity': '#C62828',
}
DEFAULT_DOT_COLOR = '#1976D2'
EXCLUDED_COLOR = '#E53935'
KEPT_COLOR = '#43A047'
WARN_COLOR = '#F9A825'

# --- Meaningful actions for idle detection (from M49) ---
MEANINGFUL_ACTIONS = [
    'article_open', 'search', 'link_click', 'back_navigation', 'paste',
]
SNAPSHOT_ACTIONS = ['answer_snapshot', 'answer_snapshot_cursor_leave']


def load_lda_assignments():
    with open(DATA_DIR / 'topic_model.json') as f:
        tm = json.load(f)
    return {slug: int(np.argmax(dist)) for slug, dist in tm['topic_distributions'].items()}


def compute_switch_count(labels):
    if len(labels) < 2:
        return np.nan
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def compute_idle_pct_for_trial(trial_events_df, t0, t_end):
    """Compute idle % after last meaningful event for a single trial."""
    total_sec = (t_end - t0).total_seconds()
    if total_sec <= 0:
        return np.nan

    meaningful = trial_events_df[trial_events_df['Action'].isin(MEANINGFUL_ACTIONS)]

    snapshots = trial_events_df[trial_events_df['Action'].isin(SNAPSHOT_ACTIONS)].copy()
    if len(snapshots) > 1:
        snapshots['prev_len'] = snapshots['AnswerLength'].shift(1)
        writing_snapshots = snapshots[snapshots['AnswerLength'] != snapshots['prev_len']]
    else:
        writing_snapshots = snapshots.iloc[:0]

    all_meaningful = pd.concat([meaningful, writing_snapshots]).sort_values('Time')
    all_meaningful = all_meaningful[
        (all_meaningful['Time'] >= t0) & (all_meaningful['Time'] <= t_end)
    ]

    if len(all_meaningful) == 0:
        return 100.0

    last_event = all_meaningful['Time'].iloc[-1]
    idle_after = (t_end - last_event).total_seconds()
    return (idle_after / total_sec) * 100


def build_question_data(trials):
    """Build per-question rows with counts and exclusion info."""
    lda_assignments = load_lda_assignments()
    pids, pid_trials = get_pids_and_trials(trials)

    rows = []
    for pid in pids:
        for tr in pid_trials[pid]:
            domain = tr['domain']
            if domain == 'practice':
                continue

            pvs = tr['page_visits']
            n_pages = len(pvs)

            # Idle %
            events_df = tr['events']
            t0_abs = tr['t0']
            t_end_abs = t0_abs + pd.Timedelta(seconds=tr['duration'])
            idle_pct = compute_idle_pct_for_trial(events_df, t0_abs, t_end_abs)

            # Exclusion flags
            excluded_pages = n_pages < MIN_PAGE_VISITS
            excluded_idle = idle_pct >= IDLE_THRESHOLD_PCT

            # Switch counts
            time_labels = ['exploit' if pv['duration'] > THRESHOLD_S else 'explore' for pv in pvs]
            count_time = compute_switch_count(time_labels)

            topic_labels = [lda_assignments.get(pv['title'], -1) for pv in pvs]
            count_topic = compute_switch_count(topic_labels)

            typing_labels = [
                page_had_typing_or_paste(pv, tr['typing_intervals'], tr['paste_times'])
                for pv in pvs
            ]
            count_typing = compute_switch_count(typing_labels)

            rows.append({
                'participant_id': pid,
                'domain': domain,
                'n_pages': n_pages,
                'idle_pct': idle_pct,
                'excluded_pages': excluded_pages,
                'excluded_idle': excluded_idle,
                'count_time': count_time,
                'count_topic': count_topic,
                'count_typing': count_typing,
            })

    return pd.DataFrame(rows)


def apply_exclusions(question_df):
    """Apply all exclusion criteria and return clean participant-level df."""
    n_total_questions = len(question_df)
    n_participants_before = question_df['participant_id'].nunique()

    # Step 1: exclude questions by idle and page count
    excluded_mask = question_df['excluded_pages'] | question_df['excluded_idle']
    clean_questions = question_df[~excluded_mask].copy()

    excl_pages = question_df['excluded_pages'].sum()
    excl_idle = question_df['excluded_idle'].sum()
    excl_both = (question_df['excluded_pages'] & question_df['excluded_idle']).sum()
    excl_total = excluded_mask.sum()

    print(f'  Questions excluded (< {MIN_PAGE_VISITS} pages): {excl_pages}')
    print(f'  Questions excluded (idle >= {IDLE_THRESHOLD_PCT:.0f}%): {excl_idle}')
    print(f'  Questions excluded (both): {excl_both}')
    print(f'  Questions remaining: {len(clean_questions)} / {n_total_questions}')

    # Step 2: average per participant
    count_cols = ['count_time', 'count_topic', 'count_typing']
    avg_df = clean_questions.groupby('participant_id')[count_cols].mean().reset_index()
    avg_df = avg_df.dropna()

    # Step 3: 3 SD outlier exclusion
    outlier_mask = pd.Series(False, index=avg_df.index)
    outlier_details = []
    for col in count_cols:
        mean_val = avg_df[col].mean()
        sd_val = avg_df[col].std()
        lower = mean_val - OUTLIER_SD * sd_val
        upper = mean_val + OUTLIER_SD * sd_val
        col_outliers = (avg_df[col] < lower) | (avg_df[col] > upper)
        if col_outliers.any():
            for pid in avg_df.loc[col_outliers, 'participant_id']:
                val = avg_df.loc[avg_df['participant_id'] == pid, col].values[0]
                outlier_details.append(
                    f'    P{pid}: {col} = {val:.2f} (mean={mean_val:.2f}, sd={sd_val:.2f})'
                )
        outlier_mask = outlier_mask | col_outliers

    n_outliers = outlier_mask.sum()
    print(f'  Outliers (>{OUTLIER_SD} SD): {n_outliers}')
    for detail in outlier_details:
        print(detail)

    final_df = avg_df[~outlier_mask].reset_index(drop=True)
    print(f'  Final N: {len(final_df)} participants')

    exclusion_summary = {
        'n_total_questions': n_total_questions,
        'n_participants_before': n_participants_before,
        'excl_pages': int(excl_pages),
        'excl_idle': int(excl_idle),
        'excl_both': int(excl_both),
        'excl_total': int(excl_total),
        'n_clean_questions': len(clean_questions),
        'n_outliers': int(n_outliers),
        'outlier_details': outlier_details,
        'n_final': len(final_df),
    }

    return final_df, exclusion_summary


def run_pca(avg_df):
    """Run PCA on raw counts (no standardization)."""
    count_cols = ['count_time', 'count_topic', 'count_typing']
    X = avg_df[count_cols].values
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X)
    pct = pca.explained_variance_ratio_ * 100
    return pca, scores, pct


def plot_scree(ax, pct):
    cumulative = np.cumsum(pct)
    pc_labels = [f'PC{i+1}' for i in range(len(pct))]

    ax.set_facecolor(BG_COLOR)
    bars = ax.bar(pc_labels, pct, color=BAR_COLOR, zorder=2, edgecolor=BORDER_COLOR)
    ax.plot(pc_labels, cumulative, color=LINE_COLOR, marker='o', linewidth=2, zorder=3)

    for bar, val in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', color=TEXT_COLOR,
                fontsize=10, fontweight='bold')

    for x, cum_val in zip(pc_labels, cumulative):
        ax.text(x, cum_val + 2, f'{cum_val:.1f}%', ha='center', va='bottom',
                color=LINE_COLOR, fontsize=9)

    ax.set_ylabel('Variance Explained (%)', color=LABEL_COLOR, fontweight='bold')
    ax.set_title('Variance Explained by Each\nPrincipal Component',
                 color=TEXT_COLOR, fontweight='bold', fontsize=13)
    ax.set_ylim(0, 115)
    ax.tick_params(colors=MUTED_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


def plot_biplot(ax, pca, scores, pids, conditions, pct):
    loadings = pca.components_

    ax.set_facecolor(BG_COLOR)
    for i, pid in enumerate(pids):
        cond = conditions.get(pid, '')
        color = CONDITION_COLORS.get(cond, DEFAULT_DOT_COLOR)
        ax.scatter(scores[i, 0], scores[i, 1], color=color, s=60, alpha=0.85,
                   edgecolors='#333', linewidth=0.6, zorder=3)
        ax.annotate(f'P{pid}', (scores[i, 0], scores[i, 1]),
                    fontsize=7, color=LABEL_COLOR, ha='left', va='bottom',
                    xytext=(4, 4), textcoords='offset points')

    x_range = scores[:, 0].max() - scores[:, 0].min() if len(scores) > 1 else 1
    y_range = scores[:, 1].max() - scores[:, 1].min() if len(scores) > 1 else 1
    scale = 0.4 * max(x_range, y_range)

    for j, name in enumerate(FEATURE_NAMES):
        lx = loadings[0, j] * scale
        ly = loadings[1, j] * scale
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2.5))
        ax.text(lx * 1.18, ly * 1.18, name, color=ARROW_COLOR, fontsize=10,
                ha='center', va='center', fontweight='bold')

    ax.axhline(0, color=BORDER_COLOR, linewidth=0.5)
    ax.axvline(0, color=BORDER_COLOR, linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pct[0]:.1f}%)', color=LABEL_COLOR, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pct[1]:.1f}%)', color=LABEL_COLOR, fontweight='bold')
    ax.set_title('Biplot: Participants in\nPC1 vs PC2 Space',
                 color=TEXT_COLOR, fontweight='bold', fontsize=13)
    ax.tick_params(colors=MUTED_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


def plot_loadings_table(ax, pca):
    """Loadings as a color-coded heatmap."""
    loadings = pca.components_

    ax.set_facecolor(BG_COLOR)
    cmap_load = plt.cm.RdBu_r
    norm_load = plt.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(loadings.T, cmap=cmap_load, norm=norm_load, aspect='auto')

    for i in range(3):
        for j in range(3):
            val = loadings[j, i]
            c = cmap_load(norm_load(val))
            brightness = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            txt_c = 'white' if brightness < 0.6 else TEXT_COLOR
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=txt_c)

    ax.set_xticks(range(3))
    ax.set_xticklabels(['PC1', 'PC2', 'PC3'], fontsize=10, fontweight='bold')
    ax.set_yticks(range(3))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
    ax.set_title('PCA Loadings:\nContribution of Each Variable',
                 color=TEXT_COLOR, fontweight='bold', fontsize=13)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


def plot_exclusion_summary(ax, question_df, exclusion_summary):
    """Bar chart showing excluded vs kept questions per participant."""
    ax.set_facecolor(BG_COLOR)

    pids = sorted(question_df['participant_id'].unique())
    kept_counts = []
    excl_counts = []
    for pid in pids:
        pid_q = question_df[question_df['participant_id'] == pid]
        excl = (pid_q['excluded_pages'] | pid_q['excluded_idle']).sum()
        kept = len(pid_q) - excl
        kept_counts.append(kept)
        excl_counts.append(excl)

    x = np.arange(len(pids))
    ax.bar(x, kept_counts, color=KEPT_COLOR, label='Kept', edgecolor='white', linewidth=0.5)
    ax.bar(x, excl_counts, bottom=kept_counts, color=EXCLUDED_COLOR,
           label='Excluded', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'P{p}' for p in pids], fontsize=8, rotation=45)
    ax.set_ylabel('Questions', color=LABEL_COLOR, fontweight='bold')
    ax.set_title('Exclusion Summary:\nQuestions Kept vs Excluded per Participant',
                 color=TEXT_COLOR, fontweight='bold', fontsize=13)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.tick_params(colors=MUTED_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_counts_distribution(ax, avg_df):
    """Box plots of the three count variables."""
    ax.set_facecolor(BG_COLOR)
    count_cols = ['count_time', 'count_topic', 'count_typing']

    data = [avg_df[col].values for col in count_cols]
    bp = ax.boxplot(data, labels=FEATURE_NAMES, patch_artist=True,
                    boxprops=dict(facecolor='#BBDEFB', edgecolor=BORDER_COLOR),
                    medianprops=dict(color=LINE_COLOR, linewidth=2),
                    whiskerprops=dict(color=MUTED_COLOR),
                    capprops=dict(color=MUTED_COLOR),
                    flierprops=dict(marker='o', markerfacecolor=EXCLUDED_COLOR, markersize=5))

    ax.set_ylabel('Switch Count (mean per participant)', color=LABEL_COLOR, fontweight='bold')
    ax.set_title('Distribution of Raw Switch Counts\n(Averaged per Participant)',
                 color=TEXT_COLOR, fontweight='bold', fontsize=13)
    ax.tick_params(colors=MUTED_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_correlation(ax, avg_df):
    """Polished triangular correlation heatmap with rounded cells."""
    ax.set_facecolor(BG_COLOR)
    count_cols = ['count_time', 'count_topic', 'count_typing']
    short_names = ['Time', 'Topic', 'Typing']
    corr = avg_df[count_cols].corr().values
    n = len(short_names)

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # Draw cells manually with rounded rectangles
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import Normalize
    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            val = corr[i, j]
            color = cmap(norm(val))
            rect = FancyBboxPatch(
                (j - 0.42, i - 0.42), 0.84, 0.84,
                boxstyle='round,pad=0.05,rounding_size=0.15',
                facecolor=color, edgecolor='white', linewidth=2,
            )
            ax.add_patch(rect)

            # Text color: white on dark, dark on light
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            txt_color = 'white' if brightness < 0.6 else TEXT_COLOR

            if i == j:
                ax.text(j, i, short_names[i], ha='center', va='center',
                        fontsize=11, fontweight='bold', color=txt_color)
            else:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=txt_color)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Correlation Between\nSwitch Count Variables',
                 color=TEXT_COLOR, fontweight='bold', fontsize=13, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_correlation_grid(ax, avg_df):
    """Full-grid correlation heatmap with polished colors for PDF."""
    ax.set_facecolor(BG_COLOR)
    count_cols = ['count_time', 'count_topic', 'count_typing']
    short_labels = ['Time', 'Topic', 'Typing']
    corr = avg_df[count_cols].corr().values
    n = len(short_labels)

    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(vmin=0, vmax=1)

    im = ax.imshow(corr, cmap=cmap, norm=norm, aspect='equal')

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            c = cmap(norm(val))
            brightness = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            txt_c = 'white' if brightness < 0.55 else TEXT_COLOR
            if i == j:
                label = f'{short_labels[i]}\n(1.00)'
            else:
                label = f'{val:.2f}'
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=txt_c)

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=11, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=11, fontweight='bold')
    ax.set_title('Correlation Between\nSwitch Count Variables',
                 color=TEXT_COLOR, fontweight='bold', fontsize=13, pad=12)

    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1.5)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.outline.set_edgecolor(BORDER_COLOR)
    cb.ax.tick_params(colors=MUTED_COLOR)


def create_png(avg_df, pca, scores, pct, conditions, question_df, exclusion_summary):
    """Main PNG figure with all panels."""
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor(BG_COLOR)

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          top=0.91, bottom=0.06, left=0.06, right=0.96,
                          width_ratios=[1, 1.3, 0.8])

    pids = avg_df['participant_id'].values

    # Top row: Scree, Biplot, Loadings
    plot_scree(fig.add_subplot(gs[0, 0]), pct)
    plot_biplot(fig.add_subplot(gs[0, 1]), pca, scores, pids, conditions, pct)
    plot_loadings_table(fig.add_subplot(gs[0, 2]), pca)

    # Bottom row: Exclusion summary, Counts distribution, Correlation
    plot_exclusion_summary(fig.add_subplot(gs[1, 0]), question_df, exclusion_summary)
    plot_counts_distribution(fig.add_subplot(gs[1, 1]), avg_df)

    # Correlation matrix
    plot_correlation(fig.add_subplot(gs[1, 2]), avg_df)

    n = exclusion_summary['n_final']
    fig.suptitle(
        'M51: Final Composite Dependent Variable\n'
        f'PCA on Raw Switch Counts (Time, Topic-LDA, Typing)  |  N = {n}',
        color=TEXT_COLOR, fontsize=16, fontweight='bold', y=0.99,
    )

    return fig


def create_pdf_report(avg_df, pca, scores, pct, conditions, question_df,
                      exclusion_summary, output_path):
    """Multi-page PDF report."""
    with PdfPages(output_path) as pdf:
        # --- Page 1: Title + Exclusion ---
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5),
                                 gridspec_kw={'height_ratios': [1, 1.5]})
        fig.patch.set_facecolor(BG_COLOR)

        # Title section
        ax_title = axes[0]
        ax_title.axis('off')
        ax_title.text(0.5, 0.90, 'M51: Final Composite Dependent Variable',
                      transform=ax_title.transAxes, fontsize=22, fontweight='bold',
                      ha='center', va='top', color=TEXT_COLOR)
        ax_title.text(0.5, 0.72, 'PCA on Raw Switch Counts (Time, Topic-LDA, Typing)',
                      transform=ax_title.transAxes, fontsize=13,
                      ha='center', va='top', color=MUTED_COLOR)
        ax_title.text(0.5, 0.62, 'No standardization (z-score) applied',
                      transform=ax_title.transAxes, fontsize=11,
                      ha='center', va='top', color=MUTED_COLOR, style='italic')

        summary_text = (
            f"Exclusion Criteria Applied:\n"
            f"  1. Questions with fewer than {MIN_PAGE_VISITS} page visits: "
            f"{exclusion_summary['excl_pages']} excluded\n"
            f"  2. Questions with idle time >= {IDLE_THRESHOLD_PCT:.0f}% "
            f"after last meaningful event: {exclusion_summary['excl_idle']} excluded\n"
            f"  3. Participants > {OUTLIER_SD} SD from mean on any DV: "
            f"{exclusion_summary['n_outliers']} excluded\n"
            f"\n"
            f"Result: {exclusion_summary['n_clean_questions']} / "
            f"{exclusion_summary['n_total_questions']} questions retained\n"
            f"Final sample: N = {exclusion_summary['n_final']} participants"
        )
        ax_title.text(0.5, 0.48, summary_text,
                      transform=ax_title.transAxes, fontsize=10,
                      ha='center', va='top', color=TEXT_COLOR,
                      family='monospace', linespacing=1.7)

        plot_exclusion_summary(axes[1], question_df, exclusion_summary)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, facecolor=BG_COLOR)
        plt.close()

        # --- Page 2: Descriptive Statistics ---
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.patch.set_facecolor(BG_COLOR)
        fig.suptitle('Descriptive Statistics: Raw Switch Counts',
                     fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98)

        plot_counts_distribution(axes[0], avg_df)
        plot_correlation_grid(axes[1], avg_df)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, facecolor=BG_COLOR)
        plt.close()

        # --- Page 3: PCA Results ---
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(BG_COLOR)
        fig.suptitle('PCA Results: Dimensionality Reduction on Raw Switch Counts',
                     fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98)

        gs = fig.add_gridspec(1, 2, wspace=0.05,
                              top=0.88, bottom=0.08, left=0.08, right=0.95,
                              width_ratios=[1, 1.2])

        # Left: Scree plot (with fixed label positions)
        ax_scree = fig.add_subplot(gs[0, 0])
        cumulative = np.cumsum(pct)
        pc_labels = [f'PC{i+1}' for i in range(len(pct))]

        ax_scree.set_facecolor(BG_COLOR)
        bars = ax_scree.bar(pc_labels, pct, color=BAR_COLOR, zorder=2, edgecolor=BORDER_COLOR)
        ax_scree.plot(pc_labels, cumulative, color=LINE_COLOR, marker='o', linewidth=2, zorder=3)

        # Bar value labels - inside bars to avoid overlap with cumulative
        for bar, val in zip(bars, pct):
            y_pos = bar.get_height() / 2
            ax_scree.text(bar.get_x() + bar.get_width() / 2, y_pos,
                          f'{val:.1f}%', ha='center', va='center', color='white',
                          fontsize=11, fontweight='bold', zorder=4)

        # Cumulative labels - above the line with enough offset
        for i, (x, cum_val) in enumerate(zip(pc_labels, cumulative)):
            ax_scree.text(x, cum_val + 4, f'{cum_val:.1f}%', ha='center', va='bottom',
                          color=LINE_COLOR, fontsize=9, fontweight='bold')

        ax_scree.set_ylabel('Variance Explained (%)', color=LABEL_COLOR, fontweight='bold')
        ax_scree.set_title('Variance Explained by Each\nPrincipal Component',
                           color=TEXT_COLOR, fontweight='bold', fontsize=13)
        ax_scree.set_ylim(0, 120)
        ax_scree.tick_params(colors=MUTED_COLOR)
        ax_scree.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
        for spine in ax_scree.spines.values():
            spine.set_color(BORDER_COLOR)

        # Right: Text explanation of results
        ax_text = fig.add_subplot(gs[0, 1])
        ax_text.axis('off')
        ax_text.set_facecolor(BG_COLOR)

        loadings = pca.components_

        explanation = 'PCA Loadings (variable weights per component):\n\n'
        for i in range(3):
            explanation += f'  PC{i+1} ({pct[i]:.1f}% variance):\n'
            # Sort by absolute loading
            load_order = np.argsort(-np.abs(loadings[i]))
            for j in load_order:
                val = loadings[i, j]
                bar_len = int(abs(val) * 20)
                bar_char = '+' if val > 0 else '-'
                bar = bar_char * bar_len
                explanation += f'    {FEATURE_NAMES[j]:14s}  {val:+.3f}  {bar}\n'
            explanation += '\n'

        explanation += 'Interpretation:\n\n'
        for i in range(3):
            dominant_idx = np.argmax(np.abs(loadings[i]))
            dominant_name = FEATURE_NAMES[dominant_idx]
            dominant_val = loadings[i, dominant_idx]
            direction = 'more' if dominant_val > 0 else 'fewer'
            var_short = dominant_name.split()[-1].lower()
            explanation += (
                f'  PC{i+1}: Primarily reflects {var_short} switching\n'
                f'       (loading = {dominant_val:+.3f})\n'
                f'       Higher score = {direction} {var_short} transitions\n\n'
            )

        ax_text.text(0.02, 0.98, explanation, transform=ax_text.transAxes,
                     fontsize=10, family='monospace', va='top', color=TEXT_COLOR,
                     linespacing=1.4)

        pdf.savefig(fig, facecolor=BG_COLOR)
        plt.close()

        # --- Page 4: Per-participant scores ---
        pids = avg_df['participant_id'].values
        n_participants = len(pids)
        fig_h = max(6, 0.4 * n_participants + 2)
        fig, ax = plt.subplots(figsize=(11, min(fig_h, 14)))
        fig.patch.set_facecolor(BG_COLOR)

        pc1 = scores[:, 0]
        idx = np.argsort(pc1)
        sorted_pids = pids[idx]
        sorted_pc1 = pc1[idx]
        colors = [CONDITION_COLORS.get(conditions.get(p, ''), DEFAULT_DOT_COLOR)
                  for p in sorted_pids]

        bar_height = 0.65 if n_participants <= 30 else 0.5
        ax.barh(range(n_participants), sorted_pc1, color=colors,
                edgecolor='white', linewidth=0.5, height=bar_height)

        # Value labels on bars
        for i, val in enumerate(sorted_pc1):
            ha = 'left' if val >= 0 else 'right'
            offset = 0.08 if val >= 0 else -0.08
            ax.text(val + offset, i, f'{val:.1f}', ha=ha, va='center',
                    fontsize=8, color=MUTED_COLOR)

        ax.set_yticks(range(n_participants))
        fs = 9 if n_participants <= 25 else 7
        ax.set_yticklabels([f'P{p}' for p in sorted_pids], fontsize=fs)
        ax.set_xlabel('PC1 Score (Composite Explore-Exploit Index)',
                      color=LABEL_COLOR, fontweight='bold', fontsize=11)
        ax.axvline(0, color=BORDER_COLOR, linewidth=1)

        mean_pc1 = np.mean(pc1)
        ax.axvline(mean_pc1, color=EXCLUDED_COLOR, linestyle='--', linewidth=1.5)
        ax.text(mean_pc1 + 0.1, -1.2, f'Mean = {mean_pc1:.2f}',
                color=EXCLUDED_COLOR, fontsize=10, fontweight='bold', va='top')

        ax.set_title('Individual PC1 Scores: Composite Explore-Exploit Index',
                     color=TEXT_COLOR, fontweight='bold', fontsize=14, pad=15)
        ax.set_facecolor(BG_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        ax.tick_params(colors=MUTED_COLOR)
        ax.set_ylim(-1.5, n_participants - 0.3)

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        pdf.savefig(fig, facecolor=BG_COLOR)
        plt.close()

        # --- Page 5: Condition Comparison (Clumpy vs Diffuse) ---
        _plot_condition_comparison_page(pdf, avg_df, scores, pct, conditions)

    print(f'Saved: {output_path}')


def _plot_condition_comparison_page(pdf, avg_df, scores, pct, conditions):
    """Two-page condition comparison: visuals + stats with t-tests."""
    pids = avg_df['participant_id'].values
    cond_labels = [conditions.get(p, '') for p in pids]
    pc1 = scores[:, 0]

    diffuse_mask = np.array([c == 'diffuse' for c in cond_labels])
    clumpy_mask = np.array([c == 'clumpy' for c in cond_labels])
    pc1_diffuse = pc1[diffuse_mask]
    pc1_clumpy = pc1[clumpy_mask]
    n_d, n_c = int(diffuse_mask.sum()), int(clumpy_mask.sum())

    count_cols = ['count_time', 'count_topic', 'count_typing']
    col_labels = ['Time', 'Topic (LDA)', 'Typing/Paste']

    mean_d = np.mean(pc1_diffuse)
    mean_c = np.mean(pc1_clumpy)

    # ---- Page 5a: Three visual panels ----
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        'Condition Comparison: Clumpy vs Diffuse\n'
        f'N = {len(pids)} (Diffuse: {n_d}, Clumpy: {n_c})',
        fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                          top=0.88, bottom=0.07, left=0.09, right=0.95)

    # --- Top-left: PC1 strip plot by condition ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG_COLOR)

    jitter_d = np.random.default_rng(42).uniform(-0.06, 0.06, size=len(pc1_diffuse))
    jitter_c = np.random.default_rng(7).uniform(-0.06, 0.06, size=len(pc1_clumpy))

    ax1.scatter(np.zeros(len(pc1_diffuse)) + jitter_d, pc1_diffuse,
                color=CONDITION_COLORS['diffuse'], s=70, alpha=0.85,
                edgecolors='#333', linewidth=0.6, zorder=3)
    ax1.scatter(np.ones(len(pc1_clumpy)) + jitter_c, pc1_clumpy,
                color=CONDITION_COLORS['clumpy'], s=70, alpha=0.85,
                edgecolors='#333', linewidth=0.6, zorder=3)

    pids_d = pids[diffuse_mask]
    pids_c = pids[clumpy_mask]
    for i, (pid, val) in enumerate(zip(pids_d, pc1_diffuse)):
        ax1.annotate(f'P{pid}', (jitter_d[i], val), fontsize=7,
                     color=LABEL_COLOR, xytext=(6, 0), textcoords='offset points')
    for i, (pid, val) in enumerate(zip(pids_c, pc1_clumpy)):
        ax1.annotate(f'P{pid}', (1 + jitter_c[i], val), fontsize=7,
                     color=LABEL_COLOR, xytext=(6, 0), textcoords='offset points')

    bar_w = 0.3
    ax1.hlines(mean_d, -bar_w, bar_w, color=CONDITION_COLORS['diffuse'],
               linewidth=2.5, zorder=4)
    ax1.hlines(mean_c, 1 - bar_w, 1 + bar_w, color=CONDITION_COLORS['clumpy'],
               linewidth=2.5, zorder=4)
    ax1.text(-bar_w - 0.05, mean_d, f'M={mean_d:.2f}', ha='right', va='center',
             fontsize=9, color=CONDITION_COLORS['diffuse'], fontweight='bold')
    ax1.text(1 + bar_w + 0.05, mean_c, f'M={mean_c:.2f}', ha='left', va='center',
             fontsize=9, color=CONDITION_COLORS['clumpy'], fontweight='bold')

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Diffuse', 'Clumpy'], fontsize=11, fontweight='bold')
    ax1.set_ylabel('PC1 Score (Composite Index)', color=LABEL_COLOR, fontweight='bold')
    ax1.set_title('PC1: Composite Explore-Exploit\nby Condition', color=TEXT_COLOR,
                  fontweight='bold', fontsize=12)
    ax1.set_xlim(-0.5, 1.5)
    ax1.axhline(0, color=BORDER_COLOR, linewidth=0.5, linestyle=':')
    ax1.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax1.spines.values():
        spine.set_color(BORDER_COLOR)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Top-right: grouped bar chart ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG_COLOR)

    means_d_list = [avg_df.loc[diffuse_mask, c].mean() for c in count_cols]
    means_c_list = [avg_df.loc[clumpy_mask, c].mean() for c in count_cols]
    sds_d_list = [avg_df.loc[diffuse_mask, c].std() for c in count_cols]
    sds_c_list = [avg_df.loc[clumpy_mask, c].std() for c in count_cols]

    x = np.arange(len(count_cols))
    w = 0.32
    bars_d = ax2.bar(x - w / 2, means_d_list, w, yerr=sds_d_list, capsize=4,
                     color=CONDITION_COLORS['diffuse'], edgecolor='white',
                     linewidth=0.5, label='Diffuse', alpha=0.85, zorder=2)
    bars_c = ax2.bar(x + w / 2, means_c_list, w, yerr=sds_c_list, capsize=4,
                     color=CONDITION_COLORS['clumpy'], edgecolor='white',
                     linewidth=0.5, label='Clumpy', alpha=0.85, zorder=2)

    for bar, val in zip(bars_d, means_d_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                 color=CONDITION_COLORS['diffuse'], fontweight='bold')
    for bar, val in zip(bars_c, means_c_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                 color=CONDITION_COLORS['clumpy'], fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(col_labels, fontsize=10)
    ax2.set_ylabel('Mean Switch Count', color=LABEL_COLOR, fontweight='bold')
    ax2.set_title('Raw Switch Counts\nby Condition (Mean +/- SD)', color=TEXT_COLOR,
                  fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax2.spines.values():
        spine.set_color(BORDER_COLOR)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # --- Bottom-left: individual profiles (parallel coordinates) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG_COLOR)

    for i, pid in enumerate(pids):
        cond = conditions.get(pid, '')
        color = CONDITION_COLORS.get(cond, DEFAULT_DOT_COLOR)
        vals = [avg_df.loc[avg_df['participant_id'] == pid, c].values[0] for c in count_cols]
        alpha = 0.7 if cond == 'diffuse' else 0.9
        lw = 1.5 if cond == 'diffuse' else 2.5
        ax3.plot(range(3), vals, marker='o', color=color, alpha=alpha,
                 linewidth=lw, markersize=5, zorder=3)
        ax3.annotate(f'P{pid}', (2, vals[2]), fontsize=7, color=color,
                     xytext=(5, 0), textcoords='offset points')

    ax3.set_xticks(range(3))
    ax3.set_xticklabels(col_labels, fontsize=10)
    ax3.set_ylabel('Switch Count', color=LABEL_COLOR, fontweight='bold')
    ax3.set_title('Individual Profiles\nby Switch Type', color=TEXT_COLOR,
                  fontweight='bold', fontsize=12)
    ax3.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
    for spine in ax3.spines.values():
        spine.set_color(BORDER_COLOR)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # --- Bottom-right: empty placeholder (stats on next page) ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_facecolor(BG_COLOR)

    # Legend box
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CONDITION_COLORS['diffuse'],
               markersize=10, label=f'Diffuse (n={n_d})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CONDITION_COLORS['clumpy'],
               markersize=10, label=f'Clumpy (n={n_c})'),
    ]
    ax4.legend(handles=legend_elements, loc='center', fontsize=13,
               framealpha=0.9, edgecolor=BORDER_COLOR, title='Conditions',
               title_fontsize=14)

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()

    # ---- Page 5b: Statistics + t-tests ----
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        'Condition Comparison: Descriptive Statistics & t-Tests\n'
        f'N = {len(pids)} (Diffuse: {n_d}, Clumpy: {n_c})',
        fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.97,
    )

    ax = fig.add_axes([0.06, 0.05, 0.88, 0.82])
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)

    # Compute t-tests
    test_rows = []
    all_vars = count_cols + ['PC1']
    all_labels = col_labels + ['PC1 (composite)']
    for var, label in zip(all_vars, all_labels):
        if var == 'PC1':
            d_vals = pc1_diffuse
            c_vals = pc1_clumpy
        else:
            d_vals = avg_df.loc[diffuse_mask, var].values
            c_vals = avg_df.loc[clumpy_mask, var].values

        d_mean, d_sd = np.mean(d_vals), np.std(d_vals, ddof=1)
        c_mean, c_sd = np.mean(c_vals), np.std(c_vals, ddof=1)

        t_stat, p_val = sp_stats.ttest_ind(d_vals, c_vals, equal_var=False)
        df_welch = _welch_df(d_vals, c_vals)
        pooled_sd = np.sqrt(((len(d_vals) - 1) * d_sd**2 + (len(c_vals) - 1) * c_sd**2)
                            / (len(d_vals) + len(c_vals) - 2)) if (d_sd + c_sd) > 0 else 0
        cohens_d = (c_mean - d_mean) / pooled_sd if pooled_sd > 0 else 0

        test_rows.append({
            'label': label, 'd_mean': d_mean, 'd_sd': d_sd,
            'c_mean': c_mean, 'c_sd': c_sd,
            't': t_stat, 'df': df_welch, 'p': p_val, 'd_cohen': cohens_d,
        })

    # Build formatted text
    lines = []
    lines.append('DESCRIPTIVE STATISTICS')
    lines.append('=' * 72)
    lines.append(f'{"Variable":<18} {"Diffuse (n="+str(n_d)+")":>22}  {"Clumpy (n="+str(n_c)+")":>22}')
    lines.append('-' * 72)
    for r in test_rows:
        d_str = f'M = {r["d_mean"]:5.2f}, SD = {r["d_sd"]:4.2f}'
        c_str = f'M = {r["c_mean"]:5.2f}, SD = {r["c_sd"]:4.2f}'
        lines.append(f'{r["label"]:<18} {d_str:>22}  {c_str:>22}')
    lines.append('')
    lines.append('')
    lines.append('INDEPENDENT-SAMPLES t-TESTS (Welch)')
    lines.append('=' * 72)
    lines.append(f'{"Variable":<18} {"t":>7} {"df":>6} {"p":>9} {"Cohen d":>9}  {"Sig.":>6}')
    lines.append('-' * 72)
    for r in test_rows:
        sig = '***' if r['p'] < .001 else '**' if r['p'] < .01 else '*' if r['p'] < .05 else 'n.s.'
        lines.append(
            f'{r["label"]:<18} {r["t"]:>7.2f} {r["df"]:>6.1f} {r["p"]:>9.4f} {r["d_cohen"]:>+9.2f}  {sig:>6}'
        )
    lines.append('')
    lines.append('  * p < .05   ** p < .01   *** p < .001')
    lines.append('')
    lines.append('')
    lines.append('INTERPRETATION')
    lines.append('=' * 72)

    pc1_row = test_rows[-1]
    diff = pc1_row['c_mean'] - pc1_row['d_mean']
    direction = 'more' if diff > 0 else 'fewer'
    lines.append(f'  PC1 difference (Clumpy - Diffuse): {diff:+.2f}')
    lines.append(f'  Clumpy condition shows {direction} explore-exploit switches')
    lines.append(f'  Effect size (Cohen\'s d): {pc1_row["d_cohen"]:+.2f}')
    if pc1_row['p'] < .05:
        lines.append(f'  Result is statistically significant (p = {pc1_row["p"]:.4f})')
    else:
        lines.append(f'  Result is NOT statistically significant (p = {pc1_row["p"]:.4f})')
        lines.append(f'  Low statistical power due to small sample (N = {len(pids)})')

    # Largest individual-variable effect
    max_d_row = max(test_rows[:-1], key=lambda r: abs(r['d_cohen']))
    lines.append('')
    lines.append(f'  Largest sub-variable effect: {max_d_row["label"]}')
    lines.append(f'    d = {max_d_row["d_cohen"]:+.2f}, p = {max_d_row["p"]:.4f}')

    lines.append('')
    lines.append('-' * 72)
    lines.append('  Note: Typing metric relies on paste events only')
    lines.append('  (answer_snapshot missing from current log).')

    text = '\n'.join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9.5, family='monospace', va='top', color=TEXT_COLOR,
            linespacing=1.45)

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()


def _welch_df(a, b):
    """Welch-Satterthwaite degrees of freedom."""
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    if denom == 0:
        return n1 + n2 - 2
    return num / denom


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M51] Final Composite DV - Switch Counts + PCA (No Z-Score)')
    print('=' * 60)

    trials = load_trials(DATA_DIR / 'Game_new.csv')
    conditions = {tr['pid']: tr['condition'] for tr in trials}

    # Build question-level data
    print('\n--- Building question-level data ---')
    question_df = build_question_data(trials)
    print(f'  Total questions: {len(question_df)}')

    # Apply exclusions
    print('\n--- Applying exclusion criteria ---')
    avg_df, exclusion_summary = apply_exclusions(question_df)

    # Descriptive stats
    count_cols = ['count_time', 'count_topic', 'count_typing']
    print('\n--- Descriptive statistics (final sample) ---')
    for col in count_cols:
        vals = avg_df[col]
        print(f'  {col}: mean={vals.mean():.2f}, sd={vals.std():.2f}, '
              f'min={vals.min():.2f}, max={vals.max():.2f}')

    corr = avg_df[count_cols].corr()
    print('\n--- Correlation matrix ---')
    print(corr.to_string())

    # PCA
    print('\n--- PCA (raw counts, no standardization) ---')
    pca, scores, pct = run_pca(avg_df)

    for i, v in enumerate(pct):
        print(f'  PC{i+1}: {v:.1f}%')
    print('  Loadings:')
    for i, comp in enumerate(pca.components_):
        parts = ', '.join(f'{FEATURE_NAMES[j]}: {comp[j]:+.3f}' for j in range(3))
        print(f'    PC{i+1}: {parts}')

    # PNG
    print('\n--- Saving outputs ---')
    fig = create_png(avg_df, pca, scores, pct, conditions, question_df, exclusion_summary)
    out_png = OUTPUT_DIR / 'm51_final_composite_dv.png'
    fig.savefig(out_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_png}')

    # CSV
    scores_df = pd.DataFrame({
        'participant_id': avg_df['participant_id'].values,
        'condition': [conditions.get(p, '') for p in avg_df['participant_id']],
        'count_time': avg_df['count_time'].values,
        'count_topic': avg_df['count_topic'].values,
        'count_typing': avg_df['count_typing'].values,
        'PC1': scores[:, 0],
        'PC2': scores[:, 1],
        'PC3': scores[:, 2],
    })
    csv_path = OUTPUT_DIR / 'm51_final_composite_dv.csv'
    scores_df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    # PDF report
    pdf_path = OUTPUT_DIR / 'm51_final_composite_dv.pdf'
    create_pdf_report(avg_df, pca, scores, pct, conditions, question_df,
                      exclusion_summary, pdf_path)

    print('\nDone.')


if __name__ == '__main__':
    main()

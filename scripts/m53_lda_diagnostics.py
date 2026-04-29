#!/usr/bin/env python3
"""
M53: LDA Diagnostics
====================
Investigate why count_topic (LDA) diverges from count_time and count_typing in
the M52 PCA. Per-advisor request: are the differences driven by outliers, by
computational artefacts, or by something real?

Outputs:
  - output/m53_lda_diagnostics.pdf  (multi-page report)
  - output/m53_lda_diagnostics.csv  (per-participant z-scores + residuals)
  - output/m53_lda_per_question.csv (per-question topic sequences for the
                                     largest LDA-residual participants)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from helpers import load_trials, get_pids_and_trials, OUTPUT_DIR
from m52_final_composite_dv import (
    build_question_data,
    apply_exclusions,
    load_lda_assignments,
    BG_COLOR, TEXT_COLOR, LABEL_COLOR, GRID_COLOR, BORDER_COLOR,
    MUTED_COLOR, BAR_COLOR, LINE_COLOR, EXCLUDED_COLOR, KEPT_COLOR,
    THRESHOLD_S,
)
from m18_typing_binary import page_had_typing_or_paste

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

TOP_K_SUSPECTS = 8  # how many participants to deep-dive
RESIDUAL_FLAG_SD = 1.5  # |residual| above this is flagged


def zscore(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0


def compute_residuals(avg_df):
    """Per-participant z-scores + LDA residual vs time/typing average."""
    df = avg_df.copy()
    df['z_time'] = zscore(df['count_time'])
    df['z_topic'] = zscore(df['count_topic'])
    df['z_typing'] = zscore(df['count_typing'])
    df['z_time_typing_mean'] = (df['z_time'] + df['z_typing']) / 2
    df['lda_residual'] = df['z_topic'] - df['z_time_typing_mean']
    df['abs_residual'] = df['lda_residual'].abs()
    return df.sort_values('abs_residual', ascending=False).reset_index(drop=True)


def collect_suspect_sequences(suspects, trials, lda):
    """For each suspect pid, dump the topic sequence per kept question."""
    rows = []
    pids, pid_trials = get_pids_and_trials(trials)
    for pid in suspects:
        if pid not in pid_trials:
            continue
        for tr in pid_trials[pid]:
            if tr['domain'] == 'practice':
                continue
            pvs = tr['page_visits']
            titles = [pv['title'] for pv in pvs]
            topics = [lda.get(t, -1) for t in titles]
            time_labels = ['exploit' if pv['duration'] > THRESHOLD_S else 'explore' for pv in pvs]
            typing_labels = [
                'T' if page_had_typing_or_paste(pv, tr['typing_intervals'], tr['paste_times']) else '-'
                for pv in pvs
            ]
            switches_topic = sum(1 for i in range(1, len(topics)) if topics[i] != topics[i - 1])
            switches_time = sum(1 for i in range(1, len(time_labels)) if time_labels[i] != time_labels[i - 1])
            switches_typing = sum(1 for i in range(1, len(typing_labels)) if typing_labels[i] != typing_labels[i - 1])
            rows.append({
                'pid': pid,
                'trial': tr['trial'],
                'domain': tr['domain'],
                'n_pages': len(pvs),
                'switches_time': switches_time,
                'switches_topic': switches_topic,
                'switches_typing': switches_typing,
                'topic_sequence': ','.join(str(t) for t in topics),
                'time_sequence': ''.join('E' if x == 'exploit' else 'X' for x in time_labels),
                'typing_sequence': ''.join(typing_labels),
                'titles': ' | '.join(titles),
            })
    return pd.DataFrame(rows)


def plot_zscore_table(ax, diag_df):
    """Bar chart of LDA residual per participant."""
    ax.set_facecolor(BG_COLOR)
    sorted_df = diag_df.sort_values('lda_residual')
    pids = sorted_df['participant_id'].values
    res = sorted_df['lda_residual'].values
    colors = [EXCLUDED_COLOR if abs(r) >= RESIDUAL_FLAG_SD else BAR_COLOR for r in res]
    ax.barh(range(len(pids)), res, color=colors, edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels([f'P{p}' for p in pids], fontsize=6)
    ax.axvline(0, color=BORDER_COLOR, linewidth=0.8)
    ax.axvline(RESIDUAL_FLAG_SD, color=EXCLUDED_COLOR, linestyle='--', linewidth=0.8)
    ax.axvline(-RESIDUAL_FLAG_SD, color=EXCLUDED_COLOR, linestyle='--', linewidth=0.8)
    ax.set_xlabel('LDA residual (z_topic - mean(z_time, z_typing))',
                  color=LABEL_COLOR, fontweight='bold', fontsize=10)
    ax.set_title('LDA Residual by Participant\n'
                  '(positive = LDA inflated relative to time/typing)',
                  color=TEXT_COLOR, fontweight='bold', fontsize=12)
    ax.tick_params(colors=MUTED_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_scatter(ax, diag_df, x_col, y_col, x_label, y_label, title):
    ax.set_facecolor(BG_COLOR)
    x = diag_df[x_col].values
    y = diag_df[y_col].values
    pids = diag_df['participant_id'].values
    flagged = diag_df['abs_residual'].values >= RESIDUAL_FLAG_SD

    ax.scatter(x[~flagged], y[~flagged], color=BAR_COLOR, s=45, alpha=0.75,
               edgecolors='#333', linewidth=0.5, zorder=2, label='kept')
    ax.scatter(x[flagged], y[flagged], color=EXCLUDED_COLOR, s=70, alpha=0.9,
               edgecolors='#333', linewidth=0.7, zorder=3, label=f'|residual| >= {RESIDUAL_FLAG_SD}')

    # Identity line
    lo = min(x.min(), y.min()) - 0.3
    hi = max(x.max(), y.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color=MUTED_COLOR, linewidth=1, linestyle=':',
             zorder=1, label='y = x')

    # Annotate flagged
    for xi, yi, p, f in zip(x, y, pids, flagged):
        if f:
            ax.annotate(f'P{p}', (xi, yi), fontsize=7, color=EXCLUDED_COLOR,
                          xytext=(4, 4), textcoords='offset points')

    # Pearson r
    if len(x) > 2:
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.04, 0.96, f'r = {r:.2f}', transform=ax.transAxes, fontsize=10,
                  color=TEXT_COLOR, fontweight='bold', va='top',
                  bbox=dict(facecolor='#FAFAFA', edgecolor=BORDER_COLOR, boxstyle='round,pad=0.3'))

    ax.set_xlabel(x_label, color=LABEL_COLOR, fontweight='bold')
    ax.set_ylabel(y_label, color=LABEL_COLOR, fontweight='bold')
    ax.set_title(title, color=TEXT_COLOR, fontweight='bold', fontsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.tick_params(colors=MUTED_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, zorder=0)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


def plot_count_vs_pages(ax, diag_df):
    """count_topic vs n_pages_avg (per participant) - tests computational stability."""
    ax.set_facecolor(BG_COLOR)
    x = diag_df['n_pages_avg'].values
    y = diag_df['count_topic'].values
    pids = diag_df['participant_id'].values
    flagged = diag_df['abs_residual'].values >= RESIDUAL_FLAG_SD

    ax.scatter(x[~flagged], y[~flagged], color=BAR_COLOR, s=45, alpha=0.75,
                edgecolors='#333', linewidth=0.5, zorder=2)
    ax.scatter(x[flagged], y[flagged], color=EXCLUDED_COLOR, s=70, alpha=0.9,
                edgecolors='#333', linewidth=0.7, zorder=3)
    for xi, yi, p, f in zip(x, y, pids, flagged):
        if f:
            ax.annotate(f'P{p}', (xi, yi), fontsize=7, color=EXCLUDED_COLOR,
                          xytext=(4, 4), textcoords='offset points')

    if len(x) > 2:
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.04, 0.96, f'r = {r:.2f}', transform=ax.transAxes, fontsize=10,
                  color=TEXT_COLOR, fontweight='bold', va='top',
                  bbox=dict(facecolor='#FAFAFA', edgecolor=BORDER_COLOR, boxstyle='round,pad=0.3'))

    ax.set_xlabel('Avg pages per kept question', color=LABEL_COLOR, fontweight='bold')
    ax.set_ylabel('count_topic (LDA)', color=LABEL_COLOR, fontweight='bold')
    ax.set_title('LDA Switches vs Page Count\n'
                  '(computational stability check)',
                  color=TEXT_COLOR, fontweight='bold', fontsize=12)
    ax.tick_params(colors=MUTED_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


def plot_suspect_text(ax, suspect_seq_df, suspects):
    """Text page showing topic sequences for the highest-residual participants."""
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)

    lines = []
    lines.append('TOP LDA-RESIDUAL PARTICIPANTS - PAGE SEQUENCES')
    lines.append('=' * 92)
    lines.append('Notation:  topic = LDA topic id (0..9)   |   time: E=exploit (>60s)  X=explore (<=60s)')
    lines.append('           typing: T=typing/paste on page   -=none')
    lines.append('')

    for pid in suspects:
        rows = suspect_seq_df[suspect_seq_df['pid'] == pid]
        if rows.empty:
            continue
        for _, r in rows.iterrows():
            lines.append(f'P{int(r["pid"])} | trial {int(r["trial"])} ({r["domain"]}) '
                         f'| pages={int(r["n_pages"])}  '
                         f'switches  time={int(r["switches_time"])} '
                         f'topic={int(r["switches_topic"])} '
                         f'typing={int(r["switches_typing"])}')
            lines.append(f'    topics:  {r["topic_sequence"]}')
            lines.append(f'    time:    {r["time_sequence"]}')
            lines.append(f'    typing:  {r["typing_sequence"]}')
        lines.append('-' * 92)

    text = '\n'.join(lines)
    ax.text(0.01, 0.99, text, transform=ax.transAxes,
              fontsize=7.5, family='monospace', va='top', color=TEXT_COLOR,
              linespacing=1.4)


def plot_topic_distribution(ax, suspect_seq_df, kept_question_df, lda):
    """Show topic frequency for suspects vs the rest (per-page topic share)."""
    ax.set_facecolor(BG_COLOR)
    suspect_topics = []
    for s in suspect_seq_df['topic_sequence']:
        for x in s.split(','):
            if x and x != '-1':
                suspect_topics.append(int(x))

    other_topics = []
    pids_suspect = set(suspect_seq_df['pid'].unique())
    pids, pid_trials = get_pids_and_trials([])  # placeholder; will compute below

    n_topics = max(suspect_topics + [0]) + 1
    n_topics = max(n_topics, 10)
    bins = np.arange(n_topics + 1) - 0.5
    s_counts, _ = np.histogram(suspect_topics, bins=bins)
    s_pct = s_counts / s_counts.sum() * 100 if s_counts.sum() > 0 else s_counts

    x = np.arange(n_topics)
    ax.bar(x, s_pct, color=EXCLUDED_COLOR, edgecolor='white', linewidth=0.5)
    for i, v in enumerate(s_pct):
        if v > 0:
            ax.text(i, v + 0.5, f'{v:.0f}%', ha='center', va='bottom',
                      fontsize=8, color=EXCLUDED_COLOR, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{i}' for i in x], fontsize=8)
    ax.set_xlabel('LDA topic id', color=LABEL_COLOR, fontweight='bold')
    ax.set_ylabel('% of page visits (suspects only)', color=LABEL_COLOR, fontweight='bold')
    ax.set_title('Topic Distribution Across Suspect Sequences',
                  color=TEXT_COLOR, fontweight='bold', fontsize=12)
    ax.tick_params(colors=MUTED_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, axis='y', zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_n_pages_avg(diag_df, question_df):
    """Compute average pages per kept question for each participant."""
    keep = question_df[~(question_df['excluded_pages'] | question_df['excluded_idle'])]
    avg_pages = keep.groupby('participant_id')['n_pages'].mean()
    diag_df['n_pages_avg'] = diag_df['participant_id'].map(avg_pages)
    return diag_df


def create_pdf(diag_df, suspect_seq_df, suspects, question_df, lda, output_path):
    with PdfPages(output_path) as pdf:
        # --- Page 1: Title + summary ---
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(BG_COLOR)
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
        ax.axis('off')

        ax.text(0.5, 0.96, 'M53: LDA Switch-Count Diagnostics',
                  transform=ax.transAxes, fontsize=22, fontweight='bold',
                  ha='center', va='top', color=TEXT_COLOR)
        ax.text(0.5, 0.90, 'Why does count_topic diverge from count_time and count_typing?',
                  transform=ax.transAxes, fontsize=12, ha='center', va='top',
                  color=MUTED_COLOR, style='italic')

        n = len(diag_df)
        n_flag = (diag_df['abs_residual'] >= RESIDUAL_FLAG_SD).sum()
        z_corr_tt = np.corrcoef(diag_df['z_time'], diag_df['z_topic'])[0, 1]
        z_corr_typ = np.corrcoef(diag_df['z_typing'], diag_df['z_topic'])[0, 1]
        z_corr_time_typ = np.corrcoef(diag_df['z_time'], diag_df['z_typing'])[0, 1]

        summary = (
            f'Final M52 sample: N = {n} participants\n\n'
            f'Pairwise correlations on raw counts (z-equivalent):\n'
            f'  Time x Typing : r = {z_corr_time_typ:+.3f}  (the two that "go together")\n'
            f'  Time x Topic  : r = {z_corr_tt:+.3f}\n'
            f'  Typing x Topic: r = {z_corr_typ:+.3f}\n\n'
            f'Residual flag rule:  |z_topic - mean(z_time, z_typing)| >= {RESIDUAL_FLAG_SD} SD\n'
            f'Participants flagged: {n_flag} / {n}\n\n'
            f'Top suspects deep-dived in pages 4-5: {len(suspects)} participants'
            f' (highest |residual|).\n\n'
            f'Reading guide:\n'
            f'  Page 2 - LDA residual ranking (which participants drive the divergence)\n'
            f'  Page 3 - Pairwise scatter plots of z-scored counts\n'
            f'  Page 4 - count_topic vs avg n_pages (computational stability)\n'
            f'  Page 5 - Page-by-page topic sequences for the top suspects\n'
        )
        ax.text(0.04, 0.78, summary, transform=ax.transAxes, fontsize=11,
                  family='monospace', va='top', color=TEXT_COLOR, linespacing=1.7)

        pdf.savefig(fig, facecolor=BG_COLOR)
        plt.close()

        # --- Page 2: Residual bar chart ---
        fig, ax = plt.subplots(figsize=(11, 14))
        fig.patch.set_facecolor(BG_COLOR)
        plot_zscore_table(ax, diag_df)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG_COLOR)
        plt.close()

        # --- Page 3: Scatter pair plots ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        fig.patch.set_facecolor(BG_COLOR)
        plot_scatter(axes[0], diag_df, 'z_time', 'z_topic',
                      'z(count_time)', 'z(count_topic)',
                      'z_time vs z_topic')
        plot_scatter(axes[1], diag_df, 'z_typing', 'z_topic',
                      'z(count_typing)', 'z(count_topic)',
                      'z_typing vs z_topic')
        plot_scatter(axes[2], diag_df, 'z_time', 'z_typing',
                      'z(count_time)', 'z(count_typing)',
                      'z_time vs z_typing  (the "agreeing" pair)')
        fig.suptitle('Pairwise Z-Score Scatter Plots',
                      fontsize=15, fontweight='bold', color=TEXT_COLOR, y=1.02)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches='tight')
        plt.close()

        # --- Page 4: count_topic vs n_pages ---
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.patch.set_facecolor(BG_COLOR)
        plot_count_vs_pages(axes[0], diag_df)
        plot_topic_distribution(axes[1], suspect_seq_df, question_df, lda)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG_COLOR)
        plt.close()

        # --- Page 5: Suspect sequences (text) ---
        fig = plt.figure(figsize=(11, 14))
        fig.patch.set_facecolor(BG_COLOR)
        ax = fig.add_axes([0.03, 0.02, 0.94, 0.94])
        plot_suspect_text(ax, suspect_seq_df, suspects)
        pdf.savefig(fig, facecolor=BG_COLOR)
        plt.close()

    print(f'Saved: {output_path}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M53] LDA Switch-Count Diagnostics')
    print('=' * 60)

    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    lda = load_lda_assignments()

    print('\n--- Building question-level data ---')
    question_df = build_question_data(trials)

    print('\n--- Applying M52 exclusions ---')
    avg_df, _ = apply_exclusions(question_df)

    print('\n--- Computing z-scores and LDA residuals ---')
    diag_df = compute_residuals(avg_df)
    diag_df = add_n_pages_avg(diag_df, question_df)

    suspects = diag_df.head(TOP_K_SUSPECTS)['participant_id'].tolist()
    print(f'  Top {len(suspects)} suspects by |LDA residual|: {suspects}')

    suspect_seq_df = collect_suspect_sequences(suspects, trials, lda)

    # Save CSVs
    diag_csv = OUTPUT_DIR / 'm53_lda_diagnostics.csv'
    diag_df.to_csv(diag_csv, index=False)
    print(f'\nSaved: {diag_csv}')

    seq_csv = OUTPUT_DIR / 'm53_lda_per_question.csv'
    suspect_seq_df.to_csv(seq_csv, index=False)
    print(f'Saved: {seq_csv}')

    # PDF
    pdf_path = OUTPUT_DIR / 'm53_lda_diagnostics.pdf'
    create_pdf(diag_df, suspect_seq_df, suspects, question_df, lda, pdf_path)

    # Print key takeaways
    n_flag = (diag_df['abs_residual'] >= RESIDUAL_FLAG_SD).sum()
    print(f'\n--- Summary ---')
    print(f'  Participants flagged (|residual| >= {RESIDUAL_FLAG_SD}): '
          f'{n_flag} / {len(diag_df)}')
    pos = (diag_df['lda_residual'] >= RESIDUAL_FLAG_SD).sum()
    neg = (diag_df['lda_residual'] <= -RESIDUAL_FLAG_SD).sum()
    print(f'    LDA inflated:  {pos}')
    print(f'    LDA deflated:  {neg}')


if __name__ == '__main__':
    main()

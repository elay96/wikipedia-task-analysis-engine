#!/usr/bin/env python3
"""
M80: Hunter vs Busybody classification (Zhou et al. 2024 inspired)
==================================================================
Operationalises Zhou et al. 2024's distinction between two Wikipedia browsing
styles, applied to OUR participants' Wikipedia sessions:

  - Hunter   : tight, focused trajectories. Returns to recently-visited topics.
               High topic-concentration + low transition entropy.
  - Busybody : scattered, broad trajectories. Jumps between unrelated topics.
               Low concentration + high entropy.

Per-participant features (over all page visits across all 6 trials):
  - n_visits                    total page visits (mapped to LDA topic)
  - n_unique_topics             how many of the 10 LDA topics they touched
  - topic_concentration         share of visits to their most-visited topic
  - topic_entropy               Shannon entropy of topic visit distribution
                                (normalised by log2(10))
  - transition_entropy          mean entropy of P(next_topic | current_topic)
                                across rows of transition matrix (also normalised)
  - same_topic_repeat_rate      fraction of consecutive visits in same topic

Classification: KMeans (k=2) on standardised
  [topic_concentration, transition_entropy].
Cluster with HIGHER topic_concentration is labelled 'hunter', the other 'busybody'.

Then we ask:
  Q1. Does Clumpy/Diffuse condition predict hunter/busybody? (chi-square)
  Q2. Do spatial-search features (M76) differ between hunters & busybodies?
       (Welch t-tests on 30 features, FDR-BH)

Inputs:
  data/cleaned_new/Game.csv
  data/cleaned_new/topic_model.json
  output/m76_spatial_features.csv

Outputs:
  output/m80_hunter_busybody_per_participant.csv
  output/m80_hunter_busybody_features.csv
  output/m80_hunter_busybody.pdf  (5 pages: features dist, clustering, condition,
                                              spatial differences, top-spatial)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from helpers import load_trials                                          # noqa: E402

OUTPUT_DIR = SCRIPT_DIR.parent / 'output'
DATA_DIR = SCRIPT_DIR.parent / 'data'
GAME_CSV = DATA_DIR / 'cleaned_new' / 'Game.csv'
TOPIC_MODEL = DATA_DIR / 'cleaned_new' / 'topic_model.json'
SPATIAL_IN = OUTPUT_DIR / 'm76_spatial_features.csv'

PER_PID_OUT = OUTPUT_DIR / 'm80_hunter_busybody_per_participant.csv'
DIFF_OUT = OUTPUT_DIR / 'm80_hunter_busybody_features.csv'
PDF_OUT = OUTPUT_DIR / 'm80_hunter_busybody.pdf'

N_TOPICS = 10
HUNTER_COLOR = '#1976D2'
BUSYBODY_COLOR = '#E65100'
CLUMPY_COLOR = '#C62828'
DIFFUSE_COLOR = '#2E7D32'
BG = '#FFFFFF'
TEXT_COLOR = '#1a1a1a'
GRID_COLOR = '#E0E0E0'

FDR_ALPHA = 0.05


def load_lda_assignments(topic_model_path):
    with open(topic_model_path, encoding='utf-8') as f:
        tm = json.load(f)
    return {slug.replace('_', ' '): int(np.argmax(dist))
            for slug, dist in tm['topic_distributions'].items()}


def per_pid_topic_features(trials, lda_map):
    """Aggregate page visits across ALL trials per participant, compute network
    style features."""
    by_pid = {}
    for t in trials:
        pid = int(t['pid'])
        by_pid.setdefault(pid, {'condition': t['condition'], 'visits': []})
        for pv in t['page_visits']:
            topic = lda_map.get(pv['title'])
            if topic is None:
                continue
            by_pid[pid]['visits'].append(topic)

    rows = []
    for pid, info in by_pid.items():
        visits = info['visits']
        n = len(visits)
        if n < 5:
            continue
        topic_counts = Counter(visits)
        n_unique = len(topic_counts)
        concentration = max(topic_counts.values()) / n
        # Shannon entropy of topic visit distribution
        probs = np.array([c / n for c in topic_counts.values()])
        topic_ent = -np.sum(probs * np.log2(probs)) / np.log2(N_TOPICS)
        # transition matrix
        trans = np.zeros((N_TOPICS, N_TOPICS))
        for a, b in zip(visits[:-1], visits[1:]):
            trans[a, b] += 1
        same_topic_rate = np.trace(trans) / max(trans.sum(), 1)
        # mean row-entropy of transition matrix (over rows that have outflows)
        row_ents = []
        for row in trans:
            s = row.sum()
            if s == 0:
                continue
            p = row / s
            p = p[p > 0]
            row_ents.append(-np.sum(p * np.log2(p)))
        if row_ents:
            trans_ent = float(np.mean(row_ents)) / np.log2(N_TOPICS)
        else:
            trans_ent = np.nan
        rows.append({
            'participant_id': pid,
            'condition': info['condition'],
            'n_visits': n,
            'n_unique_topics': n_unique,
            'topic_concentration': concentration,
            'topic_entropy': topic_ent,
            'transition_entropy': trans_ent,
            'same_topic_repeat_rate': same_topic_rate,
        })
    return pd.DataFrame(rows)


def cluster_hunter_busybody(df):
    feats = ['topic_concentration', 'transition_entropy']
    X = df[feats].dropna().values
    keep_idx = df.dropna(subset=feats).index
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    # Whichever cluster has higher mean topic_concentration is hunter
    mean_conc = [X[labels == k, 0].mean() for k in (0, 1)]
    hunter_label = int(np.argmax(mean_conc))
    style = ['hunter' if lab == hunter_label else 'busybody' for lab in labels]
    df = df.copy()
    df['style'] = ''
    df.loc[keep_idx, 'style'] = style
    df['style'] = df['style'].replace('', np.nan)
    centers = scaler.inverse_transform(km.cluster_centers_)
    return df, {'centers': centers, 'hunter_idx': hunter_label}


# ---------- statistical tests -----------------------------------------------

def fdr_bh(pvals, alpha=FDR_ALPHA):
    p = np.asarray(pvals, dtype=float)
    n_total = len(p)
    finite = np.isfinite(p)
    m = int(finite.sum())
    out = np.full(n_total, np.nan)
    if m == 0:
        return out
    idx = np.where(finite)[0]
    pf = p[idx]
    order = np.argsort(pf)
    ranked = pf[order]
    adj = ranked * m / (np.arange(m) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out_finite = np.empty(m)
    out_finite[order] = adj
    out[idx] = out_finite
    return out


def spatial_diff_test(df_with_style):
    spatial_cols = [c for c in df_with_style.columns
                    if c not in ('participant_id', 'condition', 'n_visits',
                                 'n_unique_topics', 'topic_concentration',
                                 'topic_entropy', 'transition_entropy',
                                 'same_topic_repeat_rate', 'style')]
    rows = []
    for col in spatial_cols:
        h = df_with_style.loc[df_with_style['style'] == 'hunter', col].dropna().values
        b = df_with_style.loc[df_with_style['style'] == 'busybody', col].dropna().values
        if len(h) < 2 or len(b) < 2:
            rows.append({'feature': col, 'n_hunter': len(h), 'n_busybody': len(b),
                         'mean_hunter': np.nan, 'mean_busybody': np.nan,
                         't': np.nan, 'p': np.nan, 'cohens_d': np.nan})
            continue
        s1, s2 = np.var(h, ddof=1), np.var(b, ddof=1)
        pooled = np.sqrt(((len(h) - 1) * s1 + (len(b) - 1) * s2) / (len(h) + len(b) - 2))
        d = (np.mean(h) - np.mean(b)) / pooled if pooled > 0 else np.nan
        t_res = sp_stats.ttest_ind(h, b, equal_var=False)
        rows.append({
            'feature': col, 'n_hunter': len(h), 'n_busybody': len(b),
            'mean_hunter': float(np.mean(h)),
            'mean_busybody': float(np.mean(b)),
            't': float(t_res.statistic), 'p': float(t_res.pvalue),
            'cohens_d': float(d) if np.isfinite(d) else np.nan,
        })
    res = pd.DataFrame(rows)
    res['p_fdr'] = fdr_bh(res['p'].values)
    res['fdr_significant'] = res['p_fdr'] < FDR_ALPHA
    return res


# ---------- plotting ---------------------------------------------------------

def page_feature_distributions(pdf, df):
    feats = ['topic_concentration', 'topic_entropy', 'transition_entropy',
             'same_topic_repeat_rate', 'n_visits', 'n_unique_topics']
    fig = plt.figure(figsize=(13, 8.5), facecolor=BG)
    fig.suptitle('M80  Per-participant Wikipedia browsing style features',
                 fontsize=12, color=TEXT_COLOR, y=0.97)

    for i, f in enumerate(feats):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.set_facecolor(BG)
        ax.hist(df[f].dropna().values, bins=20, color='#1976D2',
                edgecolor='#FFFFFF', alpha=0.85)
        ax.axvline(df[f].mean(), color='#C62828', lw=1.0, ls='--',
                   label=f'mean={df[f].mean():.2f}')
        ax.set_title(f, fontsize=10, color=TEXT_COLOR)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page_clusters_scatter(pdf, df, info):
    fig = plt.figure(figsize=(10, 8), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)

    h_mask = df['style'] == 'hunter'
    b_mask = df['style'] == 'busybody'
    ax.scatter(df.loc[h_mask, 'topic_concentration'],
               df.loc[h_mask, 'transition_entropy'],
               s=44, color=HUNTER_COLOR, alpha=0.7, label=f'Hunter (n={h_mask.sum()})',
               edgecolors='#FFFFFF', lw=0.5)
    ax.scatter(df.loc[b_mask, 'topic_concentration'],
               df.loc[b_mask, 'transition_entropy'],
               s=44, color=BUSYBODY_COLOR, alpha=0.7, label=f'Busybody (n={b_mask.sum()})',
               edgecolors='#FFFFFF', lw=0.5)

    centers = info['centers']
    ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=180, color='#1a1a1a',
               edgecolors='#FFFFFF', lw=1.2, label='cluster centers', zorder=5)

    ax.set_xlabel('topic_concentration  (max topic share / total visits)',
                  color=TEXT_COLOR)
    ax.set_ylabel('transition_entropy  (mean row entropy / log2(K))',
                  color=TEXT_COLOR)
    ax.set_title('M80  Hunter vs Busybody clustering  (KMeans k=2 on standardised features)',
                 fontsize=11, color=TEXT_COLOR)
    ax.legend(fontsize=10)
    ax.grid(True, color=GRID_COLOR, alpha=0.4)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page_condition_vs_style(pdf, df):
    cross = pd.crosstab(df['condition'], df['style'])
    cross = cross.reindex(columns=['hunter', 'busybody'])
    if 'hunter' not in cross.columns:
        cross['hunter'] = 0
    if 'busybody' not in cross.columns:
        cross['busybody'] = 0

    chi2, p, dof, exp = sp_stats.chi2_contingency(cross.values)

    fig = plt.figure(figsize=(11, 7), facecolor=BG)
    fig.suptitle(
        f'M80  Foraging condition vs Wikipedia style\n'
        f'chi-square = {chi2:.2f}, df={dof}, p={p:.3f}',
        fontsize=12, color=TEXT_COLOR, y=0.98,
    )

    ax = fig.add_subplot(121)
    ax.set_facecolor(BG)
    rates = cross.div(cross.sum(axis=1), axis=0) * 100
    rates.plot(kind='bar', ax=ax, color=[HUNTER_COLOR, BUSYBODY_COLOR],
               edgecolor='#FFFFFF', lw=0.8, width=0.7)
    ax.set_ylabel('% of participants', color=TEXT_COLOR)
    ax.set_xlabel('')
    ax.set_title('Style proportions within each condition', fontsize=10, color=TEXT_COLOR)
    ax.legend(title='style')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis='y', color=GRID_COLOR, alpha=0.4)

    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    table_text = [['', 'hunter', 'busybody', 'total']]
    for c in cross.index:
        table_text.append([c, str(int(cross.loc[c, 'hunter'])),
                           str(int(cross.loc[c, 'busybody'])),
                           str(int(cross.loc[c].sum()))])
    table_text.append(['total',
                       str(int(cross['hunter'].sum())),
                       str(int(cross['busybody'].sum())),
                       str(int(cross.values.sum()))])
    tbl = ax2.table(cellText=table_text, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.6)
    for j in range(4):
        tbl[0, j].set_text_props(weight='bold')
        tbl[0, j].set_facecolor('#F0F4F8')

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page_spatial_diff_table(pdf, res, n_h, n_b):
    sig = res[res['fdr_significant']].sort_values('p_fdr')
    if len(sig) == 0:
        sig = res.reindex(res['cohens_d'].abs().sort_values(ascending=False).index).head(15)
        title = (f'M80  Spatial features by Wikipedia style  '
                 f'(Hunter N={n_h}, Busybody N={n_b})\n'
                 f'No FDR-significant features. Top 15 by |Cohen\'s d|:')
    else:
        title = (f'M80  Spatial features by Wikipedia style  '
                 f'(Hunter N={n_h}, Busybody N={n_b})\n'
                 f'{len(sig)} FDR-significant features:')

    fig = plt.figure(figsize=(13, max(7, 0.45 * len(sig) + 2)), facecolor=BG)
    fig.suptitle(title, fontsize=11, color=TEXT_COLOR, y=0.97)

    cols = [
        ('feature', 'Feature'),
        ('n_hunter', 'N_H'), ('n_busybody', 'N_B'),
        ('mean_hunter', 'Mean_H'), ('mean_busybody', 'Mean_B'),
        ('cohens_d', 'd'), ('t', 't'), ('p', 'p'), ('p_fdr', 'p_FDR'),
    ]
    cell_text = []
    cell_colors = []
    for _, r in sig.iterrows():
        row = []
        for k, _ in cols:
            v = r[k]
            if k == 'feature':
                row.append(str(v))
            elif k in ('n_hunter', 'n_busybody'):
                row.append(f'{int(v)}')
            elif k in ('mean_hunter', 'mean_busybody'):
                row.append(f'{v:.2f}' if pd.notna(v) else '-')
            elif k in ('cohens_d', 't'):
                row.append(f'{v:+.2f}' if pd.notna(v) else '-')
            else:
                if pd.isna(v):
                    row.append('-')
                elif v < 0.001:
                    row.append('<.001')
                else:
                    row.append(f'{v:.3f}')
        cell_text.append(row)
        cell_colors.append(['#E8F5E9'] * len(cols) if r['fdr_significant']
                           else [BG] * len(cols))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    ax.axis('off')
    tbl = ax.table(cellText=cell_text, colLabels=[c[1] for c in cols],
                   cellColours=cell_colors, loc='upper center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    for j in range(len(cols)):
        tbl[0, j].set_text_props(weight='bold', color=TEXT_COLOR)
        tbl[0, j].set_facecolor('#F0F4F8')
    for i in range(1, len(cell_text) + 1):
        tbl[i, 0].set_text_props(ha='left')
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page_top_spatial_strip(pdf, df, res, n_show=6):
    sig = res[res['fdr_significant']].sort_values('p_fdr')
    if len(sig) == 0:
        sig = res.reindex(res['cohens_d'].abs().sort_values(ascending=False).index)
    feats = sig.head(n_show)['feature'].tolist()
    cols = 3
    rows = (len(feats) + cols - 1) // cols
    fig = plt.figure(figsize=(13, 3.0 * rows + 0.8), facecolor=BG)
    fig.suptitle('M80  Spatial features by Wiki style: top contrasts',
                 fontsize=12, color=TEXT_COLOR, y=0.99)

    rng = np.random.default_rng(42)
    for i, feat in enumerate(feats):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor(BG)
        h = df.loc[df['style'] == 'hunter', feat].dropna().values
        b = df.loc[df['style'] == 'busybody', feat].dropna().values
        ax.scatter(rng.normal(0, 0.06, len(h)), h, s=18, color=HUNTER_COLOR,
                   alpha=0.55, edgecolors='none')
        ax.scatter(1 + rng.normal(0, 0.06, len(b)), b, s=18, color=BUSYBODY_COLOR,
                   alpha=0.55, edgecolors='none')
        for x_pos, vals, col in ((0, h, HUNTER_COLOR), (1, b, BUSYBODY_COLOR)):
            if len(vals) >= 2:
                m = np.mean(vals)
                ax.plot([x_pos - 0.18, x_pos + 0.18], [m, m], color=col, lw=2.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Hunter', 'Busybody'], fontsize=9)
        ax.set_xlim(-0.5, 1.5)
        rrow = res[res['feature'] == feat].iloc[0]
        ax.set_title(f'{feat}\nd={rrow["cohens_d"]:+.2f}, p_FDR='
                     f'{"<.001" if rrow["p_fdr"] < 0.001 else f"{rrow["p_fdr"]:.3f}"}',
                     fontsize=9)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis='y', color=GRID_COLOR, alpha=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


# ---------- main -------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print('loading trials...')
    trials = load_trials(GAME_CSV)
    print(f'  {len(trials)} trials loaded')
    lda = load_lda_assignments(TOPIC_MODEL)

    feats = per_pid_topic_features(trials, lda)
    print(f'pid-level features: {feats.shape}')
    print(feats[['n_visits', 'topic_concentration', 'transition_entropy']].describe())

    feats, info = cluster_hunter_busybody(feats)
    print(f'\ncluster centers (concentration, transition_entropy): {info["centers"]}')
    print('style counts:', feats['style'].value_counts().to_dict())

    # merge to spatial features
    spatial = pd.read_csv(SPATIAL_IN)
    spatial['condition'] = spatial['condition'].str.lower()
    feats['condition'] = feats['condition'].str.lower()
    merged = feats.merge(spatial.drop(columns=['condition']), on='participant_id', how='inner')
    print(f'merged with spatial: {merged.shape}')

    feats.to_csv(PER_PID_OUT, index=False)
    print(f'wrote {PER_PID_OUT.name}')

    diff = spatial_diff_test(merged)
    diff.to_csv(DIFF_OUT, index=False)
    print(f'wrote {DIFF_OUT.name}')

    n_h = int((merged['style'] == 'hunter').sum())
    n_b = int((merged['style'] == 'busybody').sum())
    n_sig = int(diff['fdr_significant'].sum())
    print(f'\n  Hunters: {n_h}, Busybodies: {n_b}')
    print(f'  Spatial features differing FDR (alpha={FDR_ALPHA}): {n_sig}/{len(diff)}')

    # condition x style
    cross = pd.crosstab(merged['condition'], merged['style'])
    chi2, p_chi, _, _ = sp_stats.chi2_contingency(cross.values)
    print(f'\n  chi-square (condition x style): chi2={chi2:.2f}, p={p_chi:.3f}')
    print(f'  contingency:\n{cross.to_string()}')

    print('\n  Top 5 spatial features by |d|:')
    top = diff.reindex(diff['cohens_d'].abs().sort_values(ascending=False).index).head(5)
    for _, r in top.iterrows():
        marker = ' *' if r['fdr_significant'] else '  '
        print(f'   {marker} {r["feature"]:35s}  d={r["cohens_d"]:+.2f}  '
              f'p={r["p"]:.4f}  p_FDR={r["p_fdr"]:.4f}')

    with PdfPages(PDF_OUT) as pdf:
        page_feature_distributions(pdf, feats)
        page_clusters_scatter(pdf, feats, info)
        page_condition_vs_style(pdf, merged)
        page_spatial_diff_table(pdf, diff, n_h, n_b)
        page_top_spatial_strip(pdf, merged, diff)
    print(f'wrote {PDF_OUT.name}')


if __name__ == '__main__':
    main()

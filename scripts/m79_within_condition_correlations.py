#!/usr/bin/env python3
"""
M79: Within-condition Wikipedia correlations
============================================
Same as M78 (30 spatial x 6 wiki = 180 correlations) but computed SEPARATELY
within each condition:

  - Clumpy  (N=72)
  - Diffuse (N=60)

Rationale: M78 collapsed across condition found 0/180 FDR-significant pairs
with max |r|=0.25. A within-condition correlation could be hidden if the
direction reverses between conditions.

FDR-BH applied independently within each condition.

Outputs:
  output/m79_within_condition_correlations.csv      # 360 rows (180 per side)
  output/m79_within_condition_correlations.pdf      # heatmaps + comparison
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'output'

SPATIAL_IN = OUTPUT_DIR / 'm76_spatial_features.csv'
WIKI_M72_IN = OUTPUT_DIR / 'm72_new_per_participant.csv'
WIKI_M73_IN = OUTPUT_DIR / 'm73_new_per_participant_entropy.csv'

CSV_OUT = OUTPUT_DIR / 'm79_within_condition_correlations.csv'
PDF_OUT = OUTPUT_DIR / 'm79_within_condition_correlations.pdf'

WIKI_FEATURES = [
    'count_time', 'count_topic', 'count_typing',
    'PC1', 'seq_typing_mean_run_explore', 'seq_typing_entropy',
]

FDR_ALPHA = 0.05

BG = '#FFFFFF'
TEXT_COLOR = '#1a1a1a'
GRID_COLOR = '#E0E0E0'


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


def load_merged():
    spatial = pd.read_csv(SPATIAL_IN)
    m72 = pd.read_csv(WIKI_M72_IN)
    m73 = pd.read_csv(WIKI_M73_IN)
    m72_keep = m72[['participant_id', 'count_time', 'count_topic', 'count_typing',
                    'PC1', 'seq_typing_mean_run_explore']]
    m73_keep = m73[['participant_id', 'seq_typing_entropy']]
    return (spatial
            .merge(m72_keep, on='participant_id', how='inner')
            .merge(m73_keep, on='participant_id', how='inner'))


def correlations_in_subset(df_sub, spatial_feats, condition_label):
    rows = []
    for sf in spatial_feats:
        for wf in WIKI_FEATURES:
            x = df_sub[sf].values
            y = df_sub[wf].values
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 4:
                rows.append({
                    'condition': condition_label,
                    'spatial': sf, 'wiki': wf, 'n': n,
                    'r_pearson': np.nan, 'p_pearson': np.nan,
                    'rho_spearman': np.nan, 'p_spearman': np.nan,
                })
                continue
            r_p, p_p = sp_stats.pearsonr(x[mask], y[mask])
            r_s, p_s = sp_stats.spearmanr(x[mask], y[mask])
            rows.append({
                'condition': condition_label,
                'spatial': sf, 'wiki': wf, 'n': n,
                'r_pearson': float(r_p), 'p_pearson': float(p_p),
                'rho_spearman': float(r_s), 'p_spearman': float(p_s),
            })
    sub_res = pd.DataFrame(rows)
    sub_res['p_fdr'] = fdr_bh(sub_res['p_pearson'].values)
    sub_res['fdr_significant'] = sub_res['p_fdr'] < FDR_ALPHA
    return sub_res


# ---------- plotting ---------------------------------------------------------

def heatmap_panel(ax, R, P_FDR, spatial_feats, wiki_feats, title):
    im = ax.imshow(R, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    n_sp, n_wk = R.shape
    for i in range(n_sp):
        for j in range(n_wk):
            r = R[i, j]
            if not np.isfinite(r):
                continue
            star = '*' if (np.isfinite(P_FDR[i, j]) and P_FDR[i, j] < FDR_ALPHA) else ''
            color = '#1a1a1a' if abs(r) < 0.4 else '#FFFFFF'
            ax.text(j, i, f'{r:+.2f}{star}', ha='center', va='center',
                    fontsize=6.5, color=color)
    ax.set_xticks(range(n_wk))
    ax.set_xticklabels(wiki_feats, rotation=30, ha='right', fontsize=8, color=TEXT_COLOR)
    ax.set_yticks(range(n_sp))
    ax.set_yticklabels(spatial_feats, fontsize=7, color=TEXT_COLOR)
    ax.set_title(title, fontsize=10, color=TEXT_COLOR)
    return im


def page_dual_heatmap(pdf, res, spatial_feats, wiki_feats):
    n_sp = len(spatial_feats)
    n_wk = len(wiki_feats)
    fig = plt.figure(figsize=(15, max(11, 0.36 * n_sp + 1.5)), facecolor=BG)
    fig.suptitle(
        f'M79  Within-condition correlations (Spatial 30 x Wiki 6 x 2 conditions = 360)\n'
        f'* = FDR-BH significant within that condition (alpha={FDR_ALPHA})',
        fontsize=12, color=TEXT_COLOR, y=0.985,
    )

    for k, cond in enumerate(['Clumpy', 'Diffuse']):
        sub = res[res['condition'] == cond]
        R = np.full((n_sp, n_wk), np.nan)
        P_FDR = np.full((n_sp, n_wk), np.nan)
        for _, r in sub.iterrows():
            i = spatial_feats.index(r['spatial'])
            j = wiki_feats.index(r['wiki'])
            R[i, j] = r['r_pearson']
            P_FDR[i, j] = r['p_fdr']
        ax = fig.add_subplot(1, 2, k + 1)
        ax.set_facecolor(BG)
        n_sub = sub['n'].max()
        im = heatmap_panel(ax, R, P_FDR, spatial_feats, wiki_feats,
                           f'{cond} (N={int(n_sub)})')

    cbar = fig.colorbar(im, ax=fig.axes, fraction=0.025, pad=0.02)
    cbar.set_label('Pearson r', color=TEXT_COLOR)
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page_clumpy_vs_diffuse_scatter(pdf, res):
    pivot_r = res.pivot_table(index=['spatial', 'wiki'], columns='condition',
                              values='r_pearson').reset_index()
    pivot_r = pivot_r.dropna(subset=['Clumpy', 'Diffuse'])

    fig = plt.figure(figsize=(10, 9), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)

    flips = (np.sign(pivot_r['Clumpy']) != np.sign(pivot_r['Diffuse'])) & \
            ((pivot_r['Clumpy'].abs() > 0.2) | (pivot_r['Diffuse'].abs() > 0.2))

    ax.scatter(pivot_r.loc[~flips, 'Clumpy'], pivot_r.loc[~flips, 'Diffuse'],
               s=22, color='#9E9E9E', alpha=0.55, label='consistent or weak')
    ax.scatter(pivot_r.loc[flips, 'Clumpy'], pivot_r.loc[flips, 'Diffuse'],
               s=42, color='#C62828', alpha=0.85, label='sign flip (|r|>0.2)')

    lim = 0.7
    ax.plot([-lim, lim], [-lim, lim], color='#999999', lw=0.8, ls='--', label='y=x')
    ax.axhline(0, color='#CCCCCC', lw=0.5)
    ax.axvline(0, color='#CCCCCC', lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('r in Clumpy (N=72)', color=TEXT_COLOR)
    ax.set_ylabel('r in Diffuse (N=60)', color=TEXT_COLOR)
    ax.set_title('M79  Each (spatial, wiki) pair: r in Clumpy vs Diffuse',
                 fontsize=11, color=TEXT_COLOR)
    ax.legend(loc='lower right')
    ax.grid(True, color=GRID_COLOR, alpha=0.4)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    flip_rows = pivot_r[flips]
    if len(flip_rows) > 0:
        for _, r in flip_rows.iterrows():
            ax.annotate(f'{r["spatial"][:18]}~{r["wiki"][:14]}',
                        xy=(r['Clumpy'], r['Diffuse']), fontsize=6,
                        color='#C62828',
                        xytext=(5, 5), textcoords='offset points')
    fig.tight_layout()
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page_top_pairs_table(pdf, res):
    sig = res[res['fdr_significant']].copy()
    if len(sig) == 0:
        sig = res.reindex(res['r_pearson'].abs().sort_values(ascending=False).index).head(20)
        title = 'M79  No FDR-sig pairs in either condition.  Top 20 by |r|:'
    else:
        sig = sig.sort_values('p_fdr')
        title = f'M79  {len(sig)} FDR-significant pairs:'

    fig = plt.figure(figsize=(13, max(7, 0.5 * len(sig) + 2)), facecolor=BG)
    fig.suptitle(title, fontsize=12, color=TEXT_COLOR, y=0.97)

    cols = [
        ('condition', 'Cond'),
        ('spatial', 'Spatial feature'),
        ('wiki', 'Wiki measure'),
        ('n', 'N'),
        ('r_pearson', 'r'), ('p_pearson', 'p'),
        ('rho_spearman', 'rho'),
        ('p_fdr', 'p_FDR'),
    ]
    cell_text = []
    cell_colors = []
    for _, r in sig.iterrows():
        row = []
        for k, _ in cols:
            v = r[k]
            if k in ('condition', 'spatial', 'wiki'):
                row.append(str(v))
            elif k == 'n':
                row.append(f'{int(v)}')
            elif k in ('r_pearson', 'rho_spearman'):
                row.append(f'{v:+.2f}' if pd.notna(v) else '-')
            else:
                if pd.isna(v):
                    row.append('-')
                elif v < 0.001:
                    row.append('<.001')
                else:
                    row.append(f'{v:.3f}')
        cell_text.append(row)
        cell_colors.append(['#E8F5E9'] * len(cols) if r['fdr_significant'] else [BG] * len(cols))

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.93])
    ax.axis('off')
    tbl = ax.table(
        cellText=cell_text,
        colLabels=[c[1] for c in cols],
        cellColours=cell_colors,
        loc='upper center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_text_props(weight='bold', color=TEXT_COLOR)
        cell.set_facecolor('#F0F4F8')
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_merged()
    print(f'merged: {df.shape}')

    spatial_feats = [
        c for c in df.columns
        if c not in ['participant_id', 'condition'] + WIKI_FEATURES
    ]

    parts = []
    for cond, label in (('Clumpy', 'Clumpy'), ('Diffuse', 'Diffuse')):
        sub = df[df['condition'] == cond]
        sub_res = correlations_in_subset(sub, spatial_feats, label)
        n_sig = int(sub_res['fdr_significant'].sum())
        max_abs_r = float(sub_res['r_pearson'].abs().max())
        print(f'  {label}: N={len(sub)}, max |r|={max_abs_r:.3f}, '
              f'FDR-sig={n_sig}/{len(sub_res)}')
        parts.append(sub_res)
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(CSV_OUT, index=False)
    print(f'wrote {CSV_OUT.name}')

    # any sign flip with absolute >0.2 in either?
    pivot = res.pivot_table(index=['spatial', 'wiki'], columns='condition',
                            values='r_pearson').reset_index()
    pivot = pivot.dropna(subset=['Clumpy', 'Diffuse'])
    flips = pivot[(np.sign(pivot['Clumpy']) != np.sign(pivot['Diffuse'])) &
                  ((pivot['Clumpy'].abs() > 0.2) | (pivot['Diffuse'].abs() > 0.2))]
    print(f'  sign-flip pairs with |r|>0.2 in either condition: {len(flips)}')
    if len(flips) > 0:
        print('  flips:')
        for _, r in flips.head(10).iterrows():
            print(f'    {r["spatial"]:30s} ~ {r["wiki"]:30s}  '
                  f'Clumpy={r["Clumpy"]:+.2f}  Diffuse={r["Diffuse"]:+.2f}')

    print('\n  Top 5 |r| in Clumpy:')
    top_c = res[res['condition'] == 'Clumpy'].reindex(
        res[res['condition'] == 'Clumpy']['r_pearson'].abs().sort_values(ascending=False).index
    ).head(5)
    for _, r in top_c.iterrows():
        print(f'    {r["spatial"]:30s} ~ {r["wiki"]:30s}  r={r["r_pearson"]:+.2f}  p_FDR={r["p_fdr"]:.4f}')

    print('\n  Top 5 |r| in Diffuse:')
    top_d = res[res['condition'] == 'Diffuse'].reindex(
        res[res['condition'] == 'Diffuse']['r_pearson'].abs().sort_values(ascending=False).index
    ).head(5)
    for _, r in top_d.iterrows():
        print(f'    {r["spatial"]:30s} ~ {r["wiki"]:30s}  r={r["r_pearson"]:+.2f}  p_FDR={r["p_fdr"]:.4f}')

    with PdfPages(PDF_OUT) as pdf:
        page_dual_heatmap(pdf, res, spatial_feats, WIKI_FEATURES)
        page_clumpy_vs_diffuse_scatter(pdf, res)
        page_top_pairs_table(pdf, res)
    print(f'wrote {PDF_OUT.name}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
M78: Spatial-Search x Wikipedia Correlations
============================================
Correlates each of 30 spatial-search features (M76) with each of 6 Wikipedia
metrics (M72 + M73), collapsed across condition.

Inputs:
  output/m76_spatial_features.csv               # 30 features (mean+sd) per pid
  output/m72_new_per_participant.csv             # count_time, count_topic,
                                                 # count_typing, PC1,
                                                 # seq_typing_mean_run_explore
  output/m73_new_per_participant_entropy.csv     # seq_typing_entropy

Inner join on participant_id -> N=132 (72 Clumpy + 60 Diffuse).

For each (spatial, wiki) pair:
  - Pearson r + p
  - Spearman rho + p
  - Partial Pearson controlling for condition (residualised; sanity check)

FDR-BH on the 180 Pearson p-values.

Outputs:
  output/m78_correlations.csv                    # long table
  output/m78_correlations.pdf                    # heatmap + top-10 + collapsed-vs-partial
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

CSV_OUT = OUTPUT_DIR / 'm78_correlations.csv'
PDF_OUT = OUTPUT_DIR / 'm78_correlations.pdf'

WIKI_FEATURES = [
    'count_time',
    'count_topic',
    'count_typing',
    'PC1',
    'seq_typing_mean_run_explore',
    'seq_typing_entropy',
]

FDR_ALPHA = 0.05

BG = '#FFFFFF'
TEXT_COLOR = '#1a1a1a'
MUTED_COLOR = '#666666'
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
    m72_keep = m72[['participant_id'] + [
        c for c in [
            'count_time', 'count_topic', 'count_typing',
            'PC1', 'seq_typing_mean_run_explore'
        ] if c in m72.columns
    ]]
    m73_keep = m73[['participant_id', 'seq_typing_entropy']]
    merged = (
        spatial
        .merge(m72_keep, on='participant_id', how='inner')
        .merge(m73_keep, on='participant_id', how='inner')
    )
    return merged


def partial_pearson_residualise_condition(x, y, cond_indicator):
    """Partial Pearson r between x and y after residualising out a binary
    condition indicator (0/1). NaN-safe (drops rows with any NaN).
    Returns (r_partial, p_partial, n)."""
    arr = np.column_stack([x, y, cond_indicator]).astype(float)
    keep = np.all(np.isfinite(arr), axis=1)
    arr = arr[keep]
    n = len(arr)
    if n < 4:
        return np.nan, np.nan, n
    xv, yv, cv = arr[:, 0], arr[:, 1], arr[:, 2]
    # OLS residuals against [1, cv]
    X = np.column_stack([np.ones(n), cv])
    bx, *_ = np.linalg.lstsq(X, xv, rcond=None)
    by, *_ = np.linalg.lstsq(X, yv, rcond=None)
    rx = xv - X @ bx
    ry = yv - X @ by
    r, p = sp_stats.pearsonr(rx, ry)
    # df adjustment for one covariate
    df_adj = n - 2 - 1
    if df_adj <= 0 or abs(r) >= 1:
        return float(r), np.nan, n
    t = r * np.sqrt(df_adj / max(1 - r**2, 1e-12))
    p_adj = 2 * (1 - sp_stats.t.cdf(abs(t), df=df_adj))
    return float(r), float(p_adj), n


def run_correlations(df, spatial_feats, wiki_feats):
    cond_indicator = (df['condition'] == 'Clumpy').astype(float).values
    rows = []
    for sf in spatial_feats:
        for wf in wiki_feats:
            x = df[sf].values
            y = df[wf].values
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 4:
                rows.append({
                    'spatial': sf, 'wiki': wf, 'n': n,
                    'r_pearson': np.nan, 'p_pearson': np.nan,
                    'rho_spearman': np.nan, 'p_spearman': np.nan,
                    'r_partial': np.nan, 'p_partial': np.nan,
                })
                continue
            r_p, p_p = sp_stats.pearsonr(x[mask], y[mask])
            r_s, p_s = sp_stats.spearmanr(x[mask], y[mask])
            r_par, p_par, _ = partial_pearson_residualise_condition(x, y, cond_indicator)
            rows.append({
                'spatial': sf, 'wiki': wf, 'n': n,
                'r_pearson': float(r_p), 'p_pearson': float(p_p),
                'rho_spearman': float(r_s), 'p_spearman': float(p_s),
                'r_partial': r_par, 'p_partial': p_par,
            })
    res = pd.DataFrame(rows)
    res['p_fdr'] = fdr_bh(res['p_pearson'].values)
    res['fdr_significant'] = res['p_fdr'] < FDR_ALPHA
    return res


# ---------- plotting ---------------------------------------------------------

def page_heatmap(pdf, res, spatial_feats, wiki_feats):
    n_sp = len(spatial_feats)
    n_wk = len(wiki_feats)
    R = np.full((n_sp, n_wk), np.nan)
    P_FDR = np.full((n_sp, n_wk), np.nan)
    for _, r in res.iterrows():
        i = spatial_feats.index(r['spatial'])
        j = wiki_feats.index(r['wiki'])
        R[i, j] = r['r_pearson']
        P_FDR[i, j] = r['p_fdr']

    fig = plt.figure(figsize=(11, max(10, 0.36 * n_sp + 1.8)), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    im = ax.imshow(R, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')

    for i in range(n_sp):
        for j in range(n_wk):
            r = R[i, j]
            if not np.isfinite(r):
                continue
            star = '*' if (np.isfinite(P_FDR[i, j]) and P_FDR[i, j] < FDR_ALPHA) else ''
            txt_color = '#1a1a1a' if abs(r) < 0.4 else '#FFFFFF'
            ax.text(j, i, f'{r:+.2f}{star}', ha='center', va='center',
                    fontsize=7, color=txt_color)

    ax.set_xticks(range(n_wk))
    ax.set_xticklabels(wiki_feats, rotation=30, ha='right', fontsize=9, color=TEXT_COLOR)
    ax.set_yticks(range(n_sp))
    ax.set_yticklabels(spatial_feats, fontsize=8, color=TEXT_COLOR)
    ax.set_title(
        f'M78 Pearson r (Spatial 30 x Wiki 6 = 180)   * = FDR-BH < {FDR_ALPHA}',
        fontsize=11, color=TEXT_COLOR,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Pearson r', color=TEXT_COLOR)
    fig.tight_layout()
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page_top_correlations(pdf, res):
    top = res.reindex(res['r_pearson'].abs().sort_values(ascending=False).index).head(20)
    fig = plt.figure(figsize=(13, 10), facecolor=BG)
    fig.suptitle(
        'M78 Top 20 correlations by |r_pearson|',
        fontsize=12, color=TEXT_COLOR, y=0.985,
    )

    cols = [
        ('spatial', 'Spatial feature'),
        ('wiki', 'Wiki measure'),
        ('n', 'N'),
        ('r_pearson', 'r'), ('p_pearson', 'p'),
        ('rho_spearman', 'rho'), ('p_spearman', 'p_S'),
        ('r_partial', 'r|cond'), ('p_partial', 'p|cond'),
        ('p_fdr', 'p_FDR'),
    ]

    cell_text = []
    cell_colors = []
    for _, r in top.iterrows():
        row = []
        for k, _ in cols:
            v = r[k]
            if k in ('spatial', 'wiki'):
                row.append(str(v))
            elif k == 'n':
                row.append(f'{int(v)}')
            elif k in ('r_pearson', 'rho_spearman', 'r_partial'):
                row.append(f'{v:+.2f}' if pd.notna(v) else '-')
            else:
                if pd.isna(v):
                    row.append('-')
                elif v < 0.001:
                    row.append('<.001')
                else:
                    row.append(f'{v:.3f}')
            cell_colors_row = ['#E8F5E9'] * len(cols) if r['fdr_significant'] else [BG] * len(cols)
        cell_text.append(row)
        cell_colors.append(cell_colors_row)

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
    for i in range(1, len(cell_text) + 1):
        tbl[i, 0].set_text_props(ha='left')
        tbl[i, 1].set_text_props(ha='left')

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page_collapsed_vs_partial(pdf, res):
    valid = res.dropna(subset=['r_pearson', 'r_partial']).copy()
    fig = plt.figure(figsize=(9, 9), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)

    sig_mask = valid['fdr_significant']
    ax.scatter(valid.loc[~sig_mask, 'r_pearson'], valid.loc[~sig_mask, 'r_partial'],
               s=22, color='#9E9E9E', alpha=0.55, label='not FDR-sig')
    ax.scatter(valid.loc[sig_mask, 'r_pearson'], valid.loc[sig_mask, 'r_partial'],
               s=34, color='#2E7D32', alpha=0.85, label='FDR-sig (collapsed)')

    lim = 0.85
    ax.plot([-lim, lim], [-lim, lim], color='#999999', lw=0.8, ls='--')
    ax.axhline(0, color='#CCCCCC', lw=0.5)
    ax.axvline(0, color='#CCCCCC', lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('r_pearson  (collapsed across condition)', color=TEXT_COLOR)
    ax.set_ylabel('r_partial  (controlling for condition)', color=TEXT_COLOR)
    ax.set_title(
        'M78 Robustness: each correlation, collapsed vs partial-out condition',
        fontsize=11, color=TEXT_COLOR,
    )
    ax.legend(loc='lower right')
    ax.grid(True, color=GRID_COLOR, alpha=0.4)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    # annotate any flips (sign change between collapsed and partial)
    flips = valid[(np.sign(valid['r_pearson']) != np.sign(valid['r_partial'])) &
                  (valid['r_pearson'].abs() > 0.15)]
    for _, r in flips.iterrows():
        ax.annotate(f'{r["spatial"][:18]}~{r["wiki"][:14]}',
                    xy=(r['r_pearson'], r['r_partial']), fontsize=6,
                    color='#C62828',
                    xytext=(5, 5), textcoords='offset points')

    fig.tight_layout()
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_merged()
    print(f'merged shape: {df.shape}')
    print(f'  Clumpy={int((df["condition"]=="Clumpy").sum())}, '
          f'Diffuse={int((df["condition"]=="Diffuse").sum())}')

    spatial_feats = [
        c for c in df.columns
        if c not in ['participant_id', 'condition'] + WIKI_FEATURES
    ]
    print(f'  spatial features: {len(spatial_feats)}')
    print(f'  wiki features: {len(WIKI_FEATURES)}')
    print(f'  -> {len(spatial_feats) * len(WIKI_FEATURES)} correlations')

    res = run_correlations(df, spatial_feats, WIKI_FEATURES)
    res.to_csv(CSV_OUT, index=False)
    print(f'wrote {CSV_OUT.name}')

    n_sig = int(res['fdr_significant'].sum())
    print(f'\n  FDR-significant pairs (alpha={FDR_ALPHA}): {n_sig}/{len(res)}')

    print('\n  Top 10 by |r|:')
    top = res.reindex(res['r_pearson'].abs().sort_values(ascending=False).index).head(10)
    for _, r in top.iterrows():
        marker = ' *' if r['fdr_significant'] else '  '
        print(f'   {marker} {r["spatial"]:32s} ~ {r["wiki"]:30s}  '
              f'r={r["r_pearson"]:+.2f}  rho={r["rho_spearman"]:+.2f}  '
              f'r|cond={r["r_partial"]:+.2f}  p_FDR={r["p_fdr"]:.4f}')

    n_flip = ((np.sign(res['r_pearson']) != np.sign(res['r_partial'])) &
              (res['r_pearson'].abs() > 0.15)).sum()
    print(f'\n  sign flips collapsed -> partial (|r|>0.15): {int(n_flip)}')

    with PdfPages(PDF_OUT) as pdf:
        page_heatmap(pdf, res, spatial_feats, WIKI_FEATURES)
        page_top_correlations(pdf, res)
        page_collapsed_vs_partial(pdf, res)
    print(f'wrote {PDF_OUT.name}')


if __name__ == '__main__':
    main()

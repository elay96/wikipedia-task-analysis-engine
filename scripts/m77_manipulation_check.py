#!/usr/bin/env python3
"""
M77: Manipulation Check on Spatial-Search features (Clumpy vs Diffuse)
======================================================================
Input:
  output/m76_spatial_features.csv     # N=132, 30 features (mean+sd) per pid

For each of the 30 features:
  - Welch t-test            (Clumpy mean - Diffuse mean)
  - Mann-Whitney U test     (rank-sum, distribution-free)
  - Cohen's d               (pooled SD, Hedges-style small-sample correction)
  - 95% CI on Cohen's d
  - n per group (NaN-aware)

FDR correction (Benjamini-Hochberg) applied to the 30 Welch p-values.

Theory predictions:
  Clumpy > Diffuse on  pct_time_exploit_mean, exploit_dur_mean_mean
  Clumpy < Diffuse on  n_transitions_mean

Outputs:
  output/m77_manipulation_check.csv
  output/m77_manipulation_check.pdf  (table + forest plot + raincloud of FDR passers)
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

FEATURES_IN = OUTPUT_DIR / 'm76_spatial_features.csv'
CSV_OUT = OUTPUT_DIR / 'm77_manipulation_check.csv'
PDF_OUT = OUTPUT_DIR / 'm77_manipulation_check.pdf'

CLUMPY_COLOR = '#C62828'
DIFFUSE_COLOR = '#1976D2'
GRID_COLOR = '#E0E0E0'
TEXT_COLOR = '#1a1a1a'
MUTED_COLOR = '#666666'
SIG_COLOR = '#2E7D32'
NS_COLOR = '#9E9E9E'
BG = '#FFFFFF'

FDR_ALPHA = 0.05

# theory predictions: feature -> direction Clumpy vs Diffuse ('+' = C > D, '-' = C < D)
THEORY = {
    'pct_time_exploit_mean': '+',
    'exploit_dur_mean_mean': '+',
    'exploit_dur_median_mean': '+',
    'n_transitions_mean': '-',
}


def cohens_d(x, y):
    """Pooled-SD Cohen's d with Hedges small-sample correction.
    Returns (d, se_d, n1, n2). NaN-safe."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, n1, n2
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan, np.nan, n1, n2
    d = (np.mean(x) - np.mean(y)) / pooled
    # Hedges g correction
    j = 1.0 - 3.0 / (4 * (n1 + n2) - 9)
    d = j * d
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2.0 * (n1 + n2)))
    return d, se, n1, n2


def fdr_bh(pvals, alpha=FDR_ALPHA):
    """Benjamini-Hochberg adjusted p-values. NaN-safe (NaN -> NaN, ignored in rank)."""
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
    # enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out_finite = np.empty(m)
    out_finite[order] = adj
    out[idx] = out_finite
    return out


def per_feature_test(df, feat):
    clumpy = df.loc[df['condition'] == 'Clumpy', feat].dropna().values
    diffuse = df.loc[df['condition'] == 'Diffuse', feat].dropna().values
    n_c, n_d = len(clumpy), len(diffuse)

    if n_c < 2 or n_d < 2:
        return {
            'feature': feat,
            'n_clumpy': n_c, 'n_diffuse': n_d,
            'mean_clumpy': float(np.mean(clumpy)) if n_c else np.nan,
            'sd_clumpy': float(np.std(clumpy, ddof=1)) if n_c >= 2 else np.nan,
            'mean_diffuse': float(np.mean(diffuse)) if n_d else np.nan,
            'sd_diffuse': float(np.std(diffuse, ddof=1)) if n_d >= 2 else np.nan,
            't': np.nan, 'df': np.nan, 'p_welch': np.nan,
            'U': np.nan, 'p_mwu': np.nan,
            'cohens_d': np.nan, 'd_se': np.nan,
            'd_ci_lo': np.nan, 'd_ci_hi': np.nan,
        }

    t_res = sp_stats.ttest_ind(clumpy, diffuse, equal_var=False, nan_policy='omit')
    # Welch-Satterthwaite df:
    s1, s2 = np.var(clumpy, ddof=1), np.var(diffuse, ddof=1)
    welch_df = (s1 / n_c + s2 / n_d) ** 2 / (
        (s1 / n_c) ** 2 / (n_c - 1) + (s2 / n_d) ** 2 / (n_d - 1)
    )
    u_res = sp_stats.mannwhitneyu(clumpy, diffuse, alternative='two-sided')
    d, d_se, _, _ = cohens_d(clumpy, diffuse)
    d_lo = d - 1.96 * d_se if np.isfinite(d) and np.isfinite(d_se) else np.nan
    d_hi = d + 1.96 * d_se if np.isfinite(d) and np.isfinite(d_se) else np.nan

    return {
        'feature': feat,
        'n_clumpy': n_c, 'n_diffuse': n_d,
        'mean_clumpy': float(np.mean(clumpy)),
        'sd_clumpy': float(np.std(clumpy, ddof=1)),
        'mean_diffuse': float(np.mean(diffuse)),
        'sd_diffuse': float(np.std(diffuse, ddof=1)),
        't': float(t_res.statistic),
        'df': float(welch_df),
        'p_welch': float(t_res.pvalue),
        'U': float(u_res.statistic),
        'p_mwu': float(u_res.pvalue),
        'cohens_d': float(d) if np.isfinite(d) else np.nan,
        'd_se': float(d_se) if np.isfinite(d_se) else np.nan,
        'd_ci_lo': float(d_lo) if np.isfinite(d_lo) else np.nan,
        'd_ci_hi': float(d_hi) if np.isfinite(d_hi) else np.nan,
    }


def run_tests(df):
    feats = [c for c in df.columns if c not in ('participant_id', 'condition')]
    rows = [per_feature_test(df, f) for f in feats]
    res = pd.DataFrame(rows)
    res['p_fdr'] = fdr_bh(res['p_welch'].values)
    res['fdr_significant'] = res['p_fdr'] < FDR_ALPHA
    res['theory_pred'] = res['feature'].map(THEORY).fillna('')
    res['matches_theory'] = res.apply(
        lambda r: (
            (r['theory_pred'] == '+' and r['cohens_d'] > 0) or
            (r['theory_pred'] == '-' and r['cohens_d'] < 0)
        ) if r['theory_pred'] else False,
        axis=1,
    )
    return res


# ---------- plotting ---------------------------------------------------------

def page_summary_table(pdf, res, n_clumpy_total, n_diffuse_total):
    res_sorted = res.reindex(res['cohens_d'].abs().sort_values(ascending=False).index)
    fig = plt.figure(figsize=(13, 12), facecolor=BG)
    fig.suptitle(
        f'M77 Manipulation Check  -  Clumpy (N={n_clumpy_total}) vs Diffuse (N={n_diffuse_total})\n'
        f'Sorted by |Cohen\'s d|.   FDR-BH alpha={FDR_ALPHA}.',
        fontsize=12, color=TEXT_COLOR, y=0.985,
    )

    cols = [
        ('feature', 'Feature'),
        ('n_clumpy', 'N_C'), ('n_diffuse', 'N_D'),
        ('mean_clumpy', 'Mean_C'), ('mean_diffuse', 'Mean_D'),
        ('cohens_d', 'd'),
        ('d_ci_lo', 'CI_lo'), ('d_ci_hi', 'CI_hi'),
        ('p_welch', 'p'), ('p_fdr', 'p_FDR'),
        ('p_mwu', 'p_MWU'),
        ('theory_pred', 'thr'), ('matches_theory', 'OK'),
    ]
    cell_text = []
    cell_colors = []
    for _, r in res_sorted.iterrows():
        row = []
        for k, _ in cols:
            v = r[k]
            if k == 'feature':
                row.append(str(v))
            elif k in ('n_clumpy', 'n_diffuse'):
                row.append(f'{int(v)}' if pd.notna(v) else '-')
            elif k in ('mean_clumpy', 'mean_diffuse'):
                row.append(f'{v:.2f}' if pd.notna(v) else '-')
            elif k in ('cohens_d', 'd_ci_lo', 'd_ci_hi'):
                row.append(f'{v:+.2f}' if pd.notna(v) else '-')
            elif k in ('p_welch', 'p_fdr', 'p_mwu'):
                if pd.isna(v):
                    row.append('-')
                elif v < 0.001:
                    row.append('<.001')
                else:
                    row.append(f'{v:.3f}')
            elif k == 'theory_pred':
                row.append(str(v) if v else '')
            elif k == 'matches_theory':
                row.append('Y' if v else ('-' if not r['theory_pred'] else 'N'))
            else:
                row.append(str(v))
        cell_text.append(row)

        c = [BG] * len(cols)
        # highlight FDR-significant rows in green tint
        if r['fdr_significant']:
            c = ['#E8F5E9'] * len(cols)
        # red tint for theory-prediction features that did NOT match
        if r['theory_pred'] and not r['matches_theory']:
            c = ['#FFEBEE'] * len(cols)
        cell_colors.append(c)

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
    tbl.scale(1.0, 1.15)
    # bold header
    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_text_props(weight='bold', color=TEXT_COLOR)
        cell.set_facecolor('#F0F4F8')
    # left-align feature column
    for i in range(1, len(cell_text) + 1):
        tbl[i, 0].set_text_props(ha='left')

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page_forest_plot(pdf, res):
    res_sorted = res.sort_values('cohens_d', ascending=True).reset_index(drop=True)
    n = len(res_sorted)
    fig = plt.figure(figsize=(11, max(8, 0.32 * n + 1.5)), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.axvline(0.0, color='#999999', lw=0.8)

    for i, r in res_sorted.iterrows():
        d = r['cohens_d']
        lo = r['d_ci_lo']
        hi = r['d_ci_hi']
        if pd.isna(d):
            continue
        col = SIG_COLOR if r['fdr_significant'] else NS_COLOR
        ax.plot([lo, hi], [i, i], color=col, lw=1.5, alpha=0.7)
        ax.plot(d, i, marker='o', ms=6, color=col)
        if r['theory_pred']:
            ax.annotate(
                f'  pred {r["theory_pred"]}', xy=(hi, i), color=MUTED_COLOR,
                fontsize=7, va='center',
            )

    ax.set_yticks(range(n))
    ax.set_yticklabels(res_sorted['feature'], fontsize=8)
    ax.set_xlabel("Cohen's d  (Clumpy - Diffuse, pooled SD)", color=TEXT_COLOR)
    ax.set_title(
        f'M77 Forest plot: 30 features  (green = FDR-significant at alpha={FDR_ALPHA})',
        color=TEXT_COLOR, fontsize=11,
    )
    ax.grid(True, axis='x', color=GRID_COLOR, alpha=0.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#CCCCCC')
    fig.tight_layout()
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def _strip_with_means(ax, df, feat):
    rng = np.random.default_rng(42)
    xc = df.loc[df['condition'] == 'Clumpy', feat].dropna().values
    xd = df.loc[df['condition'] == 'Diffuse', feat].dropna().values
    jitter_c = rng.normal(0, 0.06, size=len(xc))
    jitter_d = rng.normal(0, 0.06, size=len(xd))

    ax.scatter(np.full(len(xc), 0) + jitter_c, xc, s=18,
               color=CLUMPY_COLOR, alpha=0.55, edgecolors='none')
    ax.scatter(np.full(len(xd), 1) + jitter_d, xd, s=18,
               color=DIFFUSE_COLOR, alpha=0.55, edgecolors='none')

    # mean +/- 1 SE markers
    for x_pos, vals, col in ((0, xc, CLUMPY_COLOR), (1, xd, DIFFUSE_COLOR)):
        if len(vals) >= 2:
            m = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.plot([x_pos - 0.18, x_pos + 0.18], [m, m], color=col, lw=2.0)
            ax.plot([x_pos, x_pos], [m - se, m + se], color=col, lw=1.4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Clumpy', 'Diffuse'], fontsize=9, color=TEXT_COLOR)
    ax.set_xlim(-0.5, 1.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', labelsize=8, colors=TEXT_COLOR)
    ax.grid(True, axis='y', color=GRID_COLOR, alpha=0.4)


def page_raincloud(pdf, df, res, max_show=9):
    sig = res[res['fdr_significant']].sort_values('p_fdr')
    if len(sig) == 0:
        feats_to_show = res.reindex(
            res['p_welch'].fillna(1.0).sort_values().index
        ).head(6)['feature'].tolist()
        title_extra = '(no features survived FDR; showing 6 lowest-p)'
    else:
        feats_to_show = sig['feature'].head(max_show).tolist()
        title_extra = f'({len(sig)} features survived FDR; showing top {len(feats_to_show)})'

    n = len(feats_to_show)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(13, 3.0 * rows + 1.0), facecolor=BG)
    fig.suptitle(
        f'M77 Per-feature distribution by condition  {title_extra}',
        fontsize=12, color=TEXT_COLOR, y=0.99,
    )

    for i, feat in enumerate(feats_to_show):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor(BG)
        _strip_with_means(ax, df, feat)
        r = res[res['feature'] == feat].iloc[0]
        d = r['cohens_d']
        p_fdr = r['p_fdr']
        ax.set_title(
            f'{feat}\nd={d:+.2f},  p_FDR={"<.001" if p_fdr < 0.001 else f"{p_fdr:.3f}"}',
            fontsize=9, color=TEXT_COLOR,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(FEATURES_IN)
    print(f'loaded {FEATURES_IN.name}: {df.shape}')

    n_c = int((df['condition'] == 'Clumpy').sum())
    n_d = int((df['condition'] == 'Diffuse').sum())
    print(f'  Clumpy={n_c}, Diffuse={n_d}')

    res = run_tests(df)
    res.to_csv(CSV_OUT, index=False)
    print(f'wrote {CSV_OUT.name}  ({len(res)} features)')

    n_sig = int(res['fdr_significant'].sum())
    print(f'  features passing FDR (alpha={FDR_ALPHA}): {n_sig}/{len(res)}')

    print('\n  Top 10 by |d|:')
    top = res.reindex(res['cohens_d'].abs().sort_values(ascending=False).index).head(10)
    for _, r in top.iterrows():
        marker = ' *' if r['fdr_significant'] else '  '
        print(f'   {marker} {r["feature"]:35s}  d={r["cohens_d"]:+.2f}  '
              f'p={r["p_welch"]:.4f}  p_FDR={r["p_fdr"]:.4f}')

    print('\n  Theory-predicted features:')
    for f in THEORY:
        if f in res['feature'].values:
            r = res[res['feature'] == f].iloc[0]
            ok = 'OK' if r['matches_theory'] else 'MISS'
            print(f'   [{ok}] {f}: pred {r["theory_pred"]}, '
                  f'd={r["cohens_d"]:+.2f}, p_FDR={r["p_fdr"]:.4f}')

    with PdfPages(PDF_OUT) as pdf:
        page_summary_table(pdf, res, n_c, n_d)
        page_forest_plot(pdf, res)
        page_raincloud(pdf, df, res)
    print(f'wrote {PDF_OUT.name}')


if __name__ == '__main__':
    main()

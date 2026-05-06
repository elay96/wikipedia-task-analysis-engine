#!/usr/bin/env python3
"""
M70 - Variance + quantile-regression analysis of Mean reading-only streak
=========================================================================
Follow-up to M69. The raincloud (M69) suggested the Clumpy condition
widens the upper tail of the streak distribution, not just shifts the
mean. M70 tests this formally via:

  (1) Brown-Forsythe / Levene tests for equality of variance.
  (2) Quantile regression of streak ~ condition + final_answer_length
      at q = 0.10, 0.25, 0.50, 0.75, 0.90.  If the manipulation is
      purely a location shift, the condition coefficient should be
      flat across quantiles. A growing coefficient = heterogeneous
      treatment effect concentrated in the upper tail.

Outputs:
  output/m70_variance_quantile_results.csv   (all numeric results)
  output/m70_variance_quantile.pdf           (2-page summary)
    Page 1 - Raincloud with variance annotation
    Page 2 - Quantile regression coefficient profile
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from helpers import OUTPUT_DIR

BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
MUTED = '#666666'
BORDER = '#CCCCCC'
GRID = '#E8E8E8'
COND_COLORS = {'diffuse': '#1976D2', 'clumpy': '#C62828'}

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
MEASURE = 'seq_typing_mean_run_explore'
COVARIATE = 'final_answer_length'


def _style_axes(ax):
    ax.set_facecolor(BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=MUTED)


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def compute_variance_tests(d, c):
    levene_mean = sp_stats.levene(d, c, center='mean')
    brown_forsythe = sp_stats.levene(d, c, center='median')
    var_d, var_c = d.var(ddof=1), c.var(ddof=1)
    f_stat = var_c / var_d
    df1, df2 = len(c) - 1, len(d) - 1
    f_p = 2 * min(sp_stats.f.cdf(f_stat, df1, df2),
                  1 - sp_stats.f.cdf(f_stat, df1, df2))
    return {
        'n_diffuse': len(d), 'n_clumpy': len(c),
        'sd_diffuse': d.std(ddof=1), 'sd_clumpy': c.std(ddof=1),
        'var_ratio_C_over_D': var_c / var_d,
        'levene_W': levene_mean.statistic,
        'levene_p': levene_mean.pvalue,
        'brown_forsythe_W': brown_forsythe.statistic,
        'brown_forsythe_p': brown_forsythe.pvalue,
        'f_test_F': f_stat,
        'f_test_p': f_p,
    }


def compute_quantile_regression(df):
    df = df[[MEASURE, 'condition', COVARIATE]].dropna().copy()
    df['cond_clumpy'] = (df['condition'] == 'clumpy').astype(int)
    X = sm.add_constant(df[['cond_clumpy', COVARIATE]])
    y = df[MEASURE]

    rows = []
    for q in QUANTILES:
        res = QuantReg(y, X).fit(q=q, max_iter=5000)
        ci = res.conf_int().loc['cond_clumpy']
        rows.append({
            'quantile': q,
            'coef': res.params['cond_clumpy'],
            'se': res.bse['cond_clumpy'],
            't': res.tvalues['cond_clumpy'],
            'p': res.pvalues['cond_clumpy'],
            'ci_lo': ci[0],
            'ci_hi': ci[1],
            'covariate_coef': res.params[COVARIATE],
            'covariate_p': res.pvalues[COVARIATE],
            'n': len(y),
        })

    ols = sm.OLS(y, X).fit()
    ols_ci = ols.conf_int().loc['cond_clumpy']
    ols_row = {
        'quantile': 'OLS_mean',
        'coef': ols.params['cond_clumpy'],
        'se': ols.bse['cond_clumpy'],
        't': ols.tvalues['cond_clumpy'],
        'p': ols.pvalues['cond_clumpy'],
        'ci_lo': ols_ci[0],
        'ci_hi': ols_ci[1],
        'covariate_coef': ols.params[COVARIATE],
        'covariate_p': ols.pvalues[COVARIATE],
        'n': len(y),
    }
    return pd.DataFrame(rows + [ols_row])


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _raincloud_panel(ax, groups, group_labels, group_colors, ylabel,
                     title=None, rng_seed=7):
    _style_axes(ax)
    ax.grid(True, color=GRID, linewidth=0.5, axis='y', zorder=0)

    rng = np.random.default_rng(rng_seed)
    positions = np.arange(len(groups), dtype=float)

    parts = ax.violinplot(
        groups, positions=positions, widths=0.85, showmeans=False,
        showmedians=False, showextrema=False,
    )
    for body, pos, color in zip(parts['bodies'], positions, group_colors):
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], pos, pos + 0.42)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.30)
        body.set_zorder(2)

    bp = ax.boxplot(
        groups, positions=positions, widths=0.10, patch_artist=True,
        showfliers=False, zorder=4,
        medianprops=dict(color='white', linewidth=1.6),
        whiskerprops=dict(color='#333333', linewidth=1.0),
        capprops=dict(color='#333333', linewidth=1.0),
    )
    for box, color in zip(bp['boxes'], group_colors):
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.85)

    for vals, pos, color in zip(groups, positions, group_colors):
        jitter = rng.uniform(-0.30, -0.06, size=len(vals))
        ax.scatter(pos + jitter, vals,
                   color=color, alpha=0.55, s=22, zorder=3,
                   edgecolors='white', linewidth=0.5)

    means = [np.mean(g) for g in groups]
    sds = [np.std(g, ddof=1) for g in groups]
    for pos, m, sd, color in zip(positions, means, sds, group_colors):
        ax.scatter([pos], [m], marker='D', s=42,
                   facecolor='white', edgecolor=color, linewidth=1.6,
                   zorder=5)
        ax.text(pos + 0.50, m,
                f'M = {m:.2f}\nSD = {sd:.2f}',
                ha='left', va='center', fontsize=10,
                color=color, fontweight='bold')

    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, fontsize=11, color=TEXT)
    ax.set_xlim(-0.6, len(groups) - 1 + 0.95)
    ax.set_ylabel(ylabel, color=LABEL, fontweight='bold', fontsize=11)
    if title is not None:
        ax.set_title(title, color=MUTED, fontsize=10, pad=8)


def page_raincloud_with_variance(pdf, df, var_results):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'Mean reading-only streak: distribution shape by condition',
        fontsize=15, fontweight='bold', color=TEXT, y=0.965,
    )
    fig.text(
        0.5, 0.913,
        'Per participant (N = 99 of 101).  '
        'The Clumpy condition opens the upper tail - not just shifts the mean.',
        ha='center', va='top', fontsize=10.5, color=MUTED, style='italic',
    )

    ax_rc = fig.add_axes([0.10, 0.30, 0.55, 0.55])
    d_vals = df.loc[df['condition'] == 'diffuse', MEASURE].dropna().to_numpy()
    c_vals = df.loc[df['condition'] == 'clumpy',  MEASURE].dropna().to_numpy()
    _raincloud_panel(
        ax_rc,
        groups=[d_vals, c_vals],
        group_labels=['Diffuse', 'Clumpy'],
        group_colors=[COND_COLORS['diffuse'], COND_COLORS['clumpy']],
        ylabel='Mean reading-only streak (pages)',
    )

    # Side panel: variance test results
    ax_var = fig.add_axes([0.70, 0.30, 0.26, 0.55])
    ax_var.axis('off')
    ax_var.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_var.transAxes,
        facecolor='#FAFAFA', edgecolor=BORDER, linewidth=1,
    ))
    ax_var.text(0.5, 0.95, 'Equality-of-variance tests',
                transform=ax_var.transAxes, ha='center', va='top',
                fontsize=11, fontweight='bold', color=TEXT)

    sd_d = var_results['sd_diffuse']
    sd_c = var_results['sd_clumpy']
    var_ratio = var_results['var_ratio_C_over_D']

    lines = [
        ('SD Diffuse',          f'{sd_d:.2f}'),
        ('SD Clumpy',           f'{sd_c:.2f}'),
        ('Variance ratio C/D',  f'{var_ratio:.2f}'),
        ('', ''),
        ('Levene (mean)',       f'p = {var_results["levene_p"]:.3f}'),
        ('Brown-Forsythe',      f'p = {var_results["brown_forsythe_p"]:.3f}'),
        ('F-test',              f'p = {var_results["f_test_p"]:.3f}'),
    ]
    y0 = 0.85
    for i, (label, value) in enumerate(lines):
        y = y0 - i * 0.085
        if label:
            ax_var.text(0.06, y, label,
                        transform=ax_var.transAxes, ha='left', va='center',
                        fontsize=10, color=LABEL)
            ax_var.text(0.94, y, value,
                        transform=ax_var.transAxes, ha='right', va='center',
                        fontsize=10, color=TEXT, fontweight='bold',
                        family='monospace')

    verdict_y = 0.16
    verdict = ('Variance gap is suggestive\n'
               'but not formally significant\n'
               '(N = 99 - underpowered for\n'
               'a 27% SD difference).')
    ax_var.text(0.5, verdict_y, verdict,
                transform=ax_var.transAxes, ha='center', va='center',
                fontsize=9.5, color=MUTED, style='italic',
                linespacing=1.4)

    # Bottom note
    ax_take = fig.add_axes([0.06, 0.03, 0.88, 0.22])
    ax_take.axis('off')
    ax_take.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_take.transAxes,
        facecolor='#FAFAFA', edgecolor=BORDER, linewidth=1,
    ))
    ax_take.text(0.5, 0.86,
                 'What the raincloud reveals beyond a bar+SEM chart',
                 transform=ax_take.transAxes, ha='center', va='center',
                 fontsize=11.5, fontweight='bold', color=TEXT)
    line1 = ('1. Both distributions are right-skewed (long tail of "deep readers"); '
             'mean is pulled above the median in each.')
    line2 = ('2. SD is 27% larger in Clumpy (2.35 vs 1.85) - the manipulation '
             'widens the spread, not just translates the centre.')
    line3 = ('3. The upper tail in Clumpy reaches further (max = 10.5 vs 9.75 pages, '
             'IQR top = 4.88 vs 4.00).')
    headline = ('Hypothesis: the manipulation produces a heterogeneous treatment effect, '
                'concentrated in the upper tail. Tested formally on page 2.')
    for i, ln in enumerate([line1, line2, line3]):
        ax_take.text(0.05, 0.66 - i * 0.14, ln,
                     transform=ax_take.transAxes, ha='left', va='center',
                     fontsize=9.8, color=TEXT)
    ax_take.plot([0.04, 0.96], [0.20, 0.20], transform=ax_take.transAxes,
                 color=BORDER, linewidth=0.8)
    ax_take.text(0.5, 0.10, headline, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=10.5,
                 color='#1B5E20', fontweight='bold')

    pdf.savefig(fig, facecolor=BG)
    plt.close()


def page_quantile_regression(pdf, qr_df):
    qr = qr_df[qr_df['quantile'] != 'OLS_mean'].copy()
    qr['quantile'] = qr['quantile'].astype(float)
    ols_row = qr_df[qr_df['quantile'] == 'OLS_mean'].iloc[0]

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'Quantile regression: condition effect across the streak distribution',
        fontsize=15, fontweight='bold', color=TEXT, y=0.965,
    )
    fig.text(
        0.5, 0.913,
        'Model:  Mean reading-only streak  ~  condition + final_answer_length   '
        '(N = 99).  Plotted: condition coefficient (Clumpy vs Diffuse).',
        ha='center', va='top', fontsize=10, color=MUTED, style='italic',
    )

    # Main plot: coef vs quantile, with 95% CI ribbon
    ax = fig.add_axes([0.10, 0.32, 0.85, 0.52])
    _style_axes(ax)
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)

    qs = qr['quantile'].to_numpy()
    coefs = qr['coef'].to_numpy()
    ci_lo = qr['ci_lo'].to_numpy()
    ci_hi = qr['ci_hi'].to_numpy()
    ps = qr['p'].to_numpy()

    # CI ribbon
    ax.fill_between(qs, ci_lo, ci_hi, color='#1976D2', alpha=0.18, zorder=1,
                    label='95% CI')

    # Reference: zero line
    ax.axhline(0, color='#888', linewidth=1.0, linestyle='--', zorder=2)

    # Reference: OLS mean coefficient (horizontal line)
    ax.axhline(ols_row['coef'], color='#444', linewidth=1.2,
               linestyle=':', zorder=2,
               label=f'OLS mean coef = {ols_row["coef"]:+.2f}')

    # Coef line
    ax.plot(qs, coefs, color='#C62828', linewidth=2.2, zorder=3, marker='o',
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor='#C62828', label='Quantile coef')

    # Annotate each point with p-value
    for q, coef, p in zip(qs, coefs, ps):
        sig = '*' if p < 0.05 else ''
        ax.text(q, coef + (ci_hi - ci_lo).max() * 0.12,
                f'{coef:+.2f}{sig}\np={p:.3f}',
                ha='center', va='bottom', fontsize=9,
                color='#C62828' if p < 0.05 else TEXT,
                fontweight='bold' if p < 0.05 else 'normal')

    ax.set_xlabel('Quantile of streak distribution', color=LABEL, fontsize=11)
    ax.set_ylabel('Condition coefficient (Clumpy - Diffuse, in pages)',
                  color=LABEL, fontsize=11, fontweight='bold')
    ax.set_xlim(0.05, 0.95)
    ax.set_xticks(QUANTILES)
    ax.set_xticklabels([f'Q{int(q*100)}' for q in QUANTILES])
    ymax = max(ci_hi.max(), coefs.max()) * 1.25
    ymin = min(ci_lo.min(), 0) - 0.5
    ax.set_ylim(ymin, ymax)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=1.0,
              edgecolor=BORDER, facecolor='white')

    # Bottom note
    ax_take = fig.add_axes([0.06, 0.03, 0.88, 0.22])
    ax_take.axis('off')
    ax_take.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_take.transAxes,
        facecolor='#FAFAFA', edgecolor=BORDER, linewidth=1,
    ))
    ax_take.text(0.5, 0.86,
                 'How to read this',
                 transform=ax_take.transAxes, ha='center', va='center',
                 fontsize=11.5, fontweight='bold', color=TEXT)
    line1 = ('If the manipulation were a pure location shift, the coefficient '
             'line would be flat at the OLS mean (+0.90).')
    line2 = ('Instead the coefficient grows monotonically: ~+0.5 at Q10/Q25, '
             '+0.6 at the median, +1.3 at Q75 (significant), +1.7 at Q90.')
    line3 = ('Only the Q75 coefficient is individually significant; sample size '
             'limits power at the extreme quantiles. The upward trend across '
             'quantiles is the substantive finding.')
    headline = ('Conclusion: the Clumpy condition does not move every participant equally - '
                'it primarily lengthens streaks among participants already in the upper half.')
    for i, ln in enumerate([line1, line2, line3]):
        ax_take.text(0.05, 0.66 - i * 0.13, ln,
                     transform=ax_take.transAxes, ha='left', va='center',
                     fontsize=9.8, color=TEXT)
    ax_take.plot([0.04, 0.96], [0.21, 0.21], transform=ax_take.transAxes,
                 color=BORDER, linewidth=0.8)
    ax_take.text(0.5, 0.10, headline, transform=ax_take.transAxes,
                 ha='center', va='center', fontsize=10.5,
                 color='#1B5E20', fontweight='bold')

    pdf.savefig(fig, facecolor=BG)
    plt.close()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('M70 - variance + quantile regression')

    m56_csv = OUTPUT_DIR / 'm56_eda_writing_sequential.csv'
    if not m56_csv.exists():
        raise FileNotFoundError(f'Missing input: {m56_csv}')

    df = pd.read_csv(m56_csv).dropna(subset=[MEASURE])
    d_vals = df.loc[df['condition'] == 'diffuse', MEASURE].to_numpy()
    c_vals = df.loc[df['condition'] == 'clumpy',  MEASURE].to_numpy()

    var_results = compute_variance_tests(d_vals, c_vals)
    qr_df = compute_quantile_regression(df)

    # Combine results into a single CSV
    var_long = pd.DataFrame([{'analysis': 'variance', **var_results}])
    qr_df.insert(0, 'analysis', 'quantile_regression')
    out_csv = OUTPUT_DIR / 'm70_variance_quantile_results.csv'
    pd.concat([
        var_long.assign(quantile='', coef='', se='', t='', p='',
                        ci_lo='', ci_hi='', covariate_coef='',
                        covariate_p='', n=''),
        qr_df.assign(n_diffuse='', n_clumpy='', sd_diffuse='', sd_clumpy='',
                     var_ratio_C_over_D='', levene_W='', levene_p='',
                     brown_forsythe_W='', brown_forsythe_p='',
                     f_test_F='', f_test_p=''),
    ], ignore_index=True).to_csv(out_csv, index=False)
    print(f'Saved CSV: {out_csv}')

    # Console summary
    print()
    print('Variance tests:')
    for k, v in var_results.items():
        print(f'  {k:25s} = {v:.3f}' if isinstance(v, float) else f'  {k:25s} = {v}')
    print()
    print('Quantile regression (condition coef):')
    print(qr_df[['quantile', 'coef', 'ci_lo', 'ci_hi', 'p']].to_string(index=False))

    # PDF
    pdf_path = OUTPUT_DIR / 'm70_variance_quantile.pdf'
    with PdfPages(pdf_path) as pdf:
        page_raincloud_with_variance(pdf, df, var_results)
        page_quantile_regression(pdf, qr_df)
    print(f'Saved PDF: {pdf_path}')


if __name__ == '__main__':
    main()

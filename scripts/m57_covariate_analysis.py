#!/usr/bin/env python3
"""
M57: Covariate analysis - pattern vs effort
===========================================
Question: do the M56 sequential pattern findings (max_run, entropy) survive when
controlling for final_answer_length (a proxy for effort/output volume)?

If pattern effect survives -> real behavioral difference between conditions.
If pattern effect disappears -> the only real difference is volume of writing.

Method: linear regression  measure ~ condition + final_answer_length
Compare the condition coefficient with and without the covariate.

Inputs:  output/m56_eda_writing_sequential.csv (per-participant averaged measures)
Outputs: output/m57_covariate_analysis.csv  - summary table
         output/m57_covariate_analysis.pdf  - report
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats

from helpers import OUTPUT_DIR

# Light palette
BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'
COND_COLORS = {'diffuse': '#1976D2', 'clumpy': '#C62828'}

COVARIATE = 'final_answer_length'

# Measures to test (the M56 winners + a few borderline)
TARGET_MEASURES = [
    'seq_typing_max_run',
    'seq_typing_entropy',
    'seq_typing_mean_run_explore',
    'seq_typing_mean_run_exploit',
    'pasted_chars',
    'n_paste_events',
    'seq_time_mean_run_explore',
    'seq_topic_mean_run_exploit',
    'first_writing_time_s',
    'prop_pages_with_writing',
]


# ----------------------------------------------------------------------
# OLS regression with t-tests (no statsmodels dependency)
# ----------------------------------------------------------------------

def ols_fit(X, y):
    """Return beta, se(beta), t, p, residuals for OLS fit y = X beta + e.

    X must include intercept column. Returns df-residuals = n - p.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    df_resid = n - p
    sigma2 = (resid @ resid) / df_resid
    var_beta = sigma2 * np.diag(XtX_inv)
    se_beta = np.sqrt(var_beta)
    t_stat = beta / se_beta
    p_val = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), df_resid))
    return {
        'beta': beta, 'se': se_beta, 't': t_stat, 'p': p_val,
        'df_resid': df_resid, 'resid': resid, 'sigma2': sigma2,
    }


def cohens_d(d_vals, c_vals):
    pooled = np.sqrt(((len(d_vals) - 1) * d_vals.std(ddof=1) ** 2
                      + (len(c_vals) - 1) * c_vals.std(ddof=1) ** 2)
                     / (len(d_vals) + len(c_vals) - 2))
    if pooled == 0:
        return 0.0
    return (d_vals.mean() - c_vals.mean()) / pooled


def adjusted_means(beta, X_cols, target_cov_value):
    """Compute model-predicted means for each condition at a fixed covariate value.

    X_cols ordering: [intercept, condition_diffuse, covariate]
    """
    # Diffuse = 1 (condition_diffuse=1):
    mean_d = beta[0] + beta[1] * 1 + beta[2] * target_cov_value
    # Clumpy = baseline (condition_diffuse=0):
    mean_c = beta[0] + beta[1] * 0 + beta[2] * target_cov_value
    return mean_d, mean_c


def analyze_measure(df, measure):
    """Run unadjusted t-test + adjusted (ANCOVA-style) regression."""
    sub = df[['condition', measure, COVARIATE]].dropna()
    d_mask = sub['condition'] == 'diffuse'
    c_mask = sub['condition'] == 'clumpy'
    d_vals = sub.loc[d_mask, measure].to_numpy()
    c_vals = sub.loc[c_mask, measure].to_numpy()

    # Unadjusted (Welch t)
    t_raw, p_raw = sp_stats.ttest_ind(d_vals, c_vals, equal_var=False)
    d_raw = cohens_d(d_vals, c_vals)
    raw_diff = d_vals.mean() - c_vals.mean()

    # Adjusted: y = b0 + b1*condition_diffuse + b2*covariate
    n = len(sub)
    X = np.column_stack([
        np.ones(n),
        (sub['condition'].values == 'diffuse').astype(float),
        sub[COVARIATE].values.astype(float),
    ])
    y = sub[measure].values.astype(float)
    fit = ols_fit(X, y)

    # The condition coefficient is the adjusted between-group difference
    adj_diff = fit['beta'][1]
    adj_t = fit['t'][1]
    adj_p = fit['p'][1]
    cov_t = fit['t'][2]
    cov_p = fit['p'][2]

    # Adjusted Cohen's d: use sqrt(MSE) as the SD denominator
    sigma = np.sqrt(fit['sigma2'])
    adj_d = adj_diff / sigma if sigma > 0 else 0.0

    # Predicted means at grand mean of covariate
    cov_grand_mean = sub[COVARIATE].mean()
    pred_d, pred_c = adjusted_means(fit['beta'], X, cov_grand_mean)

    # Survival decision
    if abs(d_raw) > 0:
        d_attenuation_pct = 100 * (1 - abs(adj_d) / abs(d_raw))
    else:
        d_attenuation_pct = np.nan

    if adj_p < .05 and abs(adj_d) >= 0.3:
        survival = 'SURVIVED'
    elif abs(adj_d) >= 0.5 * abs(d_raw) and abs(adj_d) >= 0.2:
        survival = 'PARTIAL'
    else:
        survival = 'KILLED'

    return {
        'measure': measure,
        'n': n, 'n_d': int(d_mask.sum()), 'n_c': int(c_mask.sum()),
        'raw_d_mean': d_vals.mean(), 'raw_c_mean': c_vals.mean(),
        'raw_diff': raw_diff, 'raw_t': t_raw, 'raw_p': p_raw, 'raw_cohen_d': d_raw,
        'adj_diff': adj_diff, 'adj_t': adj_t, 'adj_p': adj_p, 'adj_cohen_d': adj_d,
        'covariate_t': cov_t, 'covariate_p': cov_p,
        'adj_pred_d': pred_d, 'adj_pred_c': pred_c,
        'attenuation_pct': d_attenuation_pct, 'survival': survival,
    }


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_partial_residuals(ax, df, measure, fit_info):
    """Scatter of measure vs covariate, separated by condition, with fit lines."""
    ax.set_facecolor(BG)
    sub = df[['condition', measure, COVARIATE]].dropna()
    for cond in ['diffuse', 'clumpy']:
        s = sub[sub['condition'] == cond]
        ax.scatter(s[COVARIATE], s[measure], color=COND_COLORS[cond],
                   s=28, alpha=0.65, edgecolors='#333', linewidth=0.3,
                   label=f'{cond} (n={len(s)})', zorder=3)

    # Fit lines using the model coefficients (parallel slopes ANCOVA assumption)
    x_range = np.linspace(sub[COVARIATE].min(), sub[COVARIATE].max(), 50)
    # y = b0 + b1*cond + b2*cov. For diffuse cond=1, clumpy cond=0
    b = fit_info['beta_full']
    y_d = b[0] + b[1] * 1 + b[2] * x_range
    y_c = b[0] + b[1] * 0 + b[2] * x_range
    ax.plot(x_range, y_d, color=COND_COLORS['diffuse'], linewidth=2, zorder=2)
    ax.plot(x_range, y_c, color=COND_COLORS['clumpy'], linewidth=2, zorder=2)

    ax.set_xlabel(COVARIATE, color=LABEL, fontweight='bold', fontsize=9)
    ax.set_ylabel(measure, color=LABEL, fontweight='bold', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.4, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_effect_comparison(ax, results_df):
    """Forest-style plot: raw d vs adjusted d for each measure."""
    ax.set_facecolor(BG)
    measures = results_df['measure'].tolist()
    y_pos = np.arange(len(measures))

    raw_d = results_df['raw_cohen_d'].values
    adj_d = results_df['adj_cohen_d'].values

    # Colored bars by survival
    color_map = {'SURVIVED': '#2E7D32', 'PARTIAL': '#F9A825', 'KILLED': '#C62828'}
    surv_colors = [color_map.get(s, MUTED) for s in results_df['survival']]

    bar_h = 0.36
    ax.barh(y_pos - bar_h / 2, raw_d, bar_h, color='#90CAF9',
            edgecolor='#1565C0', label='Raw d (no covariate)', alpha=0.85)
    ax.barh(y_pos + bar_h / 2, adj_d, bar_h, color=surv_colors,
            edgecolor='#333', label='Adjusted d (controlling for answer length)',
            alpha=0.9)

    for i, (r, a) in enumerate(zip(raw_d, adj_d)):
        ax.text(r + (0.02 if r >= 0 else -0.02), i - bar_h / 2,
                f'{r:+.2f}', va='center', ha='left' if r >= 0 else 'right',
                fontsize=8, color=MUTED)
        ax.text(a + (0.02 if a >= 0 else -0.02), i + bar_h / 2,
                f'{a:+.2f}', va='center', ha='left' if a >= 0 else 'right',
                fontsize=8, color=TEXT, fontweight='bold')

    ax.axvline(0, color=BORDER, linewidth=1)
    for thresh in [0.2, 0.5, -0.2, -0.5]:
        ax.axvline(thresh, color=GRID, linewidth=0.5, linestyle=':')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(measures, fontsize=9)
    ax.set_xlabel('Cohen\'s d (Diffuse - Clumpy)', color=LABEL, fontweight='bold')
    ax.set_title('Effect size: raw vs covariate-adjusted',
                 color=TEXT, fontweight='bold', fontsize=13, pad=10)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.4, axis='x', zorder=0)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()


def make_summary_table_page(results_df):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M57: Covariate Analysis - Effect Survival Table',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')

    lines = []
    lines.append(f'Covariate: {COVARIATE}')
    lines.append('')
    header = (f'{"Measure":<32} | {"Raw d":>7} {"Raw p":>7} | '
              f'{"Adj d":>7} {"Adj p":>7} | {"Atten":>6} | {"Survival"}')
    lines.append(header)
    lines.append('-' * len(header))
    for _, r in results_df.iterrows():
        atten = f'{r["attenuation_pct"]:>5.0f}%' if pd.notna(r['attenuation_pct']) else '   nan'
        lines.append(
            f'{r["measure"]:<32} | {r["raw_cohen_d"]:>+7.2f} {r["raw_p"]:>7.3f} | '
            f'{r["adj_cohen_d"]:>+7.2f} {r["adj_p"]:>7.3f} | {atten:>6} | {r["survival"]}'
        )
    lines.append('')
    lines.append('Survival rules:')
    lines.append('  SURVIVED  -> adjusted p < .05 AND |adj d| >= 0.3')
    lines.append('  PARTIAL   -> |adj d| >= 50% of |raw d| AND |adj d| >= 0.2')
    lines.append('  KILLED    -> effect mostly attributable to covariate')
    lines.append('')
    lines.append('Adj d denominator = sqrt(MSE) from full model (different scale')
    lines.append('than raw pooled SD; absolute comparison is approximate).')

    text = '\n'.join(lines)
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=9,
            family='monospace', va='top', color=TEXT, linespacing=1.4)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M57] Covariate analysis - pattern vs effort')
    print('=' * 60)

    csv_path = OUTPUT_DIR / 'm56_eda_writing_sequential.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Run M56 first to produce {csv_path}')
    df = pd.read_csv(csv_path)
    print(f'Loaded N={len(df)} participants from {csv_path}')

    print('\n--- Running covariate analysis ---')
    results = []
    fit_info_per_measure = {}
    for measure in TARGET_MEASURES:
        if measure not in df.columns:
            print(f'  SKIP (missing column): {measure}')
            continue
        result = analyze_measure(df, measure)
        results.append(result)
        # Re-fit and store full beta for plotting
        sub = df[['condition', measure, COVARIATE]].dropna()
        n = len(sub)
        X = np.column_stack([
            np.ones(n),
            (sub['condition'].values == 'diffuse').astype(float),
            sub[COVARIATE].values.astype(float),
        ])
        y = sub[measure].values.astype(float)
        fit = ols_fit(X, y)
        fit_info_per_measure[measure] = {'beta_full': fit['beta']}

        print(f'  {measure:<32}  raw d={result["raw_cohen_d"]:+.2f} '
              f'(p={result["raw_p"]:.3f})  ->  '
              f'adj d={result["adj_cohen_d"]:+.2f} (p={result["adj_p"]:.3f})  '
              f'[{result["survival"]}]')

    results_df = pd.DataFrame(results)

    # Save CSV
    csv_out = OUTPUT_DIR / 'm57_covariate_analysis.csv'
    results_df.to_csv(csv_out, index=False)
    print(f'\nSaved CSV: {csv_out}')

    # Build PDF
    pdf_path = OUTPUT_DIR / 'm57_covariate_analysis.pdf'
    with PdfPages(pdf_path) as pdf:
        # Page 1: forest plot of effect comparison
        fig, ax = plt.subplots(figsize=(11, 6.5))
        fig.patch.set_facecolor(BG)
        plot_effect_comparison(ax, results_df)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # Page 2: summary table
        fig = make_summary_table_page(results_df)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # Pages 3+: partial-residual scatter for the top-4 raw effects
        top4 = results_df.reindex(results_df['raw_cohen_d'].abs().sort_values(ascending=False).index).head(4)
        for i in range(0, len(top4), 2):
            chunk = top4.iloc[i:i + 2]
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            fig.patch.set_facecolor(BG)
            for j, (_, r) in enumerate(chunk.iterrows()):
                m = r['measure']
                plot_partial_residuals(axes[j], df, m, fit_info_per_measure[m])
                axes[j].set_title(
                    f'{m}\nraw d={r["raw_cohen_d"]:+.2f} (p={r["raw_p"]:.3f})  ->  '
                    f'adj d={r["adj_cohen_d"]:+.2f} (p={r["adj_p"]:.3f})  '
                    f'[{r["survival"]}]',
                    color=TEXT, fontweight='bold', fontsize=10,
                )
            if len(chunk) == 1:
                axes[1].axis('off')
            plt.tight_layout()
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f'Saved PDF: {pdf_path}')

    # Print verdict
    print('\n' + '=' * 60)
    print('VERDICT')
    print('=' * 60)
    survived = results_df[results_df['survival'] == 'SURVIVED']
    partial = results_df[results_df['survival'] == 'PARTIAL']
    killed = results_df[results_df['survival'] == 'KILLED']
    print(f'  SURVIVED ({len(survived)}): {", ".join(survived["measure"].tolist())}')
    print(f'  PARTIAL  ({len(partial)}): {", ".join(partial["measure"].tolist())}')
    print(f'  KILLED   ({len(killed)}): {", ".join(killed["measure"].tolist())}')
    print('\nDone.')


if __name__ == '__main__':
    main()

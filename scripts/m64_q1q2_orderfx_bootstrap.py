#!/usr/bin/env python3
"""
M64: Q1 vs Q2 correlation, order effect test, and bootstrap CIs
===============================================================
Built in response to advisor feedback (2026-04-30):

Kineret: entropy isn't order-sensitive; mean reading streak (seq_typing_mean_run_explore)
is theoretically more aligned with Yuval's CFG exploration depth measure. The
pattern of effect strengthening in Q2 contradicts a priming explanation.

Yuval: maybe the zero AH-PS correlation in M59 is driven by a strong order
effect (Q1 always lower than Q2 or vice versa). When pivoting by domain we
randomly mix Q1 and Q2 across columns, destroying the pairing.
He also asked for bootstrap CIs to verify effect coherence.

Analyses:
  1. Pivot by trial_position (Q1 vs Q2) and compute Pearson r per measure.
     Compare to the AH-PS correlation from M59.
  2. Test whether mean(Q1) != mean(Q2) - paired t per measure (order effect).
  3. Residualize each subject's value on trial_position, then re-compute
     correlation: this is what the within-subject reliability "should" be
     after stripping the order shift.
  4. Bootstrap (5000 resamples) CIs for Cohen's d on Q1 only, Q2 only,
     and pooled (per-participant average), per measure.

Inputs:  data/cleaned/Game.csv
Outputs: output/m64_q1q2_orderfx_bootstrap.{csv,pdf}
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats

from helpers import load_trials, OUTPUT_DIR
from m56_eda_writing_sequential import build_question_df
from m57_covariate_analysis import cohens_d
from m60_trial_order_moderation import attach_trial_position

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'

BG = '#FFFFFF'
TEXT = '#1a1a1a'
LABEL = '#333333'
GRID = '#E0E0E0'
BORDER = '#CCCCCC'
MUTED = '#666666'
COND_COLORS = {'diffuse': '#1976D2', 'clumpy': '#C62828'}
POS_COLORS = {1: '#1976D2', 2: '#E65100', 'pooled': '#4A4A4A'}

TARGET_MEASURES = [
    'seq_typing_entropy',
    'seq_typing_mean_run_explore',  # mean reading-only streak (Kineret's measure)
    'seq_typing_max_run',
    'first_writing_time_s',
    'final_answer_length',
]

N_BOOTSTRAP = 5000
RNG_SEED = 20260430


def pivot_by_position(qdf, measure):
    """Wide table: one row per participant, columns 'Q1' and 'Q2' (trial_position)."""
    wide = qdf.pivot_table(
        index=['participant_id', 'condition'],
        columns='trial_position',
        values=measure,
        aggfunc='first',
    ).reset_index()
    wide.columns = [c if not isinstance(c, (int, float)) else f'Q{int(c)}'
                    for c in wide.columns]
    return wide


def pivot_by_domain(qdf, measure):
    wide = qdf.pivot_table(
        index=['participant_id', 'condition'],
        columns='domain',
        values=measure,
        aggfunc='first',
    ).reset_index()
    return wide


def correlation_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 10 or a.std() == 0 or b.std() == 0:
        return np.nan, np.nan, len(a)
    r, p = sp_stats.pearsonr(a, b)
    return r, p, len(a)


def order_effect_test(qdf, measure):
    """Paired t-test of measure across trial_position (within-subject)."""
    wide = pivot_by_position(qdf, measure)
    paired = wide.dropna(subset=['Q1', 'Q2'])
    if len(paired) < 10:
        return None
    q1 = paired['Q1'].to_numpy()
    q2 = paired['Q2'].to_numpy()
    diff = q2 - q1
    t_stat, p = sp_stats.ttest_rel(q1, q2)
    d_paired = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    return {
        'n_paired': len(paired),
        'mean_q1': q1.mean(), 'mean_q2': q2.mean(),
        'sd_q1': q1.std(ddof=1), 'sd_q2': q2.std(ddof=1),
        'mean_diff_q2_minus_q1': diff.mean(),
        't_paired': t_stat, 'p_paired': p,
        'cohen_dz': d_paired,
    }


def residualized_correlation(qdf, measure):
    """Subtract per-position grand mean (Q1 mean, Q2 mean) before correlating.

    This strips a linear order-shift effect and reveals within-subject
    reliability that's not contaminated by systematic Q1/Q2 differences.
    """
    wide = pivot_by_position(qdf, measure)
    paired = wide.dropna(subset=['Q1', 'Q2']).copy()
    if len(paired) < 10:
        return np.nan, np.nan, len(paired)
    paired['Q1_resid'] = paired['Q1'] - paired['Q1'].mean()
    paired['Q2_resid'] = paired['Q2'] - paired['Q2'].mean()
    # After mean-subtraction, correlation r is identical to raw r;
    # but if we want to align "AH" and "PS" labels with Q1/Q2 corrections:
    # correlation of (Q1, Q2) IS the order-aware test-retest reliability.
    return sp_stats.pearsonr(paired['Q1'], paired['Q2'])[0], \
           sp_stats.pearsonr(paired['Q1'], paired['Q2'])[1], \
           len(paired)


def bootstrap_cohens_d(d_vals, c_vals, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Bootstrap percentile CI for Cohen's d (diffuse - clumpy)."""
    rng = np.random.default_rng(seed)
    d_vals = np.asarray(d_vals, dtype=float)
    c_vals = np.asarray(c_vals, dtype=float)
    d_vals = d_vals[~np.isnan(d_vals)]
    c_vals = c_vals[~np.isnan(c_vals)]
    if len(d_vals) < 3 or len(c_vals) < 3:
        return {'d': np.nan, 'lo': np.nan, 'hi': np.nan,
                'p_directional': np.nan, 'n_d': len(d_vals), 'n_c': len(c_vals)}
    point_d = cohens_d(d_vals, c_vals)
    boots = np.empty(n_boot)
    n_d, n_c = len(d_vals), len(c_vals)
    for i in range(n_boot):
        bd = rng.choice(d_vals, size=n_d, replace=True)
        bc = rng.choice(c_vals, size=n_c, replace=True)
        boots[i] = cohens_d(bd, bc)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Two-sided percentile p: 2 * min(P(d_boot < 0), P(d_boot > 0))
    p_dir = 2 * min((boots < 0).mean(), (boots > 0).mean())
    return {
        'd': point_d, 'lo': lo, 'hi': hi,
        'p_directional': p_dir,
        'n_d': n_d, 'n_c': n_c,
    }


def bootstrap_per_position(qdf, measure):
    """Bootstrap CIs for d on each trial_position separately + pooled."""
    out = {}
    # Q1 only, Q2 only
    for pos in [1, 2]:
        sub = qdf[qdf['trial_position'] == pos]
        d_vals = sub.loc[sub['condition'] == 'diffuse', measure].dropna().to_numpy()
        c_vals = sub.loc[sub['condition'] == 'clumpy', measure].dropna().to_numpy()
        out[f'pos{pos}'] = bootstrap_cohens_d(d_vals, c_vals)
    # Pooled = per-participant average across both positions
    pooled = (qdf.groupby(['participant_id', 'condition'], as_index=False)
                  [measure].mean())
    d_vals = pooled.loc[pooled['condition'] == 'diffuse', measure].dropna().to_numpy()
    c_vals = pooled.loc[pooled['condition'] == 'clumpy', measure].dropna().to_numpy()
    out['pooled'] = bootstrap_cohens_d(d_vals, c_vals)
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M64] Q1-Q2 correlation, order effect, bootstrap CIs')
    print('=' * 60)

    print('\n--- Loading and building question-level features ---')
    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    qdf = build_question_df(trials)
    qdf = attach_trial_position(qdf, trials)

    excluded = qdf['excluded_pages'] | qdf['excluded_idle']
    qdf = qdf[~excluded].copy()
    M52_OUTLIERS = {26, 79, 134}
    qdf = qdf[~qdf['participant_id'].isin(M52_OUTLIERS)].copy()
    print(f'  Questions kept: {len(qdf)}')

    # --- Correlations -------------------------------------------------------
    print('\n=== Q1-Q2 vs AH-PS correlations ===')
    rows = []
    for m in TARGET_MEASURES:
        wide_pos = pivot_by_position(qdf, m)
        wide_dom = pivot_by_domain(qdf, m)
        r_pos, p_pos, n_pos = correlation_pair(
            wide_pos.get('Q1', pd.Series(dtype=float)).to_numpy(),
            wide_pos.get('Q2', pd.Series(dtype=float)).to_numpy(),
        )
        r_dom, p_dom, n_dom = correlation_pair(
            wide_dom.get('art_history', pd.Series(dtype=float)).to_numpy(),
            wide_dom.get('psychology', pd.Series(dtype=float)).to_numpy(),
        )
        rows.append({
            'measure': m,
            'r_q1q2': r_pos, 'p_q1q2': p_pos, 'n_q1q2': n_pos,
            'r_ahps': r_dom, 'p_ahps': p_dom, 'n_ahps': n_dom,
            'delta_r_pos_minus_dom': r_pos - r_dom,
        })
    cor_df = pd.DataFrame(rows)
    print(cor_df.to_string(index=False, float_format=lambda x: f'{x:+.3f}'))

    # --- Order effect (paired t) -------------------------------------------
    print('\n=== Order effect: paired t Q2 - Q1 ===')
    order_rows = []
    for m in TARGET_MEASURES:
        oe = order_effect_test(qdf, m)
        if oe is None:
            continue
        oe['measure'] = m
        order_rows.append(oe)
        print(f'  {m:<32}  Q1={oe["mean_q1"]:.3f}  Q2={oe["mean_q2"]:.3f}  '
              f'diff={oe["mean_diff_q2_minus_q1"]:+.3f}  '
              f't={oe["t_paired"]:+.2f}  p={oe["p_paired"]:.3f}  '
              f'd_z={oe["cohen_dz"]:+.2f}')
    order_df = pd.DataFrame(order_rows)

    # --- Bootstrap CIs ------------------------------------------------------
    print(f'\n=== Bootstrap Cohen\'s d ({N_BOOTSTRAP} resamples) ===')
    boot_rows = []
    for m in TARGET_MEASURES:
        per_pos = bootstrap_per_position(qdf, m)
        for label, r in per_pos.items():
            boot_rows.append({
                'measure': m, 'stratum': label,
                'd': r['d'], 'lo95': r['lo'], 'hi95': r['hi'],
                'boot_p_directional': r['p_directional'],
                'n_d': r['n_d'], 'n_c': r['n_c'],
            })
            print(f'  {m:<32}  {label:<7}  d={r["d"]:+.2f}  '
                  f'95% CI [{r["lo"]:+.2f}, {r["hi"]:+.2f}]  '
                  f'p={r["p_directional"]:.3f}')
        print()
    boot_df = pd.DataFrame(boot_rows)

    # --- Save CSV (long-form combined) -------------------------------------
    csv_out = OUTPUT_DIR / 'm64_q1q2_orderfx_bootstrap.csv'
    with open(csv_out, 'w', encoding='utf-8') as f:
        f.write('# Q1-Q2 correlations\n')
        cor_df.to_csv(f, index=False)
        f.write('\n# Order effect (paired t)\n')
        order_df.to_csv(f, index=False)
        f.write('\n# Bootstrap Cohen\'s d\n')
        boot_df.to_csv(f, index=False)
    print(f'\nSaved CSV: {csv_out}')

    # --- PDF report ---------------------------------------------------------
    pdf_path = OUTPUT_DIR / 'm64_q1q2_orderfx_bootstrap.pdf'
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(make_correlation_table_page(cor_df, order_df), facecolor=BG)
        plt.close()
        pdf.savefig(make_bootstrap_forest(boot_df), facecolor=BG)
        plt.close()
        pdf.savefig(make_bootstrap_table_page(boot_df), facecolor=BG)
        plt.close()
        pdf.savefig(make_q1q2_scatter_page(qdf), facecolor=BG)
        plt.close()
    print(f'Saved PDF: {pdf_path}')
    print('\nDone.')


# ----------------------------------------------------------------------
# Plot pages
# ----------------------------------------------------------------------

def make_correlation_table_page(cor_df, order_df):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M64 - Within-subject correlation: Q1-Q2 (order-aware) vs AH-PS (domain)',
                 fontsize=13, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')

    lines = []
    lines.append('CORRELATIONS')
    lines.append(f'{"Measure":<32} {"r(Q1,Q2)":>10} {"n":>4} {"r(AH,PS)":>10} '
                 f'{"n":>4} {"diff":>8}')
    lines.append('-' * 80)
    for _, r in cor_df.iterrows():
        lines.append(
            f'{r["measure"]:<32} {r["r_q1q2"]:>+10.3f} {r["n_q1q2"]:>4} '
            f'{r["r_ahps"]:>+10.3f} {r["n_ahps"]:>4} '
            f'{r["delta_r_pos_minus_dom"]:>+8.3f}'
        )
    lines.append('')
    lines.append('Yuval\'s hypothesis: if order effect drives the AH-PS zero correlation,')
    lines.append('then r(Q1,Q2) - r(AH,PS) should be POSITIVE (Q1-Q2 more reliable).')
    lines.append('')
    lines.append('ORDER EFFECT (paired t: Q2 - Q1)')
    lines.append(f'{"Measure":<32} {"mean(Q1)":>10} {"mean(Q2)":>10} {"diff":>8} '
                 f'{"t":>6} {"p":>7} {"d_z":>6}')
    lines.append('-' * 86)
    for _, r in order_df.iterrows():
        sig = '***' if r['p_paired'] < .001 else '**' if r['p_paired'] < .01 \
              else '*' if r['p_paired'] < .05 else ''
        lines.append(
            f'{r["measure"]:<32} {r["mean_q1"]:>10.3f} {r["mean_q2"]:>10.3f} '
            f'{r["mean_diff_q2_minus_q1"]:>+8.3f} {r["t_paired"]:>+6.2f} '
            f'{r["p_paired"]:>7.3f}{sig:<3} {r["cohen_dz"]:>+6.2f}'
        )
    lines.append('')
    lines.append('A significant order effect supports Yuval\'s hypothesis (warming up,')
    lines.append('practice, fatigue) - it suggests stratifying or controlling for order.')

    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes, fontsize=8.8,
            family='monospace', va='top', color=TEXT, linespacing=1.4)
    return fig


def make_bootstrap_forest(boot_df):
    measures = TARGET_MEASURES
    n = len(measures)
    fig, ax = plt.subplots(figsize=(11, 1.4 + 0.9 * n))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    offset = {'pos1': -0.25, 'pos2': +0.25, 'pooled': 0.0}
    color = {'pos1': POS_COLORS[1], 'pos2': POS_COLORS[2], 'pooled': POS_COLORS['pooled']}
    marker = {'pos1': 'o', 'pos2': 'o', 'pooled': 'D'}
    size = {'pos1': 70, 'pos2': 70, 'pooled': 110}

    for i, m in enumerate(measures):
        for stratum in ['pooled', 'pos1', 'pos2']:
            r = boot_df[(boot_df['measure'] == m) & (boot_df['stratum'] == stratum)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            y = i + offset[stratum]
            ax.errorbar(r['d'], y, xerr=[[r['d'] - r['lo95']], [r['hi95'] - r['d']]],
                        fmt=marker[stratum], color=color[stratum],
                        markersize=np.sqrt(size[stratum]),
                        ecolor=color[stratum], capsize=3, capthick=1.5,
                        elinewidth=1.5, markeredgecolor='#000' if stratum == 'pooled'
                        else color[stratum], markeredgewidth=1.0, zorder=4)
            ax.annotate(f'{r["d"]:+.2f}', (r['d'], y), xytext=(7, 0),
                        textcoords='offset points', fontsize=8,
                        color=color[stratum], va='center', fontweight='bold')

    ax.axvline(0, color=BORDER, linewidth=1)
    for thresh in [-0.5, -0.2, 0.2, 0.5]:
        ax.axvline(thresh, color=GRID, linewidth=0.5, linestyle=':')
    ax.set_yticks(range(n))
    ax.set_yticklabels(measures, fontsize=10)
    ax.set_xlabel('Cohen\'s d (Diffuse - Clumpy) with bootstrap 95% CI',
                  color=LABEL, fontweight='bold')
    ax.set_title('M64 - Bootstrap effect sizes by trial position',
                 color=TEXT, fontweight='bold', fontsize=13, pad=10)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.4, axis='x', zorder=0)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    ax.set_xlim(-1.4, 1.4)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=POS_COLORS['pooled'],
               markeredgecolor='#000', markersize=10, label='Pooled (per-subject mean)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=POS_COLORS[1],
               markersize=8, label='Q1 only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=POS_COLORS[2],
               markersize=8, label='Q2 only'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='lower right', framealpha=0.9)
    plt.tight_layout()
    return fig


def make_bootstrap_table_page(boot_df):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f'M64 - Bootstrap Cohen\'s d ({N_BOOTSTRAP} resamples)',
                 fontsize=13, fontweight='bold', color=TEXT, y=0.97)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis('off')
    lines = []
    lines.append(f'{"Measure":<32} {"Stratum":<8} {"N(D/C)":>8} '
                 f'{"d":>7} {"95% CI":>20} {"p":>7}')
    lines.append('-' * 88)
    for m in TARGET_MEASURES:
        for stratum in ['pooled', 'pos1', 'pos2']:
            r = boot_df[(boot_df['measure'] == m) & (boot_df['stratum'] == stratum)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            ci_str = f'[{r["lo95"]:+.2f}, {r["hi95"]:+.2f}]'
            lines.append(
                f'{m if stratum == "pooled" else "":<32} {stratum:<8} '
                f'{r["n_d"]:>3}/{r["n_c"]:<4} {r["d"]:>+7.2f} '
                f'{ci_str:>20} {r["boot_p_directional"]:>7.3f}'
            )
        lines.append('')
    lines.append('p = bootstrap directional p (2 * min of tails crossing zero).')
    lines.append('CI excluding 0 = effect coherent across the group.')
    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes, fontsize=8.8,
            family='monospace', va='top', color=TEXT, linespacing=1.4)
    return fig


def make_q1q2_scatter_page(qdf):
    """Scatter Q1 vs Q2 per measure, colored by condition."""
    measures = TARGET_MEASURES
    n = len(measures)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.4 * nrows))
    fig.patch.set_facecolor(BG)
    fig.suptitle('M64 - Within-subject Q1 vs Q2 scatters',
                 fontsize=13, fontweight='bold', color=TEXT, y=0.99)
    axes = np.atleast_2d(axes).ravel()
    for i, m in enumerate(measures):
        ax = axes[i]
        ax.set_facecolor(BG)
        wide = pivot_by_position(qdf, m).dropna(subset=['Q1', 'Q2'])
        for cond in ['diffuse', 'clumpy']:
            s = wide[wide['condition'] == cond]
            ax.scatter(s['Q1'], s['Q2'], color=COND_COLORS[cond], s=28, alpha=0.7,
                       edgecolors='#333', linewidth=0.3,
                       label=f'{cond} (n={len(s)})')
        if len(wide) >= 3:
            r, p = sp_stats.pearsonr(wide['Q1'], wide['Q2'])
        else:
            r, p = (np.nan, np.nan)
        # Diagonal
        lo = min(wide['Q1'].min(), wide['Q2'].min())
        hi = max(wide['Q1'].max(), wide['Q2'].max())
        pad = (hi - lo) * 0.05 if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color=BORDER, linewidth=1, linestyle=':', zorder=1)
        ax.set_title(f'{m}\nr(Q1,Q2)={r:+.2f}  p={p:.3f}',
                     fontsize=9.5, color=TEXT, fontweight='bold')
        ax.set_xlabel('Q1', color=LABEL, fontsize=9)
        ax.set_ylabel('Q2', color=LABEL, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(fontsize=7, loc='best', framealpha=0.85)
        ax.grid(True, color=GRID, linewidth=0.4, zorder=0)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for j in range(len(measures), len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


if __name__ == '__main__':
    main()

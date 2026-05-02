#!/usr/bin/env python3
"""
M67: Bootstrap CI for Mean reading-only streak
==============================================
Bootstrap confidence intervals for the condition effect on
seq_typing_mean_run_explore (Mean reading-only streak).

Two effect sizes are bootstrapped in parallel, matching the two columns
already shown in M66 panel B:
  - raw Cohen's d  (Diffuse - Clumpy, pooled SD)
  - adjusted Cohen's d  (OLS regression on condition + final_answer_length,
                         denominator = sqrt(MSE))

Procedure:
  - Stratified resampling at the participant level (separately within
    Diffuse and Clumpy, with replacement) so group sizes stay fixed.
  - 10,000 iterations.
  - 95% percentile CI from the bootstrap distribution.
  - Empirical two-sided p-value: 2 * min(P(b >= 0), P(b <= 0)).

Inputs:  output/m56_eda_writing_sequential.csv  (per-participant averaged)
Outputs: output/m67_bootstrap_streak.csv         (full distribution)
         output/m67_bootstrap_streak_summary.csv (observed + CI + p)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from helpers import OUTPUT_DIR

MEASURE = 'seq_typing_mean_run_explore'
COVARIATE = 'final_answer_length'
N_BOOT = 10_000
SEED = 42


def cohens_d_pooled(d_vals, c_vals):
    n_d, n_c = len(d_vals), len(c_vals)
    if n_d < 2 or n_c < 2:
        return np.nan
    pooled = np.sqrt(((n_d - 1) * d_vals.std(ddof=1) ** 2
                      + (n_c - 1) * c_vals.std(ddof=1) ** 2)
                     / (n_d + n_c - 2))
    if pooled == 0:
        return np.nan
    return (d_vals.mean() - c_vals.mean()) / pooled


def adjusted_cohens_d(y, cond_diffuse, cov):
    """OLS y = b0 + b1*cond_diffuse + b2*cov, return b1 / sqrt(MSE)."""
    n = len(y)
    X = np.column_stack([np.ones(n), cond_diffuse.astype(float), cov.astype(float)])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    df_resid = n - X.shape[1]
    if df_resid <= 0:
        return np.nan
    sigma2 = (resid @ resid) / df_resid
    if sigma2 <= 0:
        return np.nan
    return beta[1] / np.sqrt(sigma2)


def two_sided_empirical_p(boot_vals):
    """Two-sided empirical p: 2 * min(P(b >= 0), P(b <= 0))."""
    p_pos = np.mean(boot_vals >= 0)
    p_neg = np.mean(boot_vals <= 0)
    return float(min(1.0, 2 * min(p_pos, p_neg)))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M67] Bootstrap CI for Mean reading-only streak')
    print('=' * 60)

    src = OUTPUT_DIR / 'm56_eda_writing_sequential.csv'
    if not src.exists():
        raise FileNotFoundError(f'Run M56 first to produce {src}')
    df = pd.read_csv(src)
    print(f'Loaded N={len(df)} participants from {src}')

    sub = df[['condition', MEASURE, COVARIATE]].dropna().reset_index(drop=True)
    print(f'After dropping NaN on {MEASURE} or {COVARIATE}: N={len(sub)}')

    diffuse_idx = sub.index[sub['condition'] == 'diffuse'].to_numpy()
    clumpy_idx = sub.index[sub['condition'] == 'clumpy'].to_numpy()
    n_d, n_c = len(diffuse_idx), len(clumpy_idx)
    print(f'  Diffuse: n={n_d}    Clumpy: n={n_c}')

    y_all = sub[MEASURE].to_numpy(dtype=float)
    cov_all = sub[COVARIATE].to_numpy(dtype=float)
    cond_diffuse_all = (sub['condition'].to_numpy() == 'diffuse').astype(float)

    obs_raw = cohens_d_pooled(y_all[cond_diffuse_all == 1.0],
                              y_all[cond_diffuse_all == 0.0])
    obs_adj = adjusted_cohens_d(y_all, cond_diffuse_all, cov_all)
    print(f'\nObserved raw Cohen\'s d      = {obs_raw:+.4f}')
    print(f'Observed adjusted Cohen\'s d = {obs_adj:+.4f}')

    print(f'\nRunning {N_BOOT:,} stratified bootstrap iterations...')
    rng = np.random.default_rng(SEED)
    boot_raw = np.empty(N_BOOT)
    boot_adj = np.empty(N_BOOT)

    for b in range(N_BOOT):
        d_pick = rng.choice(diffuse_idx, size=n_d, replace=True)
        c_pick = rng.choice(clumpy_idx, size=n_c, replace=True)
        idx = np.concatenate([d_pick, c_pick])

        y_b = y_all[idx]
        cov_b = cov_all[idx]
        cond_b = np.concatenate([np.ones(n_d), np.zeros(n_c)])

        boot_raw[b] = cohens_d_pooled(y_b[:n_d], y_b[n_d:])
        boot_adj[b] = adjusted_cohens_d(y_b, cond_b, cov_b)

        if (b + 1) % 2000 == 0:
            print(f'  ...{b + 1:,} done')

    raw_clean = boot_raw[~np.isnan(boot_raw)]
    adj_clean = boot_adj[~np.isnan(boot_adj)]

    raw_lo, raw_hi = np.percentile(raw_clean, [2.5, 97.5])
    adj_lo, adj_hi = np.percentile(adj_clean, [2.5, 97.5])

    raw_p = two_sided_empirical_p(raw_clean)
    adj_p = two_sided_empirical_p(adj_clean)

    print('\n--- Results ---')
    print(f'Raw d:       observed = {obs_raw:+.3f}   '
          f'95% CI = [{raw_lo:+.3f}, {raw_hi:+.3f}]   p_emp = {raw_p:.4f}')
    print(f'Adjusted d:  observed = {obs_adj:+.3f}   '
          f'95% CI = [{adj_lo:+.3f}, {adj_hi:+.3f}]   p_emp = {adj_p:.4f}')

    # Full distributions
    boot_df = pd.DataFrame({
        'iteration': np.arange(N_BOOT),
        'boot_raw_d': boot_raw,
        'boot_adj_d': boot_adj,
    })
    out_dist = OUTPUT_DIR / 'm67_bootstrap_streak.csv'
    boot_df.to_csv(out_dist, index=False)
    print(f'\nSaved distribution: {out_dist}')

    # Summary
    summary = pd.DataFrame([
        {
            'effect_size': 'raw_cohen_d',
            'measure': MEASURE,
            'n': len(sub),
            'n_diffuse': n_d,
            'n_clumpy': n_c,
            'observed': obs_raw,
            'ci_lo_2.5': raw_lo,
            'ci_hi_97.5': raw_hi,
            'p_empirical': raw_p,
            'n_boot': N_BOOT,
            'n_boot_valid': len(raw_clean),
        },
        {
            'effect_size': 'adj_cohen_d',
            'measure': MEASURE,
            'n': len(sub),
            'n_diffuse': n_d,
            'n_clumpy': n_c,
            'observed': obs_adj,
            'ci_lo_2.5': adj_lo,
            'ci_hi_97.5': adj_hi,
            'p_empirical': adj_p,
            'n_boot': N_BOOT,
            'n_boot_valid': len(adj_clean),
        },
    ])
    out_sum = OUTPUT_DIR / 'm67_bootstrap_streak_summary.csv'
    summary.to_csv(out_sum, index=False)
    print(f'Saved summary:      {out_sum}')

    print('\nDone.')


if __name__ == '__main__':
    main()

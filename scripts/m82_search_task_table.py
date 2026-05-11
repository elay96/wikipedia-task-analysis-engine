#!/usr/bin/env python3
"""
M82: SEARCH measures x Wikipedia style x TASK correlations
==========================================================
EDA table joining four per-participant SEARCH measures (two shallow + two
deep, from M81) with Wikipedia style groupings (Hunter / Busybody for k=2,
+ Dancer for k=3, refit on M80's features) and six TASK measures (M71/M72/M73).

Inputs (must already exist):
  output/m81_per_trial_features.csv
  output/m80_hunter_busybody_per_participant.csv
  output/m71_per_participant_reading_switches.csv
  output/m72_new_per_participant.csv
  output/m73_new_per_participant_entropy.csv

Outputs:
  output/m82_per_participant.csv
  output/m82_correlations_long.csv
  output/m82_groupstats.csv
  output/m82_search_task_table_k2.html
  output/m82_search_task_table_k3.html
  docs/m82_findings.html
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = ROOT / 'output'
DOCS_DIR = ROOT / 'docs'

M81_PER_TRIAL = OUTPUT_DIR / 'm81_per_trial_features.csv'
M80_STYLE = OUTPUT_DIR / 'm80_hunter_busybody_per_participant.csv'
M71_READING = OUTPUT_DIR / 'm71_per_participant_reading_switches.csv'
M72_TASK = OUTPUT_DIR / 'm72_new_per_participant.csv'
M73_ENTROPY = OUTPUT_DIR / 'm73_new_per_participant_entropy.csv'

PER_PID_OUT = OUTPUT_DIR / 'm82_per_participant.csv'
CORR_OUT = OUTPUT_DIR / 'm82_correlations_long.csv'
GROUPSTATS_OUT = OUTPUT_DIR / 'm82_groupstats.csv'
TABLE_K2_OUT = OUTPUT_DIR / 'm82_search_task_table_k2.html'
TABLE_K3_OUT = OUTPUT_DIR / 'm82_search_task_table_k3.html'
FINDINGS_OUT = DOCS_DIR / 'm82_findings.html'

RANDOM_STATE = 42

SEARCH_MEASURES = [
    ('time_to_first_resource', 'Time to 1st reward (s)', 'shallow'),
    ('inter_resource_mean', 'Mean inter-reward time (s)', 'shallow'),
    ('pct_time_exploit', '% time in exploit', 'deep'),
    ('n_transitions', 'N transitions', 'deep'),
]
TASK_MEASURES = [
    ('mean_reading_length_s', 'Read len'),
    ('count_time', 'Sw: time'),
    ('count_topic', 'Sw: topic'),
    ('count_typing', 'Sw: typing'),
    ('PC1', 'PC1'),
    ('seq_typing_entropy', 'Entropy'),
]

GROUPS_K2 = ['All', 'hunter', 'busybody']
GROUPS_K3 = ['All', 'hunter', 'busybody', 'dancer']

# Colour palette - matches docs/m82_table_mockup.html
SHALLOW_BG = '#FFF8E1'
DEEP_BG = '#E3F2FD'
HUNTER_COLOR = '#1976D2'
BUSYBODY_COLOR = '#E65100'
DANCER_COLOR = '#2E7D32'


def load_search_features():
    """Mean across the 5 trials of each per-trial SEARCH measure from M81."""
    df = pd.read_csv(M81_PER_TRIAL)
    df['condition'] = df['condition'].astype(str).str.lower()
    cols = [c for c, _, _ in SEARCH_MEASURES]
    grouped = df.groupby(['participant_id', 'condition'])[cols].mean().reset_index()
    grouped['participant_id'] = grouped['participant_id'].astype(int)
    return grouped


def load_task_features():
    """Join the six TASK measures from M71, M72 and M73 on participant_id."""
    m71 = pd.read_csv(M71_READING)[['participant_id', 'mean_reading_length_s']]
    m72 = pd.read_csv(M72_TASK)[['participant_id', 'count_time',
                                  'count_topic', 'count_typing', 'PC1']]
    m73 = pd.read_csv(M73_ENTROPY)[['participant_id', 'seq_typing_entropy']]
    out = m71.merge(m72, on='participant_id', how='outer') \
             .merge(m73, on='participant_id', how='outer')
    out['participant_id'] = out['participant_id'].astype(int)
    return out


def load_styles():
    """Load M80's style as style_k2; refit KMeans k=3 to add style_k3.

    Both labels come from clustering standardised
    (topic_concentration, transition_entropy). For k=3 we name the cluster
    with the highest mean topic_concentration 'hunter', the lowest
    'busybody', and the remaining one 'dancer'.
    """
    df = pd.read_csv(M80_STYLE)
    df['participant_id'] = df['participant_id'].astype(int)
    df = df.rename(columns={'style': 'style_k2'})

    feats = ['topic_concentration', 'transition_entropy']
    mask = df[feats].notna().all(axis=1)
    X = df.loc[mask, feats].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(Xs)

    means = [X[labels == k, 0].mean() for k in range(3)]
    order = np.argsort(means)            # 0=lowest -> busybody, 2=highest -> hunter
    name_for = {order[0]: 'busybody', order[1]: 'dancer', order[2]: 'hunter'}
    df['style_k3'] = ''
    df.loc[mask, 'style_k3'] = [name_for[lab] for lab in labels]
    df.loc[df['style_k3'] == '', 'style_k3'] = np.nan

    return df[['participant_id', 'style_k2', 'style_k3',
               'topic_concentration', 'transition_entropy']]


def build_per_pid(search_df, task_df, styles_df):
    """Inner-join SEARCH and styles on participant_id; left-join TASK."""
    df = search_df.merge(styles_df, on='participant_id', how='inner')
    df = df.merge(task_df, on='participant_id', how='left')
    front = ['participant_id', 'condition', 'style_k2', 'style_k3',
             'topic_concentration', 'transition_entropy']
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest].sort_values('participant_id').reset_index(drop=True)
    return df


def _welch_d_and_p(values_clumpy, values_diffuse):
    """Welch t-test + Cohen's d (pooled SD). Returns (d, t, p)."""
    a = np.asarray(values_clumpy, dtype=float)
    b = np.asarray(values_diffuse, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * s1 + (len(b) - 1) * s2) /
                     (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
    t_res = sp_stats.ttest_ind(a, b, equal_var=False)
    return float(d), float(t_res.statistic), float(t_res.pvalue)


def compute_groupstats(per_pid):
    """Per (measure, group, k): N, mean, sd, n_clumpy, n_diffuse,
    mean_clumpy, mean_diffuse, sd_clumpy, sd_diffuse, cohens_d, t, p."""
    rows = []
    for k, groups, col in [(2, GROUPS_K2, 'style_k2'),
                            (3, GROUPS_K3, 'style_k3')]:
        for group in groups:
            sub = per_pid if group == 'All' else per_pid[per_pid[col] == group]
            for src, _, _ in SEARCH_MEASURES:
                v = sub[src].dropna().values
                clumpy = sub.loc[sub['condition'] == 'clumpy', src].dropna().values
                diffuse = sub.loc[sub['condition'] == 'diffuse', src].dropna().values
                d, t, p = _welch_d_and_p(clumpy, diffuse)
                rows.append({
                    'k': k, 'group': group, 'measure': src,
                    'n': len(v),
                    'mean': float(v.mean()) if len(v) else np.nan,
                    'sd': float(v.std(ddof=1)) if len(v) >= 2 else np.nan,
                    'n_clumpy': len(clumpy), 'n_diffuse': len(diffuse),
                    'mean_clumpy': float(clumpy.mean()) if len(clumpy) else np.nan,
                    'mean_diffuse': float(diffuse.mean()) if len(diffuse) else np.nan,
                    'sd_clumpy': float(clumpy.std(ddof=1)) if len(clumpy) >= 2 else np.nan,
                    'sd_diffuse': float(diffuse.std(ddof=1)) if len(diffuse) >= 2 else np.nan,
                    'cohens_d': d, 't': t, 'p': p,
                })
    return pd.DataFrame(rows)


def compute_correlations_long(per_pid):
    """Spearman rho per (k, group, search_measure, task_measure)."""
    rows = []
    for k, groups, col in [(2, GROUPS_K2, 'style_k2'),
                            (3, GROUPS_K3, 'style_k3')]:
        for group in groups:
            sub = per_pid if group == 'All' else per_pid[per_pid[col] == group]
            for src, _, _ in SEARCH_MEASURES:
                for tgt, _ in TASK_MEASURES:
                    pair = sub[[src, tgt]].dropna()
                    if len(pair) < 3:
                        rows.append({'k': k, 'group': group,
                                     'search_measure': src, 'task_measure': tgt,
                                     'n': len(pair), 'rho': np.nan, 'p': np.nan})
                        continue
                    res = sp_stats.spearmanr(pair[src], pair[tgt])
                    rows.append({'k': k, 'group': group,
                                 'search_measure': src, 'task_measure': tgt,
                                 'n': len(pair),
                                 'rho': float(res.correlation),
                                 'p': float(res.pvalue)})
    return pd.DataFrame(rows)


def _rho_color(rho):
    """Continuous red->white->blue scale. Input rho in [-1, 1]. Returns hex."""
    if rho is None or not np.isfinite(rho):
        return '#FFFFFF', '#1a1a1a'
    r = float(np.clip(rho, -1.0, 1.0))
    # Anchor colours mirror docs/m82_table_mockup.html.
    if r >= 0:
        anchors = [(0.0, (0xFF, 0xFF, 0xFF)), (0.15, (0xE3, 0xF2, 0xFD)),
                   (0.30, (0xBB, 0xDE, 0xFB)), (0.50, (0x90, 0xCA, 0xF9)),
                   (0.70, (0x42, 0xA5, 0xF5)), (1.00, (0x15, 0x65, 0xC0))]
    else:
        anchors = [(0.00, (0xFF, 0xFF, 0xFF)), (0.15, (0xFF, 0xEB, 0xEE)),
                   (0.30, (0xFF, 0xCD, 0xD2)), (0.50, (0xEF, 0x9A, 0x9A)),
                   (0.70, (0xE5, 0x73, 0x73)), (1.00, (0xC6, 0x28, 0x28))]
        r = -r
    for i in range(1, len(anchors)):
        lo, lo_rgb = anchors[i - 1]
        hi, hi_rgb = anchors[i]
        if r <= hi:
            f = 0.0 if hi == lo else (r - lo) / (hi - lo)
            rgb = tuple(int(round(lo_rgb[j] + f * (hi_rgb[j] - lo_rgb[j])))
                        for j in range(3))
            bg = f'#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}'
            fg = '#FFFFFF' if (sum(rgb) < 320) else '#1a1a1a'
            return bg, fg
    return '#FFFFFF', '#1a1a1a'


def _format_d_cell(d, p):
    if not np.isfinite(d):
        return '<td class="cohensd">-</td>'
    sign = '+' if d >= 0 else '&minus;'
    val = f'{sign}{abs(d):.2f}'
    sig_class = ' sig' if (np.isfinite(p) and p < 0.05) else ''
    return f'<td class="cohensd{sig_class}">{val}</td>'


def _format_rho_cell(rho, p):
    if not np.isfinite(rho):
        return '<td class="r">-</td>'
    bg, fg = _rho_color(rho)
    sign = '+' if rho >= 0 else '&minus;'
    val = f'{sign}{int(round(abs(rho) * 100)):02d}'
    sig_outline = ('outline: 2px solid #1a1a1a; outline-offset: -2px;'
                   if (np.isfinite(p) and p < 0.05) else '')
    return (f'<td class="r" style="background:{bg};color:{fg};{sig_outline}">'
            f'{val}</td>')


def _build_rows_for_k(groupstats, corr_long, k):
    """Yields HTML row blocks for one k value."""
    groups = GROUPS_K2 if k == 2 else GROUPS_K3
    blocks = []
    for src, label, depth in SEARCH_MEASURES:
        cell_class = 'measure-cell ' + depth
        rowspan = len(groups)
        first = True
        for g in groups:
            gs = groupstats[(groupstats['k'] == k) &
                            (groupstats['group'] == g) &
                            (groupstats['measure'] == src)].iloc[0]
            group_class = (f'group-cell {g}' if g != 'All' else 'group-cell')
            group_label = g[0].upper() + g[1:] if g != 'All' else 'All'

            tr = ['<tr>']
            if first:
                tr.append(f'<td class="{cell_class}" rowspan="{rowspan}">{label}</td>'
                          f'<td rowspan="{rowspan}">{depth}</td>')
                first = False
            tr.append(f'<td class="{group_class}">{group_label}</td>')
            tr.append(f'<td>{int(gs["n"])}</td>')
            if np.isfinite(gs['mean']) and np.isfinite(gs['sd']):
                tr.append(f'<td>{gs["mean"]:.2f} &plusmn; {gs["sd"]:.2f}</td>')
            else:
                tr.append('<td>-</td>')
            tr.append(_format_d_cell(gs['cohens_d'], gs['p']))
            for tgt, _ in TASK_MEASURES:
                row = corr_long[(corr_long['k'] == k) &
                                (corr_long['group'] == g) &
                                (corr_long['search_measure'] == src) &
                                (corr_long['task_measure'] == tgt)].iloc[0]
                tr.append(_format_rho_cell(row['rho'], row['p']))
            tr.append('</tr>')
            blocks.append(''.join(tr))
    return '\n    '.join(blocks)


TABLE_CSS = """\
:root{--bg:#FFFFFF;--text:#1a1a1a;--muted:#666;--border:#D8D8D8;
--header-bg:#F0F4F8;--row-alt:#FAFAFA;--accent:#1976D2;}
body{font-family:-apple-system,"Segoe UI",Arial,sans-serif;background:var(--bg);
color:var(--text);line-height:1.5;max-width:1280px;margin:28px auto;padding:0 24px;}
h1{color:var(--text);border-bottom:2px solid var(--accent);padding-bottom:8px;font-size:22px;}
table{border-collapse:collapse;width:100%;margin:14px 0;direction:ltr;font-size:12.5px;}
th,td{border:1px solid var(--border);padding:6px 8px;text-align:center;vertical-align:middle;
font-variant-numeric:tabular-nums;}
th{background:var(--header-bg);font-weight:600;}
td.measure-cell{text-align:left;padding-left:10px;font-weight:600;background:var(--row-alt);min-width:130px;}
td.measure-cell.shallow{background:""" + SHALLOW_BG + """;}
td.measure-cell.deep{background:""" + DEEP_BG + """;}
td.group-cell{text-align:left;padding-left:10px;font-weight:500;}
td.group-cell.hunter{color:""" + HUNTER_COLOR + """;}
td.group-cell.busybody{color:""" + BUSYBODY_COLOR + """;}
td.group-cell.dancer{color:""" + DANCER_COLOR + """;}
td.cohensd.sig{font-weight:700;}
td.r{width:56px;font-weight:600;}
caption{caption-side:top;text-align:right;direction:rtl;padding:6px 4px 10px;color:var(--muted);font-size:12px;}
"""


def render_table_html(groupstats, corr_long, k, n_total):
    headers_search = '<th rowspan="2">Measure</th><th rowspan="2">Depth</th>' \
                     '<th rowspan="2">Group</th><th rowspan="2">N</th>' \
                     '<th rowspan="2">Mean &plusmn; SD</th>' \
                     '<th rowspan="2">Cohen\'s d<br>Clumpy &minus; Diffuse</th>'
    headers_task = ''.join(f'<th>{lbl}</th>' for _, lbl in TASK_MEASURES)
    rows = _build_rows_for_k(groupstats, corr_long, k)
    caption = (f'k={k} ({", ".join(g for g in (GROUPS_K2 if k == 2 else GROUPS_K3) if g != "All")}).'
               f' N={n_total} participants. EDA: p&lt;.05 uncorrected '
               '= bold (Cohen\'s d) or black outline (correlation cells).')
    return f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>M82 - SEARCH x Style x TASK (k={k})</title>
<style>{TABLE_CSS}</style>
</head>
<body>
<h1>M82 &mdash; SEARCH measures &times; Style (k={k}) &times; TASK correlations</h1>
<table>
  <caption>{caption}</caption>
  <thead>
    <tr>
      {headers_search}
      <th colspan="6">Correlation with TASK measure (Spearman &rho; &times; 100)</th>
    </tr>
    <tr>{headers_task}</tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
</body>
</html>
"""


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    print('M82: building search-task table...')
    search_df = load_search_features()
    task_df = load_task_features()
    styles_df = load_styles()

    per_pid = build_per_pid(search_df, task_df, styles_df)
    per_pid.to_csv(PER_PID_OUT, index=False, float_format='%.6g')
    print(f'wrote {PER_PID_OUT.name}: {len(per_pid)} rows, {len(per_pid.columns)} cols')

    groupstats = compute_groupstats(per_pid)
    groupstats.to_csv(GROUPSTATS_OUT, index=False, float_format='%.6g')
    print(f'wrote {GROUPSTATS_OUT.name}: {len(groupstats)} rows')

    corr_long = compute_correlations_long(per_pid)
    corr_long.to_csv(CORR_OUT, index=False, float_format='%.6g')
    print(f'wrote {CORR_OUT.name}: {len(corr_long)} rows')

    n_total = len(per_pid)
    TABLE_K2_OUT.write_text(
        render_table_html(groupstats, corr_long, 2, n_total), encoding='utf-8')
    print(f'wrote {TABLE_K2_OUT.name}')
    TABLE_K3_OUT.write_text(
        render_table_html(groupstats, corr_long, 3, n_total), encoding='utf-8')
    print(f'wrote {TABLE_K3_OUT.name}')


if __name__ == '__main__':
    main()

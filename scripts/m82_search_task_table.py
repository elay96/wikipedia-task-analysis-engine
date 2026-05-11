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


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    print('M82: building search-task table...')
    search_df = load_search_features()
    print(f'  search measures: {search_df.shape} (expect ~132 rows x 6 cols)')
    print(search_df[[c for c, _, _ in SEARCH_MEASURES]].describe().round(2))

    task_df = load_task_features()
    print(f'  task measures: {task_df.shape} (expect ~132 rows x 7 cols)')

    styles_df = load_styles()
    print('  style_k2 counts:', styles_df['style_k2'].value_counts().to_dict())
    print('  style_k3 counts:', styles_df['style_k3'].value_counts().to_dict())


if __name__ == '__main__':
    main()

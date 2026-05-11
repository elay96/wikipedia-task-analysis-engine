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


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    print('M82: building search-task table...')
    search_df = load_search_features()
    print(f'  search measures: {search_df.shape} (expect ~132 rows x 6 cols)')
    print(search_df[[c for c, _, _ in SEARCH_MEASURES]].describe().round(2))


if __name__ == '__main__':
    main()

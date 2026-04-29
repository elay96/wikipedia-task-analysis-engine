#!/usr/bin/env python3
"""
M62: Exclusion audit page only (single-page PDF)
================================================
Reproduces M55 page 1 (the unified Exclusion Audit funnel + per-task split)
as a standalone single-page PDF for inclusion in advisor emails.

No new logic - calls the same plot_unified_audit_slide used by M55.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

from helpers import load_trials, OUTPUT_DIR
from m52_final_composite_dv import build_question_data, apply_exclusions
from m55_exclusion_audit import plot_unified_audit_slide

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[M62] Standalone Exclusion Audit page (page 1 of M55)')

    trials = load_trials(DATA_DIR / 'cleaned' / 'Game.csv')
    question_df = build_question_data(trials)
    avg_df, exclusion_summary = apply_exclusions(question_df)

    pdf_path = OUTPUT_DIR / 'm62_exclusion_audit.pdf'
    with PdfPages(pdf_path) as pdf:
        plot_unified_audit_slide(
            pdf, question_df, exclusion_summary,
            final_pids=avg_df['participant_id'].values,
            title='Wikipedia task: exclusion audit (final N = 101)',
        )
    print(f'Saved: {pdf_path}')


if __name__ == '__main__':
    main()

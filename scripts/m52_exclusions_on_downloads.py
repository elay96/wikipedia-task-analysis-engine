#!/usr/bin/env python3
"""
Apply M52 exclusion criteria to the Game.csv and KeyTable.csv in ~/Downloads,
and emit an HTML KeyTable where excluded participants are struck through in red.

Criteria (from scripts/m52_final_composite_dv.py):
  1. Trial excluded if n_pages < 3
  2. Trial excluded if idle_pct >= 50 (after last meaningful event)
  3. Participant excluded if any count (time/topic/typing) > 3 SD from mean
Plus: participant fully excluded if ALL of their real trials are dropped by (1)+(2).
"""
import json
import sys
import html
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from helpers import load_trials, get_pids_and_trials

DOWNLOADS = Path.home() / 'Downloads'
CLEANED_DIR = SCRIPT_DIR / '..' / 'data' / 'cleaned'
GAME_CSV = DOWNLOADS / 'Game.csv'
KEYTABLE_CSV = DOWNLOADS / 'KeyTable.csv'
TOPIC_MODEL = CLEANED_DIR / 'topic_model.json'
OUT_HTML = DOWNLOADS / 'KeyTable_m52.html'

# Known test/developer accounts to ignore even if they appear in KeyTable.csv.
IGNORED_PIDS = {69, 70}

# Prolific IDs flagged NO CODE that have NO session data in Game.csv/KeyTable -
# these should be asked to RETURN on Prolific rather than rejected.
ORPHAN_NO_CODE_IDS = [
    ('57bd287fe028bc00010488ab', '23 Apr 2026, 17:27', '00:40:33'),
    ('67a517b3ce9ad5ca6839e028', '23 Apr 2026, 19:23', '00:39:39'),
]

THRESHOLD_S = 60.0
IDLE_THRESHOLD_PCT = 50.0
MIN_PAGE_VISITS = 3
OUTLIER_SD = 3

MEANINGFUL_ACTIONS = ['article_open', 'search', 'link_click', 'back_navigation', 'paste']
SNAPSHOT_ACTIONS = ['answer_snapshot', 'answer_snapshot_cursor_leave']


def load_lda_assignments():
    with open(TOPIC_MODEL, encoding='utf-8') as f:
        tm = json.load(f)
    return {slug.replace('_', ' '): int(np.argmax(dist))
            for slug, dist in tm['topic_distributions'].items()}


def compute_switch_count(labels):
    if len(labels) < 2:
        return np.nan
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def page_had_typing_or_paste(pv, typing_intervals, paste_times):
    ps, pe = pv['start'], pv['end']
    for bs, be in typing_intervals:
        if bs < pe and be > ps:
            return True
    for pt in paste_times:
        if ps <= pt <= pe:
            return True
    return False


def compute_idle_pct(events_df, t0, t_end):
    total = (t_end - t0).total_seconds()
    if total <= 0:
        return np.nan
    meaningful = events_df[events_df['Action'].isin(MEANINGFUL_ACTIONS)]
    snaps = events_df[events_df['Action'].isin(SNAPSHOT_ACTIONS)].copy()
    if len(snaps) > 1:
        snaps['prev_len'] = snaps['AnswerLength'].shift(1)
        writing = snaps[snaps['AnswerLength'] != snaps['prev_len']]
    else:
        writing = snaps.iloc[:0]
    allm = pd.concat([meaningful, writing]).sort_values('Time')
    allm = allm[(allm['Time'] >= t0) & (allm['Time'] <= t_end)]
    if len(allm) == 0:
        return 100.0
    last = allm['Time'].iloc[-1]
    return ((t_end - last).total_seconds() / total) * 100


def build_question_df(trials, lda):
    rows = []
    for tr in trials:
        if tr['domain'] == 'practice':
            continue
        pvs = tr['page_visits']
        n_pages = len(pvs)
        t0 = tr['t0']
        t_end = t0 + pd.Timedelta(seconds=tr['duration'])
        idle = compute_idle_pct(tr['events'], t0, t_end)
        time_labels = ['exploit' if pv['duration'] > THRESHOLD_S else 'explore' for pv in pvs]
        topic_labels = [lda.get(pv['title'], -1) for pv in pvs]
        typing_labels = [page_had_typing_or_paste(pv, tr['typing_intervals'], tr['paste_times']) for pv in pvs]
        rows.append({
            'pid': tr['pid'],
            'n_pages': n_pages,
            'idle_pct': idle,
            'excl_pages': n_pages < MIN_PAGE_VISITS,
            'excl_idle': (idle is not np.nan) and (idle >= IDLE_THRESHOLD_PCT),
            'count_time': compute_switch_count(time_labels),
            'count_topic': compute_switch_count(topic_labels),
            'count_typing': compute_switch_count(typing_labels),
        })
    return pd.DataFrame(rows)


def short_message(pid, qdf, kind, outlier_cols=None):
    """Short, ready-to-copy rejection sentence tailored to this participant."""
    sub = qdf[qdf['pid'] == pid].sort_values(qdf.columns[0])
    pages = sub['n_pages'].tolist()
    idles = sub['idle_pct'].tolist()

    if kind == 'outlier' and outlier_cols:
        return ('Response pattern falls substantially outside the normal range '
                f'on {", ".join(outlier_cols)}.')

    # Fully excluded - pick the tightest description
    all_pages_low = all(p < MIN_PAGE_VISITS for p in pages)
    all_idle_high = all((not np.isnan(i)) and i >= IDLE_THRESHOLD_PCT for i in idles)

    if all_pages_low and all(p == 1 for p in pages):
        return ('Visited only a single Wikipedia page per task, although the '
                'instructions asked participants to search across multiple pages.')
    if all_pages_low:
        return ('Visited only ' + ' and '.join(str(p) for p in pages) +
                ' Wikipedia pages across the two tasks, below the multi-page '
                'exploration requirement.')
    if all_idle_high:
        return ('Spent ' + ' and '.join(f'{i:.0f}%' for i in idles) +
                ' of each task idle with no meaningful activity.')
    # Mixed - just list counts
    return ('Visited only ' + ' and '.join(str(p) for p in pages) +
            ' Wikipedia pages and was idle ' +
            ' and '.join(f'{i:.0f}%' for i in idles) + ' of each task.')


def run_exclusions(qdf):
    all_pids = set(qdf['pid'].unique())
    kept_q = qdf[~(qdf['excl_pages'] | qdf['excl_idle'])].copy()
    kept_pids = set(kept_q['pid'].unique())
    fully_excluded = all_pids - kept_pids

    reasons = {}
    messages = {}

    for pid in fully_excluded:
        sub = qdf[qdf['pid'] == pid]
        n_p = int(sub['excl_pages'].sum())
        n_i = int(sub['excl_idle'].sum())
        parts = []
        if n_p:
            parts.append(f'<{MIN_PAGE_VISITS} pages on all {n_p} trials')
        if n_i:
            parts.append(f'idle>=50% on {n_i} trials')
        reasons[int(pid)] = '; '.join(parts) if parts else 'all trials excluded'
        messages[int(pid)] = short_message(pid, qdf, 'fully')

    # 3 SD outliers on participant averages
    cols = ['count_time', 'count_topic', 'count_typing']
    avg = kept_q.groupby('pid')[cols].mean().reset_index().dropna()
    outlier_pids = set()
    outlier_details = {}
    outlier_cols_by_pid = {}
    for c in cols:
        m, s = avg[c].mean(), avg[c].std()
        lo, hi = m - OUTLIER_SD * s, m + OUTLIER_SD * s
        mask = (avg[c] < lo) | (avg[c] > hi)
        for pid in avg.loc[mask, 'pid']:
            v = avg.loc[avg['pid'] == pid, c].values[0]
            outlier_pids.add(int(pid))
            outlier_details.setdefault(int(pid), []).append(
                f'{c}={v:.2f} (mean={m:.2f}, sd={s:.2f})'
            )
            outlier_cols_by_pid.setdefault(int(pid), []).append(c.replace('count_', ''))
    for pid in outlier_pids:
        reasons[int(pid)] = '3SD outlier: ' + '; '.join(outlier_details[pid])
        messages[int(pid)] = short_message(pid, qdf, 'outlier',
                                           outlier_cols=outlier_cols_by_pid[pid])

    excluded_pids = {int(p) for p in fully_excluded} | outlier_pids
    return excluded_pids, reasons, messages, fully_excluded, outlier_pids


def render_html(keytable, excluded, reasons, messages, stats):
    rows_html = []
    for _, r in keytable.iterrows():
        pid = int(r['ID'])
        is_excluded = pid in excluded
        reason = reasons.get(pid, '')
        message = messages.get(pid, '')
        cells = ''.join(f'<td>{html.escape(str(r[c]))}</td>' for c in keytable.columns)
        cls = ' class="excluded"' if is_excluded else ''
        reason_cell = f'<td class="reason">{html.escape(reason)}</td>'
        msg_cell = f'<td class="msg">{html.escape(message)}</td>'
        rows_html.append(f'<tr{cls}>{cells}{reason_cell}{msg_cell}</tr>')
    header_cells = ''.join(f'<th>{html.escape(c)}</th>' for c in keytable.columns)
    header_cells += '<th>Exclusion reason</th><th>Rejection message</th>'

    n_total = len(keytable)
    n_excluded = sum(1 for pid in keytable['ID'] if int(pid) in excluded)
    n_kept = n_total - n_excluded

    stats_html = f"""
    <div class="stats">
      <div><span class="num">{n_total}</span><span class="label">Participants in KeyTable</span></div>
      <div><span class="num kept">{n_kept}</span><span class="label">Kept (passed M52)</span></div>
      <div><span class="num excl">{n_excluded}</span><span class="label">Excluded by M52</span></div>
    </div>
    """

    criteria_html = f"""
    <div class="criteria">
      <h2>M52 Exclusion Criteria</h2>
      <ol>
        <li>Trial excluded if fewer than <b>{MIN_PAGE_VISITS}</b> page visits</li>
        <li>Trial excluded if idle time after last meaningful event &ge; <b>{IDLE_THRESHOLD_PCT:.0f}%</b></li>
        <li>Participant excluded if any of count_time / count_topic / count_typing
            is more than <b>{OUTLIER_SD} SD</b> from the group mean</li>
        <li>Participant fully excluded if every real trial is dropped by (1) or (2)</li>
      </ol>
      <p class="small">Fully excluded (no usable trials): <b>{len(stats['fully_excluded'])}</b>
         &middot; 3SD outliers: <b>{len(stats['outlier_pids'])}</b></p>
    </div>
    """

    # Orphan IDs - no session data at all
    orphan_rows = []
    for uid, dt, dur in ORPHAN_NO_CODE_IDS:
        orphan_rows.append(
            f'<tr><td class="uid">{html.escape(uid)}</td>'
            f'<td>{html.escape(dt)}</td><td>{html.escape(dur)}</td>'
            f'<td class="msg">No session data in our records; consider asking for a '
            f'RETURN on Prolific instead of rejecting.</td></tr>'
        )
    orphan_html = f"""
    <h2 class="section-h">Request RETURN &mdash; no session data in our records</h2>
    <p class="sub2">These Prolific IDs were flagged NO CODE but have no matching session
       in Game.csv or KeyTable.csv. They did not complete the study on our side, so the
       appropriate action on Prolific is a RETURN rather than a rejection.</p>
    <table class="orphan">
      <thead><tr><th>Prolific ID</th><th>Submitted</th><th>Duration</th><th>Suggested message</th></tr></thead>
      <tbody>{''.join(orphan_rows)}</tbody>
    </table>
    """

    return f"""<!doctype html>
<html lang="he"><head><meta charset="utf-8">
<title>KeyTable &mdash; M52 exclusions</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif;
          background:#fff; color:#1a1a1a; margin:24px; }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .sub {{ color:#666; margin-bottom:18px; font-size:13px; }}
  .stats {{ display:flex; gap:18px; margin:14px 0 20px; }}
  .stats > div {{ border:1px solid #ddd; border-radius:8px; padding:10px 16px;
                  background:#fafafa; min-width:140px; }}
  .stats .num {{ display:block; font-size:24px; font-weight:700; }}
  .stats .num.kept {{ color:#2e7d32; }}
  .stats .num.excl {{ color:#c62828; }}
  .stats .label {{ display:block; font-size:12px; color:#555; margin-top:2px; }}
  .criteria {{ border-left:4px solid #1976d2; background:#f5faff; padding:10px 16px;
               margin-bottom:20px; border-radius:4px; }}
  .criteria h2 {{ margin:0 0 6px; font-size:15px; color:#1565c0; }}
  .criteria ol {{ margin:4px 0 6px 22px; padding:0; font-size:13px; }}
  .criteria li {{ margin:2px 0; }}
  .criteria .small {{ font-size:12px; color:#666; margin:6px 0 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ padding:6px 10px; border-bottom:1px solid #eee; text-align:left;
            vertical-align:top; }}
  th {{ background:#f0f2f5; position:sticky; top:0; font-weight:600; }}
  tr:hover td {{ background:#fafbfc; }}
  tr.excluded td {{ color:#c62828; text-decoration:line-through;
                    text-decoration-color:#c62828; text-decoration-thickness:2px;
                    background:#fff5f5; }}
  tr.excluded td.reason {{ text-decoration:none; color:#b71c1c; font-weight:600; }}
  td.reason {{ color:#2e7d32; font-size:12px; }}
  td.msg {{ text-decoration:none !important; color:#333; font-size:12px;
            font-style:italic; max-width:320px; }}
  tr.excluded td.msg {{ color:#333; background:#fff; }}
  .section-h {{ margin:32px 0 6px; font-size:16px; color:#e65100; }}
  .sub2 {{ color:#666; font-size:12px; margin:0 0 10px; }}
  table.orphan {{ border:1px solid #f9a825; background:#fffbea; }}
  table.orphan th {{ background:#fff3e0; color:#e65100; }}
  table.orphan .uid {{ font-family:Menlo,Consolas,monospace; font-weight:600; }}
</style></head>
<body>
  <h1>KeyTable &mdash; M52 exclusions</h1>
  <div class="sub">Source: Downloads/Game.csv + Downloads/KeyTable.csv &middot;
                   Rule set: scripts/m52_final_composite_dv.py</div>
  {criteria_html}
  {stats_html}
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  {orphan_html}
</body></html>
"""


def main():
    print(f'[M52] Loading {GAME_CSV}')
    trials = load_trials(GAME_CSV)
    print(f'  {len(trials)} trials loaded')
    lda = load_lda_assignments()
    print(f'  {len(lda)} articles have LDA topic assignments')

    qdf = build_question_df(trials, lda)
    print(f'  Built {len(qdf)} question-level rows across '
          f'{qdf["pid"].nunique()} real-trial participants')

    excluded, reasons, messages, fully_excluded, outlier_pids = run_exclusions(qdf)

    # Print summary
    print('\n--- Fully excluded (no usable trials) ---')
    for pid in sorted(fully_excluded):
        print(f'  P{pid}: {reasons[pid]}')
    print(f'\n--- 3 SD outliers ---')
    for pid in sorted(outlier_pids):
        print(f'  P{pid}: {reasons[pid]}')

    keytable = pd.read_csv(KEYTABLE_CSV)
    # Drop test/developer rows
    keytable = keytable[~keytable['ID'].isin(IGNORED_PIDS)].reset_index(drop=True)

    # Also flag KeyTable IDs that are NOT present in Game.csv at all
    ids_with_data = set(qdf['pid'].unique())
    missing_ids = set(keytable['ID']) - ids_with_data
    for pid in missing_ids:
        reasons[int(pid)] = 'no real trials found in Game.csv'
        messages[int(pid)] = ('No session data in our records; consider asking '
                              'for a RETURN on Prolific instead of rejecting.')
        excluded.add(int(pid))

    html_out = render_html(keytable, excluded, reasons, messages, {
        'fully_excluded': fully_excluded,
        'outlier_pids': outlier_pids,
    })
    OUT_HTML.write_text(html_out, encoding='utf-8')
    print(f'\nSaved: {OUT_HTML}')
    print(f'Excluded: {len(excluded)} / {len(keytable)}')


if __name__ == '__main__':
    main()

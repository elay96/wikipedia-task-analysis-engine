#!/usr/bin/env python3
"""
M75c: Build an interactive HTML labelling tool.
Extracts movement samples + resources for a curated set of (pid, trial)
pairs and embeds them into a self-contained HTML page that lets the user
brush each path segment as 'explore' or 'exploit'. Labels are downloaded
as JSON for downstream model calibration.

Output: output/m75_label_tool.html
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'output'

GAME_CSV = DATA_DIR / 'cleaned' / 'spatial_search' / 'Game.csv'
LABELS_DIR = DATA_DIR / 'manual labeling'
HTML_OUT = OUTPUT_DIR / 'm75_label_tool.html'

# diverse mix: pid 8 trial 1 was specifically called out, the others give
# a spread of conditions and behaviours
LABEL_TRIALS = [
    (1, 1), (1, 2),
    (8, 1), (8, 2),
    (13, 1),
    (100, 1),
]

CANVAS_W = 960
CANVAS_H = 580
PAD = 20
COORD_SCALE = 3.0


def parse_time_to_seconds(s):
    if not isinstance(s, str):
        return np.nan
    p = s.split(':')
    if len(p) != 3:
        return np.nan
    try:
        return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    except (ValueError, TypeError):
        return np.nan


def build_trial_payload(df, pid, trial, x_lo, x_hi, y_lo, y_hi):
    sub = df[(df['ID'] == pid) & (df['Trial'] == trial)].copy()
    if len(sub) == 0:
        return None
    sub['t_sec'] = sub['Time'].apply(parse_time_to_seconds)
    sub = sub.sort_values('t_sec')

    samples = sub[sub['Action'] == 'Movement_sample']
    if len(samples) < 5:
        return None
    res = sub[sub['Action'] == 'Resource_found']
    map_id = sub['MapID'].dropna().iloc[0] if sub['MapID'].notna().any() else ''
    cond = sub['GameCondition'].iloc[0]

    sx = (CANVAS_W - 2 * PAD) / (x_hi - x_lo)
    sy = (CANVAS_H - 2 * PAD) / (y_hi - y_lo)
    s = min(sx, sy)

    def to_canvas(x, y):
        cx = PAD + (x - x_lo) * s
        cy = PAD + (y - y_lo) * s
        return float(cx), float(cy)

    sample_list = []
    for _, r in samples.iterrows():
        cx, cy = to_canvas(float(r['X']), float(r['Y']))
        sample_list.append({
            'x': float(r['X']), 'y': float(r['Y']),
            'cx': cx, 'cy': cy,
            't': float(r['t_sec']) if pd.notna(r['t_sec']) else None,
            'h': float(r['Heading']) if pd.notna(r['Heading']) else None,
        })

    resources = []
    for _, r in res.iterrows():
        if pd.isna(r['ResourceX']) or pd.isna(r['ResourceY']):
            continue
        wx = float(r['ResourceX']) * COORD_SCALE
        wy = float(r['ResourceY']) * COORD_SCALE
        rcx, rcy = to_canvas(wx, wy)
        resources.append({'x': wx, 'y': wy, 'cx': rcx, 'cy': rcy})

    return {
        'key': f'pid{pid}_trial{trial}',
        'pid': int(pid),
        'trial': int(trial),
        'condition': cond,
        'map_id': map_id,
        'samples': sample_list,
        'resources': resources,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>Mode Labelling Tool (timeline)</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Arial, sans-serif;
         background: #FFFFFF; color: #1a1a1a; margin: 16px; }
  h1 { margin: 0 0 6px 0; font-size: 18px; }
  .meta { color: #666; font-size: 13px; margin-bottom: 10px; line-height: 1.5; }
  #controls, #playbar { margin: 6px 0; display: flex; flex-wrap: wrap;
              gap: 8px; align-items: center; direction: ltr; }
  button { padding: 7px 14px; border: 1px solid #CCC; background: #F5F5F5;
           cursor: pointer; border-radius: 4px; font-size: 13px; }
  button:hover { background: #E0E0E0; }
  button.active-explore { background: #1976D2; color: white; border-color: #1976D2; }
  button.active-exploit { background: #C62828; color: white; border-color: #C62828; }
  button.active-off { background: #555; color: white; border-color: #555; }
  .sep { width: 1px; height: 22px; background: #DDD; margin: 0 4px; }
  #canvas { border: 1px solid #CCC; display: block;
            background: #FFFFFF; direction: ltr; }
  #timeline { width: 100%; direction: ltr; }
  .info { font-size: 13px; color: #555; margin-top: 6px; }
  .legend span { display: inline-block; padding: 2px 8px; margin-left: 8px;
                 border-radius: 3px; font-size: 12px; color: white; }
  .pill-explore { background: #1976D2; }
  .pill-exploit { background: #C62828; }
  .pill-unlabeled { background: #9E9E9E; }
  kbd { background: #EEE; border: 1px solid #BBB; border-radius: 3px;
        padding: 1px 5px; font-family: Consolas, monospace; font-size: 12px; }
</style>
</head>
<body>
<h1>Mode Labelling Tool (timeline)</h1>
<div class="meta">
  לחץ <b>Play</b> (או <kbd>Space</kbd>) - המסלול ייחשף לפי הזמן.
  בכל רגע בחר את המצב הנוכחי: <b>Explore</b> (<kbd>E</kbd>),
  <b>Exploit</b> (<kbd>X</kbd>), או <b>Off</b> (<kbd>N</kbd>) - והתיוג יוחל אוטומטית
  על הדגימות שנחשפות. ניתן לעצור, לסחוב את ה-slider, ולהמשיך מנקודה אחרת.
  בסיום - <b>Download labels</b>.
</div>

<div id="controls">
  <button id="btnPrev">&larr; Prev trial</button>
  <span id="trialLabel" style="font-weight:600; min-width: 280px;">-</span>
  <button id="btnNext">Next trial &rarr;</button>
  <span class="sep"></span>
  <button id="btnPickFolder">Pick save folder</button>
  <span id="folderStatus" style="font-size: 12px; color: #555;">no folder selected</span>
  <span class="sep"></span>
  <button id="btnExport" style="background: #2E7D32; color: white; border-color: #2E7D32;">
    Save THIS trial
  </button>
  <button id="btnExportAll">Save all labelled trials</button>
  <button id="btnClear">Clear this trial</button>
</div>

<canvas id="canvas" width="__CANVAS_W__" height="__CANVAS_H__"></canvas>

<canvas id="labelStrip" width="__CANVAS_W__" height="22"
        style="display:block; cursor:pointer; margin-top:4px; border:1px solid #CCC; direction:ltr;"></canvas>

<input type="range" id="timeline" min="0" max="0" value="0" step="1" />

<div id="playbar">
  <button id="btnPlay">&#9654; Play (Space)</button>
  <span>Speed:</span>
  <button class="speed-btn" data-speed="1">1&times;</button>
  <button class="speed-btn" data-speed="2">2&times;</button>
  <button class="speed-btn active-explore" data-speed="4">4&times;</button>
  <button class="speed-btn" data-speed="8">8&times;</button>
  <span class="sep"></span>
  <span>Current label:</span>
  <button id="btnExplore">Explore (E)</button>
  <button id="btnExploit" class="active-exploit">Exploit (X)</button>
  <button id="btnOff">Off (N)</button>
  <span class="sep"></span>
  <span id="timeDisplay" style="min-width: 130px; font-family: Consolas, monospace;">0.0s / 0.0s</span>
</div>

<div class="info">
  <span class="legend">
    <span class="pill-explore">Explore</span>
    <span class="pill-exploit">Exploit</span>
    <span class="pill-unlabeled">Unlabeled</span>
  </span>
  <span style="margin-right:16px;" id="counter"></span>
</div>

<script>
const TRIALS = __TRIALS_JSON__;
const PRELOADED_LABELS = __PRELOADED_LABELS_JSON__;
const labels = {};
TRIALS.forEach(t => {
  const pre = PRELOADED_LABELS[t.key];
  if (pre && pre.length === t.samples.length) {
    labels[t.key] = pre.slice();
  } else {
    labels[t.key] = new Array(t.samples.length).fill('unlabeled');
  }
});

let cur = 0;
let currentIdx = 0;
let labelMode = 'exploit';        // 'explore' | 'exploit' | 'off'
let playing = false;
let speed = 4;                    // playback speed multiplier
let playTimer = null;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const timeline = document.getElementById('timeline');
const labelStrip = document.getElementById('labelStrip');
const stripCtx = labelStrip.getContext('2d');

function color(label) {
  if (label === 'explore') return '#1976D2';
  if (label === 'exploit') return '#C62828';
  return '#BDBDBD';
}

function drawStar(x, y, r, fill) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const ang = -Math.PI / 2 + i * Math.PI / 5;
    const rr = (i % 2 === 0) ? r : r * 0.45;
    const px = x + rr * Math.cos(ang);
    const py = y + rr * Math.sin(ang);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}

function draw() {
  const t = TRIALS[cur];
  const lab = labels[t.key];
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Resources always visible
  t.resources.forEach(r => drawStar(r.cx, r.cy, 5, '#2E7D32'));

  // Path up to currentIdx (revealed)
  ctx.lineWidth = 2.4;
  for (let i = 0; i < currentIdx; i++) {
    ctx.strokeStyle = color(lab[i]);
    ctx.beginPath();
    ctx.moveTo(t.samples[i].cx, t.samples[i].cy);
    ctx.lineTo(t.samples[i+1].cx, t.samples[i+1].cy);
    ctx.stroke();
  }

  // Start marker + current head
  if (t.samples.length > 0) {
    const a = t.samples[0];
    ctx.fillStyle = '#1a1a1a';
    ctx.beginPath(); ctx.arc(a.cx, a.cy, 5, 0, 2*Math.PI); ctx.fill();
  }
  if (currentIdx > 0 && currentIdx < t.samples.length) {
    const head = t.samples[currentIdx];
    ctx.fillStyle = '#FFC107';
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(head.cx, head.cy, 7, 0, 2*Math.PI);
    ctx.fill(); ctx.stroke();
  }

  // Header
  document.getElementById('trialLabel').textContent =
    `[${cur+1}/${TRIALS.length}] pid ${t.pid} trial ${t.trial} | ${t.condition} | ${t.map_id}`;

  // Counter
  let nExplore = 0, nExploit = 0, nUn = 0;
  lab.forEach(l => {
    if (l === 'explore') nExplore++;
    else if (l === 'exploit') nExploit++;
    else nUn++;
  });
  document.getElementById('counter').textContent =
    `samples ${currentIdx}/${lab.length}  |  explore: ${nExplore}  exploit: ${nExploit}  unlabeled: ${nUn}`;

  // Time display
  const t0 = t.samples[0].t || 0;
  const tNow = (currentIdx < t.samples.length ? t.samples[currentIdx].t : t.samples[t.samples.length-1].t) || 0;
  const tEnd = (t.samples[t.samples.length-1].t || 0);
  document.getElementById('timeDisplay').textContent =
    `${(tNow - t0).toFixed(1)}s / ${(tEnd - t0).toFixed(1)}s`;

  drawStrip();
}

function drawStrip() {
  const t = TRIALS[cur];
  const lab = labels[t.key];
  const W = labelStrip.width, H = labelStrip.height;
  stripCtx.fillStyle = '#FFFFFF';
  stripCtx.fillRect(0, 0, W, H);
  const segW = W / lab.length;
  for (let i = 0; i < lab.length; i++) {
    stripCtx.fillStyle = color(lab[i]);
    stripCtx.fillRect(i * segW, 0, Math.max(1.0, segW + 0.5), H);
  }
  // current position marker
  const x = currentIdx * segW;
  stripCtx.fillStyle = '#FFC107';
  stripCtx.fillRect(x - 1, 0, 3, H);
  stripCtx.strokeStyle = '#1a1a1a';
  stripCtx.lineWidth = 1;
  stripCtx.strokeRect(x - 1.5, 0.5, 4, H - 1);
}

function loadTrial(i) {
  stopPlay();
  cur = (i + TRIALS.length) % TRIALS.length;
  const t = TRIALS[cur];
  const hasLabels = labels[t.key].some(l => l !== 'unlabeled');
  currentIdx = hasLabels ? t.samples.length - 1 : 0;
  timeline.max = t.samples.length - 1;
  timeline.value = currentIdx;
  draw();
}

function applyLabelAt(idx) {
  if (labelMode === 'off') return;
  const t = TRIALS[cur];
  if (idx >= 0 && idx < t.samples.length) {
    labels[t.key][idx] = labelMode;
  }
}

function tick() {
  const t = TRIALS[cur];
  if (currentIdx >= t.samples.length - 1) {
    stopPlay();
    return;
  }
  currentIdx++;
  applyLabelAt(currentIdx);
  timeline.value = currentIdx;
  draw();
  playTimer = setTimeout(tick, 100 / speed);
}

function startPlay() {
  if (playing) return;
  const t = TRIALS[cur];
  if (currentIdx >= t.samples.length - 1) currentIdx = 0;
  playing = true;
  document.getElementById('btnPlay').innerHTML = '&#10074;&#10074; Pause (Space)';
  playTimer = setTimeout(tick, 100 / speed);
}

function stopPlay() {
  playing = false;
  if (playTimer) { clearTimeout(playTimer); playTimer = null; }
  document.getElementById('btnPlay').innerHTML = '&#9654; Play (Space)';
}

function togglePlay() { playing ? stopPlay() : startPlay(); }

function setLabelMode(m) {
  labelMode = m;
  document.getElementById('btnExplore').className = (m === 'explore') ? 'active-explore' : '';
  document.getElementById('btnExploit').className = (m === 'exploit') ? 'active-exploit' : '';
  document.getElementById('btnOff').className = (m === 'off') ? 'active-off' : '';
}

function setSpeed(s) {
  speed = s;
  document.querySelectorAll('.speed-btn').forEach(b => {
    b.className = 'speed-btn' + (parseInt(b.dataset.speed) === s ? ' active-explore' : '');
  });
  if (playing) {
    if (playTimer) clearTimeout(playTimer);
    playTimer = setTimeout(tick, 100 / speed);
  }
}

document.getElementById('btnPlay').addEventListener('click', togglePlay);
document.getElementById('btnPrev').addEventListener('click', () => loadTrial(cur - 1));
document.getElementById('btnNext').addEventListener('click', () => loadTrial(cur + 1));
document.getElementById('btnExplore').addEventListener('click', () => setLabelMode('explore'));
document.getElementById('btnExploit').addEventListener('click', () => setLabelMode('exploit'));
document.getElementById('btnOff').addEventListener('click', () => setLabelMode('off'));
document.querySelectorAll('.speed-btn').forEach(b =>
  b.addEventListener('click', () => setSpeed(parseInt(b.dataset.speed))));

document.getElementById('btnClear').addEventListener('click', () => {
  const k = TRIALS[cur].key;
  labels[k] = new Array(TRIALS[cur].samples.length).fill('unlabeled');
  draw();
});

timeline.addEventListener('input', e => {
  stopPlay();
  currentIdx = parseInt(e.target.value);
  draw();
});

let stripDragging = false;
function stripSeekFromEvent(e) {
  const rect = labelStrip.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (labelStrip.width / rect.width);
  const t = TRIALS[cur];
  currentIdx = Math.max(0, Math.min(t.samples.length - 1,
                Math.floor(x / labelStrip.width * t.samples.length)));
  timeline.value = currentIdx;
  draw();
}
labelStrip.addEventListener('mousedown', e => { stopPlay(); stripDragging = true; stripSeekFromEvent(e); });
labelStrip.addEventListener('mousemove', e => { if (stripDragging) stripSeekFromEvent(e); });
labelStrip.addEventListener('mouseup', () => { stripDragging = false; });
labelStrip.addEventListener('mouseleave', () => { stripDragging = false; });

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'e' || e.key === 'E') setLabelMode('explore');
  else if (e.key === 'x' || e.key === 'X') setLabelMode('exploit');
  else if (e.key === 'n' || e.key === 'N') setLabelMode('off');
  else if (e.key === 'ArrowLeft') {
    stopPlay();
    const step = e.shiftKey ? 10 : 1;
    currentIdx = Math.max(0, currentIdx - step);
    timeline.value = currentIdx; draw();
  } else if (e.key === 'ArrowRight') {
    stopPlay();
    const step = e.shiftKey ? 10 : 1;
    currentIdx = Math.min(TRIALS[cur].samples.length - 1, currentIdx + step);
    timeline.value = currentIdx; draw();
  }
});

function buildPayloadFor(t) {
  return {
    key: t.key, pid: t.pid, trial: t.trial,
    condition: t.condition, map_id: t.map_id,
    n_samples: t.samples.length,
    labels: labels[t.key],
    exported_at: new Date().toISOString(),
  };
}

let folderHandle = null;
let serverLabelsDir = null;
const folderStatusEl = document.getElementById('folderStatus');
const IDB_NAME = 'm75-label-tool';
const IDB_STORE = 'handles';
const IDB_KEY = 'folder';

function setFolderStatus(msg, color) {
  folderStatusEl.textContent = msg;
  folderStatusEl.style.color = color || '#555';
}

async function probeServer() {
  try {
    const r = await fetch('/save_label', { method: 'GET' });
    if (!r.ok) return false;
    const j = await r.json();
    if (j && j.ok) {
      serverLabelsDir = j.labels_dir || '(server)';
      return true;
    }
  } catch (e) { /* not running through server */ }
  return false;
}

async function saveViaServer(text) {
  const r = await fetch('/save_label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: text,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok || !j.ok) throw new Error(j.error || ('http ' + r.status));
  return j.path;
}

function idbOpen() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet(key) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const r = tx.objectStore(IDB_STORE).get(key);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

async function idbPut(key, val) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put(val, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function tryRestoreFolder() {
  if (await probeServer()) {
    setFolderStatus('server save -> ' + serverLabelsDir, '#2E7D32');
    document.getElementById('btnPickFolder').style.display = 'none';
    return;
  }
  if (!('showDirectoryPicker' in window)) {
    setFolderStatus('no server, no File System API - saves will download', '#F57C00');
    return;
  }
  try {
    const h = await idbGet(IDB_KEY);
    if (!h) {
      setFolderStatus('no server / no folder (will prompt on first save)', '#F57C00');
      return;
    }
    folderHandle = h;
    const perm = await h.queryPermission({ mode: 'readwrite' });
    if (perm === 'granted') {
      setFolderStatus('folder: ' + h.name + ' (auto-restored)', '#2E7D32');
    } else {
      setFolderStatus('folder: ' + h.name + ' (click Save to authorize)', '#F57C00');
    }
  } catch (e) {
    console.warn('restore failed', e);
    setFolderStatus('no server / no folder (will prompt on first save)', '#F57C00');
  }
}

async function pickFolder() {
  if (!('showDirectoryPicker' in window)) {
    alert('This browser does not support direct file save. Use Chrome/Edge, ' +
          'and serve the page via a local HTTP server (file:// is blocked).');
    return false;
  }
  try {
    folderHandle = await window.showDirectoryPicker({
      id: 'm75-labels',
      mode: 'readwrite',
    });
    await idbPut(IDB_KEY, folderHandle);
    setFolderStatus('folder: ' + folderHandle.name, '#2E7D32');
    return true;
  } catch (e) {
    if (e.name !== 'AbortError') {
      console.warn(e);
      setFolderStatus('pick failed: ' + e.message, '#C62828');
    }
    return false;
  }
}

function downloadBlob(text, filename) {
  const blob = new Blob([text], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function saveTrial(t) {
  const text = JSON.stringify(buildPayloadFor(t), null, 2);
  const filename = 'm75_labels_' + t.key + '.json';

  if (serverLabelsDir !== null) {
    try {
      const path = await saveViaServer(text);
      setFolderStatus('saved -> ' + path, '#2E7D32');
      return true;
    } catch (e) {
      console.warn('server save failed', e);
      alert('Server save failed (' + e.message + '). Falling back.');
    }
  }

  if (!folderHandle && 'showDirectoryPicker' in window) {
    alert('Pick the data/manual labeling/ folder to save into.');
    const ok = await pickFolder();
    if (!ok) { downloadBlob(text, filename); return false; }
  }
  if (folderHandle) {
    try {
      const perm = await folderHandle.requestPermission({ mode: 'readwrite' });
      if (perm !== 'granted') throw new Error('permission denied');
      const fh = await folderHandle.getFileHandle(filename, { create: true });
      const w = await fh.createWritable();
      await w.write(text);
      await w.close();
      setFolderStatus('saved ' + filename + ' to ' + folderHandle.name, '#2E7D32');
      return true;
    } catch (e) {
      console.warn(e);
      alert('Failed to save to folder (' + e.message + '). Falling back to download.');
    }
  }
  downloadBlob(text, filename);
  return false;
}

document.getElementById('btnPickFolder').addEventListener('click', pickFolder);

document.getElementById('btnExport').addEventListener('click', () => {
  saveTrial(TRIALS[cur]);
});

document.getElementById('btnExportAll').addEventListener('click', async () => {
  const labelled = TRIALS.filter(t => labels[t.key].some(l => l !== 'unlabeled'));
  if (labelled.length === 0) { alert('No labelled trials yet.'); return; }
  if (!confirm('Save ' + labelled.length + ' files?')) return;
  for (const t of labelled) {
    await saveTrial(t);
  }
});

loadTrial(0);
tryRestoreFolder();
</script>
</body>
</html>
"""


def load_preloaded_labels():
    out = {}
    if not LABELS_DIR.exists():
        return out
    for f in sorted(LABELS_DIR.glob('m75_labels_*.json')):
        with open(f, 'r', encoding='utf-8') as fh:
            d = json.load(fh)
        out[d['key']] = d['labels']
    return out


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f'Loading {GAME_CSV}...')
    df = pd.read_csv(GAME_CSV, low_memory=False)
    df = df[df['GameCondition'].isin(['Clumpy', 'Diffuse'])].copy()

    samples_full = df[df['Action'] == 'Movement_sample']
    x_lo, x_hi = float(samples_full['X'].min()), float(samples_full['X'].max())
    y_lo, y_hi = float(samples_full['Y'].min()), float(samples_full['Y'].max())
    print(f'  global X range: ({x_lo:.0f}, {x_hi:.0f}) | Y range: ({y_lo:.0f}, {y_hi:.0f})')

    preloaded = load_preloaded_labels()
    print(f'  preloaded labels for {len(preloaded)} trials from {LABELS_DIR}')

    payload = []
    for pid, trial in LABEL_TRIALS:
        t = build_trial_payload(df, pid, trial, x_lo, x_hi, y_lo, y_hi)
        if t is None:
            print(f'  pid {pid} trial {trial}: skipped (no data)')
            continue
        pre = preloaded.get(t['key'])
        match = pre is not None and len(pre) == len(t['samples'])
        status = 'preloaded' if match else ('len mismatch' if pre is not None else 'no labels')
        print(f'  pid {pid} trial {trial} ({t["condition"]}): '
              f'{len(t["samples"])} samples, {len(t["resources"])} resources [{status}]')
        payload.append(t)

    html = (HTML_TEMPLATE
            .replace('__CANVAS_W__', str(CANVAS_W))
            .replace('__CANVAS_H__', str(CANVAS_H))
            .replace('__TRIALS_JSON__', json.dumps(payload))
            .replace('__PRELOADED_LABELS_JSON__', json.dumps(preloaded)))
    HTML_OUT.write_text(html, encoding='utf-8')
    print(f'wrote {HTML_OUT}')


if __name__ == '__main__':
    main()

"""Single-page HTML review report.

Results list on the left; on the right, the selected recording's EKG strips and
beat-to-beat intervals over the whole recording. The reviewer can set a final
label, mark AF sections on the interval chart and export everything as CSV.
"""
import base64
import json
import zlib
from datetime import datetime

import numpy as np

from .model import MODES

LEADS = {'Raw Ch 1': 'Ch1', 'Raw Ch 2': 'Ch2'}
DISPLAY_FS = 1000
LEVELS = 500


def noise_summary(analysis):
    decisions = analysis.decisions
    facts = []
    if analysis.noise_runs is not None:
        for run in analysis.noise_runs.itertuples():
            lead = 'both leads' if run.attribution == 'both' else LEADS[run.lead]
            facts.append(f'Noise on {lead} at {run.start_s:.0f}-{run.end_s:.0f} s')
    switched = int((decisions.lead == 'Ch2').sum())
    dropped = int((~decisions.kept).sum())
    if switched:
        facts.append(f'{switched} windows read from Ch2')
    if dropped:
        facts.append(f'{dropped} noisy windows left out')
    return facts + list(analysis.warnings)


def _encode(signal, fs):
    """Display signal at up to 1 kHz as delta-coded int16, zlib-compressed and base64-encoded.

    The signal is already band-limited to 100 Hz. One unit is 1/500 of a typical
    beat's height; artifacts beyond 32 typical beat heights are clipped.
    """
    step = max(1, fs // DISPLAY_FS)
    signal = signal[::step]
    fs = fs // step
    typical = np.median(np.abs(signal[:len(signal) // fs * fs]).reshape(-1, fs).max(axis=1))
    scale = float(typical) / LEVELS or 1.0
    levels = np.clip(np.round(signal / scale), -16000, 16000).astype(np.int64)
    deltas = np.diff(levels, prepend=0).astype('<i2')
    return base64.b64encode(zlib.compress(deltas.tobytes(), 9)).decode(), scale, fs


def _recording(analysis, detail):
    detail = detail or {}
    leads = {}
    for lead, name in LEADS.items():
        signal = analysis.leads.get(lead)
        if signal is None:
            continue
        data, scale, fs = _encode(signal.display, signal.fs)
        beat_time, interval = signal.rr()
        leads[name] = {'fs': fs, 'data': data, 'scale': scale,
                       'rr': [np.round(beat_time, 3).tolist(), np.round(interval, 1).tolist()]}
    runs = []
    if analysis.noise_runs is not None:
        for run in analysis.noise_runs.itertuples():
            names = list(LEADS.values()) if run.attribution == 'both' else [LEADS[run.lead]]
            runs.extend({'lead': n, 'start': run.start_s, 'end': run.end_s} for n in names)
    modes = {mode: {'score': round(analysis.scores[mode], 3), 'af': bool(analysis.called_af(mode)), 'basis': ''}
             for mode in MODES}
    for mode, override in detail.get('modes', {}).items():
        modes[mode].update(override)
    return {'name': detail.get('name', analysis.path.name), 'title': detail.get('title', ''),
            'lab': detail.get('lab', ''), 'label': detail.get('label', ''), 'info': detail.get('info', []),
            'duration': round(analysis.leads['Raw Ch 1'].duration_s, 2), 'modes': modes,
            'af_blocks': [[start, end, round(score, 3)] for start, end, score in analysis.af_like_blocks()],
            'facts': noise_summary(analysis), 'noise': runs, 'leads': leads}


def render_report(analyses, mode='balanced', details=None, note=''):
    """`details` maps str(path) to display overrides: name, title, lab, label, info, modes."""
    details = details or {}
    records = [_recording(a, details.get(str(a.path))) for a in analyses]
    thresholds = analyses[0].thresholds if analyses else {}
    header = f'{datetime.now():%d %b %Y} &middot; {len(records)} recordings'
    if note:
        header += ' &middot; ' + note
    settings = {'mode': mode, 'thresholds': {k: round(v, 3) for k, v in thresholds.items()},
                'key': f'{datetime.now():%Y%m%d%H%M%S}-{len(records)}'}
    return (PAGE.replace('__HEADER__', header)
            .replace('__SETTINGS__', json.dumps(settings))
            .replace('__DATA__', json.dumps(records, separators=(',', ':'))))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AF Screening Report</title>
<style>
html, body { margin: 0; height: 100%; overflow: hidden; background: #fff; color: #222;
  font: 13px/1.4 "Segoe UI", -apple-system, Helvetica, Arial, sans-serif; }
button, input, select { font: inherit; }
.page { display: grid; grid-template-rows: auto 1fr; height: 100vh; }
header { display: flex; gap: 20px; align-items: center; padding: 10px 20px; border-bottom: 1px solid #e6e6e6; }
header h1 { font-size: 15px; font-weight: 600; margin: 0; }
header .meta { color: #777; flex: 1; }
.modes { display: flex; border: 1px solid #ccc; border-radius: 4px; overflow: hidden; }
.modes button { border: none; background: #fff; padding: 4px 12px; cursor: pointer; color: #555; }
.modes button.on { background: #222; color: #fff; }
.export { border: 1px solid #ccc; background: #fff; border-radius: 4px; padding: 4px 12px; cursor: pointer; }
.body { display: grid; grid-template-columns: minmax(330px, 32%) 1fr; min-height: 0; }
.side { display: grid; grid-template-rows: auto 1fr; min-height: 0; border-right: 1px solid #e6e6e6; }
.filters { display: flex; gap: 8px; align-items: center; padding: 10px 12px; color: #555; }
.filters input[type=search] { flex: 1; padding: 5px 8px; border: 1px solid #ddd; border-radius: 3px; }
.list { overflow-y: auto; }
table { width: 100%; border-collapse: collapse; }
th { position: sticky; top: 0; background: #fff; text-align: left; font-weight: 500; color: #777;
  padding: 7px 10px; border-bottom: 1px solid #e6e6e6; }
td { padding: 5px 10px; border-bottom: 1px solid #f2f2f2; cursor: pointer; white-space: nowrap; }
td.num { font-variant-numeric: tabular-nums; }
tr.sel td { background: #f3f3f3; }
.af { color: #b42318; font-weight: 600; }
.no { color: #777; }
.view { display: grid; grid-template-rows: auto auto 1fr 1fr 1.3fr auto; padding: 10px 20px 8px; min-height: 0; gap: 4px; }
.title { display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }
.title b { font-size: 14px; }
.muted { color: #777; }
.info { color: #444; }
.facts a { color: #2b6cb0; cursor: pointer; text-decoration: none; }
.review { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; padding: 6px 0; border-top: 1px solid #f0f0f0;
  border-bottom: 1px solid #f0f0f0; }
.review label { cursor: pointer; }
.review input[type=text] { flex: 1; min-width: 160px; padding: 4px 6px; border: 1px solid #ddd; border-radius: 3px; }
.sections span { background: #fbe9e7; color: #8a1c13; border-radius: 3px; padding: 1px 6px; margin-right: 4px; }
.sections span b { cursor: pointer; font-weight: 400; margin-left: 4px; }
.panel { position: relative; min-height: 0; }
.panel label { position: absolute; left: 0; top: 0; color: #777; font-size: 12px; }
canvas { width: 100%; height: 100%; display: block; }
#rr { cursor: crosshair; }
.hint { color: #999; font-size: 12px; }
</style></head>
<body><div class="page">
<header><h1>AF screening report</h1><span class="meta">__HEADER__ &middot; <span id="count"></span></span>
  <div class="modes"><button data-mode="balanced">Balanced</button><button data-mode="sensitive">Sensitive</button></div>
  <button class="export" id="export">Export review (CSV)</button></header>
<div class="body">
  <div class="side">
    <div class="filters"><input type="search" id="search" placeholder="Search recording, lab, label">
      <label><input type="checkbox" id="flagged" checked> Flagged only</label></div>
    <div class="list"><table><thead><tr><th>Recording</th><th class="lb">Lab</th><th class="lb">Label</th>
      <th>Score</th><th>Model</th><th>Review</th></tr></thead><tbody id="rows"></tbody></table></div></div>
  <div class="view">
    <div><div class="title"><b id="name"></b><span class="muted" id="title"></span><span id="call"></span></div>
      <div class="info" id="info"></div><div class="facts muted" id="facts"></div></div>
    <div class="review">
      <span>Your label:</span>
      <label><input type="radio" name="label" value="AF"> AF</label>
      <label><input type="radio" name="label" value="Not AF"> Not AF</label>
      <label><input type="radio" name="label" value="Unsure"> Unsure</label>
      <span class="sections" id="sections"></span>
      <input type="text" id="note" placeholder="Note">
    </div>
    <div class="panel"><label>Ch1</label><canvas id="ch1"></canvas></div>
    <div class="panel"><label>Ch2</label><canvas id="ch2"></canvas></div>
    <div class="panel"><label>Beat-to-beat interval (ms) &middot; <span style="color:#2b6cb0">Ch1</span>
      <span style="color:#c05621">Ch2</span></label><canvas id="rr"></canvas></div>
    <div class="hint">Drag across the interval chart to mark an AF section; click it to view that part of the EKG.
      Arrow keys: up/down changes recording, left/right moves 3 s.</div>
  </div>
</div></div>
<script>
const RECORDS = __DATA__, SETTINGS = __SETTINGS__;
const SPAN = 3, PAD = 44;
const COLOR = { Ch1: '#2b6cb0', Ch2: '#c05621' }, NOISE = 'rgba(245, 180, 0, 0.18)', MARK = 'rgba(180, 35, 24, 0.14)';
const GRID = '#eeeeee', TEXT = '#888', STORE = 'af-review-' + SETTINGS.key;
let current = 0, t0 = 10, mode = SETTINGS.mode, drag = null;
let reviews = {};
try { reviews = JSON.parse(localStorage.getItem(STORE)) || {}; } catch (e) { reviews = {}; }

function review(rec) { return reviews[rec.name] || (reviews[rec.name] = { label: '', note: '', sections: [] }); }
function save() { try { localStorage.setItem(STORE, JSON.stringify(reviews)); } catch (e) {} }
function result(rec) { return rec.modes[mode]; }

async function decode(lead) {
  if (lead.samples) return;
  const bytes = Uint8Array.from(atob(lead.data), c => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  const deltas = new Int16Array(await new Response(stream).arrayBuffer());
  const y = new Float32Array(deltas.length), fs = lead.fs, mins = [], maxs = [];
  let level = 0;
  deltas.forEach((d, i) => { level += d; y[i] = level * lead.scale; });
  for (let i = 0; i + fs <= y.length; i += fs) {
    let lo = Infinity, hi = -Infinity;
    for (let j = i; j < i + fs; j++) { lo = Math.min(lo, y[j]); hi = Math.max(hi, y[j]); }
    mins.push(lo); maxs.push(hi);
  }
  const median = a => a.slice().sort((p, q) => p - q)[Math.floor(a.length / 2)];
  lead.range = [median(mins) * 1.4, median(maxs) * 1.4];
  lead.samples = y;
}

function setup(canvas) {
  const ratio = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * ratio; canvas.height = h * ratio;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '11px Segoe UI, Helvetica, Arial, sans-serif';
  return [ctx, w, h];
}

function shade(ctx, spans, color, x, a, b, top, bottom) {
  ctx.fillStyle = color;
  for (const [s0, e0] of spans) {
    const s = Math.max(s0, a), e = Math.min(e0, b);
    if (e > s) ctx.fillRect(x(s), top, x(e) - x(s), bottom - top);
  }
}

function drawStrip(id, rec, name) {
  const [ctx, w, h] = setup(document.getElementById(id));
  const lead = rec.leads[name];
  if (!lead) { ctx.fillStyle = TEXT; ctx.fillText('No usable signal', PAD, h / 2); return; }
  const y = lead.samples, fs = lead.fs, [lo, hi] = lead.range;
  const top = 16, bottom = h - 16;
  const x = t => PAD + (t - t0) / SPAN * (w - PAD - 8);
  const yy = v => bottom - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo) * (bottom - top);
  shade(ctx, review(rec).sections, MARK, x, t0, t0 + SPAN, top, bottom);
  shade(ctx, rec.noise.filter(r => r.lead === name).map(r => [r.start, r.end]), NOISE, x, t0, t0 + SPAN, top, bottom);
  ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.fillStyle = TEXT; ctx.textAlign = 'center';
  for (let t = Math.ceil(t0 * 2) / 2; t <= t0 + SPAN + 1e-9; t += 0.5) {
    ctx.beginPath(); ctx.moveTo(x(t), top); ctx.lineTo(x(t), bottom); ctx.stroke();
    ctx.fillText(t.toFixed(1) + ' s', x(t), h - 3);
  }
  ctx.strokeStyle = COLOR[name]; ctx.lineWidth = 1.1; ctx.lineJoin = 'round'; ctx.beginPath();
  const first = Math.max(0, Math.round(t0 * fs)), last = Math.min(Math.round((t0 + SPAN) * fs), y.length - 1);
  for (let i = first; i <= last; i++) {
    const px = x(i / fs), py = yy(y[i]);
    i === first ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function drawRR(rec) {
  const canvas = document.getElementById('rr'), [ctx, w, h] = setup(canvas);
  const top = 20, bottom = h - 20, end = rec.duration;
  const x = t => PAD + t / end * (w - PAD - 8), time = px => (px - PAD) / (w - PAD - 8) * end;
  const values = Object.values(rec.leads).flatMap(l => l.rr[1]);
  const sorted = values.slice().sort((a, b) => a - b);
  const lo = values.length ? Math.max(0, sorted[Math.floor(sorted.length * 0.01)] - 20) : 0;
  const hi = values.length ? sorted[Math.floor(sorted.length * 0.99)] + 20 : 200;
  const yy = v => bottom - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo) * (bottom - top);
  shade(ctx, review(rec).sections, MARK, x, 0, end, top, bottom);
  if (drag) shade(ctx, [[Math.min(drag.a, drag.b), Math.max(drag.a, drag.b)]], MARK, x, 0, end, top, bottom);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.06)'; ctx.fillRect(x(t0), top, x(t0 + SPAN) - x(t0), bottom - top);
  ctx.strokeStyle = GRID; ctx.fillStyle = TEXT; ctx.lineWidth = 1; ctx.textAlign = 'right';
  const step = (hi - lo) > 150 ? 50 : 20;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    ctx.beginPath(); ctx.moveTo(PAD, yy(v)); ctx.lineTo(w - 8, yy(v)); ctx.stroke(); ctx.fillText(v, PAD - 6, yy(v) + 4);
  }
  ctx.textAlign = 'center';
  const tick = end > 150 ? 20 : 10;
  for (let t = 0; t <= end; t += tick) ctx.fillText(t + ' s', x(t), h - 4);
  for (const [name, lead] of Object.entries(rec.leads)) {
    ctx.strokeStyle = COLOR[name]; ctx.lineWidth = 1.1; ctx.beginPath();
    let previous = -1;
    lead.rr[0].forEach((t, i) => {
      const px = x(t), py = yy(lead.rr[1][i]);
      t - previous > 1 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      previous = t;
    });
    ctx.stroke();
  }
  canvas.onmousedown = e => { drag = { a: time(e.offsetX), b: time(e.offsetX), px: e.offsetX }; };
  canvas.onmousemove = e => { if (drag) { drag.b = time(e.offsetX); drawRR(rec); } };
  canvas.onmouseup = e => {
    if (!drag) return;
    const moved = Math.abs(e.offsetX - drag.px) > 5, a = Math.min(drag.a, drag.b), b = Math.max(drag.a, drag.b);
    drag = null;
    if (moved) {
      review(rec).sections.push([Math.max(0, +a.toFixed(1)), Math.min(end, +b.toFixed(1))]);
      review(rec).sections.sort((p, q) => p[0] - q[0]);
      save(); showReview(rec); updateRow(current);
    } else {
      t0 = Math.min(Math.max(time(e.offsetX) - SPAN / 2, 0), end - SPAN);
    }
    drawTraces();
  };
  canvas.onmouseleave = () => { if (drag) { drag = null; drawRR(rec); } };
}

async function drawTraces() {
  const index = current, rec = RECORDS[index];
  await Promise.all(Object.values(rec.leads).map(decode));
  if (index !== current) return;
  drawStrip('ch1', rec, 'Ch1'); drawStrip('ch2', rec, 'Ch2'); drawRR(rec);
}

function jump(start) { t0 = Math.min(Math.max(start, 0), RECORDS[current].duration - SPAN); drawTraces(); }

function showReview(rec) {
  const r = review(rec);
  document.querySelectorAll('input[name=label]').forEach(i => { i.checked = i.value === r.label; });
  document.getElementById('note').value = r.note;
  const box = document.getElementById('sections');
  box.innerHTML = r.sections.length ? 'AF sections: ' : '<span style="background:none;color:#999">No AF sections marked</span>';
  r.sections.forEach(([a, b], k) => {
    const chip = document.createElement('span');
    chip.textContent = a.toFixed(1) + '-' + b.toFixed(1) + ' s';
    chip.style.cursor = 'pointer'; chip.onclick = () => jump(a);
    const remove = document.createElement('b'); remove.textContent = '×'; remove.title = 'Remove';
    remove.onclick = e => { e.stopPropagation(); r.sections.splice(k, 1); save(); showReview(rec); updateRow(current); drawTraces(); };
    chip.appendChild(remove); box.appendChild(chip);
  });
}

function showTitle(rec) {
  const m = result(rec);
  const call = document.getElementById('call');
  call.textContent = 'Model (' + mode + '): ' + (m.af ? 'AF' : 'Not AF') + ', score ' + m.score.toFixed(2) + (m.basis ? ' (' + m.basis + ')' : '');
  call.className = m.af ? 'af' : 'no';
  const facts = document.getElementById('facts');
  facts.textContent = '';
  if (rec.af_blocks.length) {
    facts.append('AF-like 30 s blocks: ');
    rec.af_blocks.forEach(([a, b, s], k) => {
      const link = document.createElement('a');
      link.textContent = a + '-' + b + ' s (' + s.toFixed(2) + ')';
      link.onclick = () => jump(a);
      facts.append(link, k < rec.af_blocks.length - 1 ? ', ' : '');
    });
    facts.append(' · ');
  }
  facts.append(rec.facts.length ? rec.facts.join(' · ') : 'No noise detected');
}

function select(i) {
  current = i;
  const rec = RECORDS[current];
  t0 = rec.af_blocks.length && mode === 'sensitive' ? rec.af_blocks[0][0] : 10;
  document.querySelectorAll('#rows tr').forEach(row => row.classList.toggle('sel', +row.dataset.i === current));
  const row = document.querySelector('#rows tr[data-i="' + current + '"]');
  if (row) row.scrollIntoView({ block: 'nearest' });
  document.getElementById('name').textContent = rec.name;
  document.getElementById('title').textContent = rec.title;
  document.getElementById('info').textContent = rec.info.join(' · ');
  showTitle(rec); showReview(rec); drawTraces();
}

function updateRow(i) {
  const rec = RECORDS[i], m = result(rec), row = document.querySelector('#rows tr[data-i="' + i + '"]');
  row.cells[3].textContent = m.score.toFixed(2);
  row.cells[4].textContent = m.af ? 'AF' : 'Not AF';
  row.cells[4].className = m.af ? 'af' : 'no';
  const r = review(rec);
  row.cells[5].textContent = r.label + (r.sections.length ? ' (' + r.sections.length + ')' : '');
}

function filter() {
  const words = document.getElementById('search').value.toLowerCase().split(/\s+/).filter(Boolean);
  const flaggedOnly = document.getElementById('flagged').checked;
  let shown = 0, flagged = 0;
  document.querySelectorAll('#rows tr').forEach(row => {
    const rec = RECORDS[+row.dataset.i], af = result(rec).af;
    flagged += af;
    const show = words.every(w => row.dataset.text.includes(w)) && (!flaggedOnly || af);
    row.style.display = show ? '' : 'none';
    shown += show;
  });
  document.getElementById('count').textContent = flagged + ' flagged as AF in ' + mode + ' mode (threshold ' +
    SETTINGS.thresholds[mode].toFixed(2) + ')';
  return shown;
}

function setMode(next) {
  mode = next;
  document.querySelectorAll('.modes button').forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
  RECORDS.forEach((_, i) => updateRow(i));
  filter();
  const shown = visible();
  select(shown.includes(current) || !shown.length ? current : shown[0]);
}

function visible() {
  return [...document.querySelectorAll('#rows tr')].filter(r => r.style.display !== 'none').map(r => +r.dataset.i);
}

function step(delta) {
  const shown = visible(), k = shown.indexOf(current);
  if (shown.length) select(shown[Math.max(0, Math.min(shown.length - 1, (k < 0 ? 0 : k + delta)))]);
}

function exportCsv() {
  const quote = v => '"' + String(v).replace(/"/g, '""') + '"';
  const lines = [['recording', 'original_file', 'lab', 'label', 'balanced_score', 'balanced_call', 'sensitive_score',
    'sensitive_call', 'reviewer_label', 'af_sections_s', 'note'].join(',')];
  RECORDS.forEach(rec => {
    const r = review(rec), b = rec.modes.balanced, s = rec.modes.sensitive;
    lines.push([rec.name, rec.title, rec.lab, rec.label, b.score, b.af ? 'AF' : 'Not AF', s.score, s.af ? 'AF' : 'Not AF',
      r.label, r.sections.map(([a, e]) => a + '-' + e).join('; '), r.note].map(quote).join(','));
  });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
  link.download = 'af_review.csv'; link.click();
}

const rows = document.getElementById('rows');
RECORDS.forEach((rec, i) => {
  const row = document.createElement('tr');
  row.dataset.i = i;
  row.innerHTML = '<td></td><td class="lb"></td><td class="lb"></td><td class="num"></td><td></td><td></td>';
  row.cells[0].textContent = rec.name; row.cells[1].textContent = rec.lab; row.cells[2].textContent = rec.label;
  row.dataset.text = [rec.name, rec.title, rec.lab, rec.label, ...rec.info].join(' ').toLowerCase();
  row.onclick = () => select(i);
  rows.appendChild(row);
});
if (!RECORDS.some(r => r.label)) document.querySelectorAll('.lb').forEach(c => c.style.display = 'none');
document.querySelectorAll('.modes button').forEach(b => b.onclick = () => setMode(b.dataset.mode));
document.getElementById('search').addEventListener('input', filter);
document.getElementById('flagged').addEventListener('change', () => { filter(); const s = visible(); if (s.length) select(s[0]); });
document.getElementById('export').onclick = exportCsv;
document.querySelectorAll('input[name=label]').forEach(input => input.onchange = () => {
  review(RECORDS[current]).label = input.value; save(); updateRow(current);
});
document.getElementById('note').oninput = e => { review(RECORDS[current]).note = e.target.value; save(); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'radio' && e.target.type !== 'checkbox') return;
  if (e.key === 'ArrowDown') { step(1); e.preventDefault(); }
  if (e.key === 'ArrowUp') { step(-1); e.preventDefault(); }
  if (e.key === 'ArrowRight') { jump(t0 + SPAN); e.preventDefault(); }
  if (e.key === 'ArrowLeft') { jump(t0 - SPAN); e.preventDefault(); }
});
window.addEventListener('resize', drawTraces);
if (RECORDS.length) {
  setMode(mode);
  const shown = visible();
  select(shown.length ? shown[0] : 0);
}
</script></body></html>"""

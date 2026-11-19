"""
ChordSensePro — Interactive Test UI
====================================
Web-based testing interface with:
  • Audio waveform visualization (WaveSurfer.js)
  • Real-time chord annotations synced to playback
  • Testing table with ✓/✗ marking per segment + feedback column
  • Auto-loads latest model from AI/results, songs from AI/songs

Usage:
    cd AI
    python test_ui.py                       # defaults
    python test_ui.py --port 5050           # custom port
    python test_ui.py --model results/model_v5_fullsong_p3.onnx
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR))

from flask import Flask, Response, jsonify, render_template_string, send_file, request, stream_with_context

from src.serve import (
    MIN_DURATION_MS,
    HarmonicSeparator,
    ONNXPredictor,
    _decode_raw_waveform,
    detect_solo_clean_audio,
    infer_audio,
    load_audio_bytes,
)
import test_worker as worker

# ---------------------------------------------------------------------------
# Model resolution — prefer the newest v5 fullsong, then fall back
# ---------------------------------------------------------------------------

RESULTS_DIR = AI_DIR / "results"
SONGS_DIR = AI_DIR / "songs"
CLASSES_JSON = str(AI_DIR.parent / "DE" / "data" / "export" / "v3" / "classes.json")

MODEL_PRIORITY = [
    "model_v5_1_fullsong_p3.onnx",
    "model_v5_fullsong_p3.onnx",
    "model_v5_p3.onnx",
    "model_v5_20s_p2.onnx",
    "model_v4_5s_p2.onnx",
    "model_v3_5s_p2.onnx",
    "model_p2.onnx",
]


def resolve_latest_model() -> str:
    for name in MODEL_PRIORITY:
        p = RESULTS_DIR / name
        if p.exists():
            return str(p)
    # Fallback: pick any .onnx in results/ sorted by mtime descending
    onnx_files = sorted(RESULTS_DIR.glob("*.onnx"), key=lambda f: f.stat().st_mtime, reverse=True)
    if onnx_files:
        return str(onnx_files[0])
    raise FileNotFoundError("No .onnx model found in AI/results/")


def list_songs() -> list[dict]:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    songs = []
    for f in sorted(SONGS_DIR.iterdir()):
        if f.suffix.lower() in exts:
            songs.append({"name": f.stem, "filename": f.name, "path": str(f)})
    return songs


# ---------------------------------------------------------------------------
# Global state (loaded once)
# ---------------------------------------------------------------------------

predictor: ONNXPredictor | None = None
_separator_ref: list = [None]      # mutable container for Demucs singleton
_separator_lock = threading.Lock() # serialize Demucs (not thread-safe)
_model_path: str = ""


def get_predictor() -> ONNXPredictor:
    global predictor
    if predictor is None:
        raise RuntimeError("Model not loaded")
    return predictor


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ChordSensePro — Test UI</title>
  <script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
  <script src="https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.min.js"></script>
  <script src="https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.min.js"></script>
  <style>
    :root {
      --bg: #0f0f13;
      --surface: #1a1a24;
      --surface2: #22223a;
      --accent: #6c63ff;
      --accent2: #ff6b9d;
      --green: #22c55e;
      --red: #ef4444;
      --yellow: #eab308;
      --text: #e4e4ef;
      --text-dim: #8888a4;
      --border: #2d2d44;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      min-height: 100vh;
    }

    /* Header */
    .header {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      padding: 20px 32px;
      display: flex;
      align-items: center;
      gap: 16px;
      border-bottom: 1px solid var(--border);
    }
    .header h1 {
      font-size: 1.5rem;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 700;
    }
    .header .model-badge {
      background: var(--surface2);
      border: 1px solid var(--border);
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 0.75rem;
      color: var(--text-dim);
    }

    /* Layout */
    .container { max-width: 1400px; margin: 0 auto; padding: 24px; }

    /* Song selector */
    .controls-bar {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }
    .controls-bar select, .controls-bar button {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px 16px;
      border-radius: 8px;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all .2s;
    }
    .controls-bar select:hover, .controls-bar button:hover {
      border-color: var(--accent);
    }
    .controls-bar button.primary {
      background: var(--accent);
      border-color: var(--accent);
      font-weight: 600;
    }
    .controls-bar button.primary:hover { opacity: 0.85; }
    .controls-bar button.primary:disabled {
      opacity: 0.5; cursor: not-allowed;
    }

    /* Waveform */
    .waveform-wrapper {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 20px;
      position: relative;
    }
    #waveform { min-height: 128px; }
    #timeline { margin-top: 4px; }

    /* Chord Timeline */
    .chord-timeline-wrap {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 16px;
      margin-bottom: 20px;
    }
    .chord-timeline-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-dim);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .chord-timeline-scroll {
      overflow-x: auto;
      overflow-y: hidden;
      border-radius: 6px;
    }
    .chord-timeline-scroll::-webkit-scrollbar { height: 5px; }
    .chord-timeline-scroll::-webkit-scrollbar-track { background: var(--bg); border-radius: 3px; }
    .chord-timeline-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    .chord-timeline {
      position: relative;
      height: 52px;
      background: var(--bg);
      /* min-width set via JS = n_chords * MIN_BLOCK_PX */
    }
    .chord-block {
      position: absolute;
      top: 0;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.78rem;
      font-weight: 700;
      color: rgba(255,255,255,0.95);
      border-right: 1px solid rgba(0,0,0,0.25);
      cursor: pointer;
      transition: filter 0.1s;
      user-select: none;
      box-sizing: border-box;
    }
    .chord-block:hover { filter: brightness(1.3); }
    .chord-block.active-block {
      outline: 2px solid #fff;
      z-index: 2;
      filter: brightness(1.4);
    }
    .chord-block span {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding: 0 5px;
      text-shadow: 0 1px 3px rgba(0,0,0,0.7);
    }
    /* Playhead on chord timeline */
    .chord-playhead {
      position: absolute;
      top: 0;
      width: 2px;
      height: 100%;
      background: #fff;
      pointer-events: none;
      z-index: 10;
      box-shadow: 0 0 6px rgba(255,255,255,0.7);
    }

    /* Current chord display */
    .current-chord-display {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 24px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .chord-big {
      font-size: 3.5rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      min-width: 180px;
      text-align: center;
      transition: all 0.15s;
    }
    .chord-meta {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 0.85rem;
      color: var(--text-dim);
    }
    .chord-meta span.conf {
      font-size: 1.2rem;
      color: var(--green);
      font-weight: 600;
    }

    /* Player controls */
    .player-controls {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      margin-bottom: 20px;
    }
    .player-controls button {
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text);
      width: 44px;
      height: 44px;
      border-radius: 50%;
      font-size: 1.1rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all .2s;
    }
    .player-controls button:hover { border-color: var(--accent); background: var(--accent); }
    .player-controls button.play-btn {
      width: 56px;
      height: 56px;
      background: var(--accent);
      border-color: var(--accent);
      font-size: 1.3rem;
    }
    .player-controls .time-display {
      font-family: 'Consolas', monospace;
      font-size: 0.95rem;
      color: var(--text-dim);
      min-width: 120px;
      text-align: center;
    }

    /* Stats */
    .stats-row {
      display: flex;
      gap: 16px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 20px;
      flex: 1;
      min-width: 150px;
    }
    .stat-card .label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }
    .stat-card .value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
    .stat-card .value.green { color: var(--green); }
    .stat-card .value.red { color: var(--red); }
    .stat-card .value.yellow { color: var(--yellow); }

    /* Results table */
    .table-wrapper {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }
    .table-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
    }
    .table-header h3 { font-size: 1rem; font-weight: 600; }
    .table-header .actions { display: flex; gap: 8px; }
    .table-header .actions button {
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all .2s;
    }
    .table-header .actions button:hover { border-color: var(--accent); }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th {
      background: var(--surface2);
      padding: 10px 14px;
      text-align: left;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-dim);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    td {
      padding: 10px 14px;
      border-bottom: 1px solid rgba(45,45,68,0.5);
      font-size: 0.85rem;
      vertical-align: middle;
    }
    tr:hover { background: rgba(108,99,255,0.05); }
    tr.active-row { background: rgba(108,99,255,0.15) !important; }
    tr.correct td { border-left: 3px solid var(--green); }
    tr.incorrect td { border-left: 3px solid var(--red); }
    .chord-cell {
      font-weight: 700;
      font-size: 1rem;
      color: var(--accent);
    }
    .conf-bar {
      display: inline-block;
      height: 6px;
      border-radius: 3px;
      background: var(--accent);
      min-width: 4px;
    }
    .verdict-btns { display: flex; gap: 6px; }
    .verdict-btns button {
      width: 32px;
      height: 32px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--surface2);
      cursor: pointer;
      font-size: 1rem;
      transition: all .15s;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .verdict-btns button.v-correct { color: var(--green); }
    .verdict-btns button.v-incorrect { color: var(--red); }
    .verdict-btns button.selected { border-width: 2px; transform: scale(1.1); }
    .verdict-btns button.v-correct.selected { background: rgba(34,197,94,0.2); border-color: var(--green); }
    .verdict-btns button.v-incorrect.selected { background: rgba(239,68,68,0.2); border-color: var(--red); }
    .feedback-input {
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      width: 100%;
      min-width: 160px;
      transition: border-color .2s;
    }
    .feedback-input:focus { outline: none; border-color: var(--accent); }
    .seek-btn {
      background: none;
      border: none;
      color: var(--text-dim);
      cursor: pointer;
      font-size: 0.8rem;
      padding: 4px 8px;
      border-radius: 4px;
      transition: all .15s;
    }
    .seek-btn:hover { color: var(--accent); background: rgba(108,99,255,0.1); }

    /* Scrollable tbody */
    .table-scroll {
      max-height: 500px;
      overflow-y: auto;
    }
    .table-scroll::-webkit-scrollbar { width: 6px; }
    .table-scroll::-webkit-scrollbar-track { background: var(--surface); }
    .table-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* Loading overlay */
    .loading-overlay {
      position: fixed;
      inset: 0;
      background: rgba(15,15,19,0.85);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 100;
    }
    .loading-overlay.hidden { display: none; }
    .spinner {
      width: 48px;
      height: 48px;
      border: 4px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-text { margin-top: 16px; color: var(--text-dim); font-size: 0.9rem; }
    .progress-bar-wrap {
      margin-top: 16px;
      width: 320px;
      height: 6px;
      background: var(--border);
      border-radius: 3px;
      overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      border-radius: 3px;
      transition: width 0.4s ease;
      width: 0%;
    }

    /* Responsive */
    @media (max-width: 768px) {
      .container { padding: 12px; }
      .chord-big { font-size: 2.5rem; min-width: 120px; }
      td, th { padding: 8px 10px; }
    }
  </style>
</head>
<body>

<div class="header">
  <h1>🎵 ChordSensePro Test UI</h1>
  <span class="model-badge" id="modelBadge">{{ model_name }}</span>
</div>

<div class="container">
  <!-- Song selector + analyze -->
  <div class="controls-bar">
    <select id="songSelect">
      <option value="">-- Chọn bài hát --</option>
      {% for s in songs %}
      <option value="{{ s.filename }}">{{ s.name }}</option>
      {% endfor %}
    </select>
    <button class="primary" id="analyzeBtn" disabled>🔍 Phân tích</button>
    <label style="display:flex;align-items:center;gap:6px;font-size:0.85rem;color:var(--text-dim)"
           title="Tách vocal/drums bằng Demucs trước khi nhận diện. Lần đầu tải ~315MB từ HuggingFace — nên tắt để test nhanh.">
      <input type="checkbox" id="demucsToggle"> Demucs (tách nhạc cụ — chậm, cần HF download lần đầu)
    </label>
  </div>

  <!-- Waveform -->
  <div class="waveform-wrapper">
    <div id="waveform"></div>
    <div id="timeline"></div>
  </div>

  <!-- Chord Timeline -->
  <div class="chord-timeline-wrap" id="chordTimelineWrap" style="display:none">
    <div class="chord-timeline-label">
      <span>🎹 Hợp âm theo thời gian</span>
      <span id="chordTimelineInfo" style="font-size:0.7rem;color:var(--text-dim)"></span>
    </div>
    <div class="chord-timeline-scroll" id="chordTimelineScroll">
      <div class="chord-timeline" id="chordTimeline">
        <div class="chord-playhead" id="chordPlayhead" style="left:0px"></div>
      </div>
    </div>
  </div>

  <!-- Current chord -->
  <div class="current-chord-display">
    <div class="chord-big" id="currentChord">—</div>
    <div class="chord-meta">
      <span>Thời gian: <span id="currentTime">0:00.0</span></span>
      <span>Confidence: <span id="currentConf" class="conf">—</span></span>
      <span>Đoạn: <span id="currentSegIdx">—</span></span>
    </div>
  </div>

  <!-- Player controls -->
  <div class="player-controls">
    <button onclick="skipBack()" title="Lùi 5s">⏪</button>
    <button class="play-btn" id="playBtn" onclick="togglePlay()">▶</button>
    <button onclick="skipForward()" title="Tiến 5s">⏩</button>
    <span class="time-display" id="timeDisplay">0:00 / 0:00</span>
    <input type="range" id="volumeSlider" min="0" max="1" step="0.05" value="0.8"
           style="width:100px;accent-color:var(--accent)" title="Âm lượng">
  </div>

  <!-- Stats -->
  <div class="stats-row" id="statsRow" style="display:none">
    <div class="stat-card">
      <div class="label">Tổng đoạn</div>
      <div class="value" id="statTotal">0</div>
    </div>
    <div class="stat-card">
      <div class="label">Đã đánh giá</div>
      <div class="value yellow" id="statReviewed">0</div>
    </div>
    <div class="stat-card">
      <div class="label">Đúng ✓</div>
      <div class="value green" id="statCorrect">0</div>
    </div>
    <div class="stat-card">
      <div class="label">Sai ✗</div>
      <div class="value red" id="statIncorrect">0</div>
    </div>
    <div class="stat-card">
      <div class="label">Accuracy</div>
      <div class="value green" id="statAccuracy">—</div>
    </div>
    <div class="stat-card">
      <div class="label">Thời gian inference</div>
      <div class="value" id="statInferTime">—</div>
    </div>
  </div>

  <!-- Results table -->
  <div class="table-wrapper" id="tableWrapper" style="display:none">
    <div class="table-header">
      <h3>📋 Kết quả nhận diện hợp âm</h3>
      <div class="actions">
        <button onclick="markAllCorrect()">✓ Đánh dấu tất cả Đúng</button>
        <button onclick="clearAllVerdicts()">↺ Reset</button>
        <button onclick="exportResults()">📥 Xuất JSON</button>
      </div>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Bắt đầu</th>
            <th>Kết thúc</th>
            <th>Hợp âm</th>
            <th>Confidence</th>
            <th>Đúng / Sai</th>
            <th>Hợp âm đúng</th>
            <th>Feedback</th>
            <th></th>
          </tr>
        </thead>
        <tbody id="resultsBody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- Loading overlay -->
<div class="loading-overlay hidden" id="loadingOverlay">
  <div class="spinner"></div>
  <div class="loading-text" id="loadingText">Đang phân tích...</div>
  <div class="progress-bar-wrap">
    <div class="progress-bar-fill" id="progressBarFill"></div>
  </div>
</div>

<script>
  // ========== State ==========
  let wavesurfer = null;
  let chords = [];
  let verdicts = {};      // idx -> 'correct' | 'incorrect'
  let corrections = {};   // idx -> corrected chord string
  let feedbacks = {};     // idx -> string
  let currentSegIdx = -1;
  let regionsPlugin = null;

  // ========== WaveSurfer setup ==========
  function initWavesurfer() {
    if (wavesurfer) {
      wavesurfer.destroy();
    }

    regionsPlugin = WaveSurfer.Regions.create();
    const timelinePlugin = WaveSurfer.Timeline.create({
      height: 20,
      timeInterval: 5,
      primaryLabelInterval: 10,
      style: { fontSize: '11px', color: '#8888a4' },
    });

    wavesurfer = WaveSurfer.create({
      container: '#waveform',
      waveColor: 'rgba(108, 99, 255, 0.4)',
      progressColor: 'rgba(108, 99, 255, 0.85)',
      cursorColor: '#ff6b9d',
      cursorWidth: 2,
      height: 128,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      responsive: true,
      plugins: [regionsPlugin, timelinePlugin],
    });

    wavesurfer.on('timeupdate', onTimeUpdate);
    wavesurfer.on('play', () => { document.getElementById('playBtn').textContent = '⏸'; });
    wavesurfer.on('pause', () => { document.getElementById('playBtn').textContent = '▶'; });

    document.getElementById('volumeSlider').addEventListener('input', (e) => {
      wavesurfer.setVolume(parseFloat(e.target.value));
    });
  }

  // ========== Time / chord tracking ==========
  function formatTime(s) {
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1);
    return `${m}:${sec.padStart(4, '0')}`;
  }

  function onTimeUpdate(currentTime) {
    const dur = wavesurfer.getDuration();
    document.getElementById('timeDisplay').textContent =
      `${formatTime(currentTime)} / ${formatTime(dur)}`;
    document.getElementById('currentTime').textContent = formatTime(currentTime);

    const ms = currentTime * 1000;

    // Update chord timeline playhead
    updateChordTimelinePlayhead(ms);

    let segIdx = -1;
    for (let i = 0; i < chords.length; i++) {
      if (ms >= chords[i].start_ms && ms < chords[i].end_ms) {
        segIdx = i;
        break;
      }
    }

    if (segIdx !== currentSegIdx) {
      currentSegIdx = segIdx;
      highlightRow(segIdx);
      if (segIdx >= 0) {
        const c = chords[segIdx];
        document.getElementById('currentChord').textContent = c.chord;
        document.getElementById('currentConf').textContent = (c.confidence * 100).toFixed(1) + '%';
        document.getElementById('currentSegIdx').textContent = `${segIdx + 1} / ${chords.length}`;
      } else {
        document.getElementById('currentChord').textContent = '—';
        document.getElementById('currentConf').textContent = '—';
        document.getElementById('currentSegIdx').textContent = '—';
      }
    }
  }

  function highlightRow(idx) {
    document.querySelectorAll('#resultsBody tr').forEach((row, i) => {
      row.classList.toggle('active-row', i === idx);
    });

  }

  // ========== Controls ==========
  function togglePlay() {
    if (wavesurfer) wavesurfer.playPause();
  }

  function skipBack() {
    if (wavesurfer) {
      const t = Math.max(0, wavesurfer.getCurrentTime() - 5);
      wavesurfer.seekTo(t / wavesurfer.getDuration());
    }
  }

  function skipForward() {
    if (wavesurfer) {
      const dur = wavesurfer.getDuration();
      const t = Math.min(dur, wavesurfer.getCurrentTime() + 5);
      wavesurfer.seekTo(t / dur);
    }
  }

  function seekTo(ms) {
    if (wavesurfer) {
      const dur = wavesurfer.getDuration();
      wavesurfer.seekTo((ms / 1000) / dur);
    }
  }

  // ========== Song selector ==========
  document.getElementById('songSelect').addEventListener('change', (e) => {
    document.getElementById('analyzeBtn').disabled = !e.target.value;
  });

  // ========== Analyze: start job → stream SSE ==========
  document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const filename = document.getElementById('songSelect').value;
    if (!filename) return;

    const useDemucs = document.getElementById('demucsToggle').checked;
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    const progressFill = document.getElementById('progressBarFill');

    overlay.classList.remove('hidden');
    loadingText.textContent = '🚀 Đang khởi động job...';
    progressFill.style.width = '2%';

    chords = [];
    verdicts = {};
    corrections = {};
    feedbacks = {};
    currentSegIdx = -1;

    // 1. POST to start background job → get job_id immediately
    let jobId;
    try {
      const r = await fetch(`/analyze/${filename}?demucs=${useDemucs}`);
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();
      jobId = j.job_id;
    } catch (err) {
      overlay.classList.add('hidden');
      alert('Lỗi khởi động: ' + err.message);
      return;
    }

    // 2. Open SSE stream — server is free to handle other requests while worker runs
    const es = new EventSource(`/analyze/stream/${jobId}`);

    es.addEventListener('progress', (e) => {
      const d = JSON.parse(e.data);
      loadingText.textContent = d.msg;
      progressFill.style.width = (d.pct || 0) + '%';
    });

    es.addEventListener('done', async (e) => {
      es.close();
      const data = JSON.parse(e.data);

      chords = data.chords;
      progressFill.style.width = '95%';
      loadingText.textContent = '📊 Đang hiển thị waveform...';

      initWavesurfer();
      const durationSec = data.duration_ms / 1000;
      const peaks = (data.peaks && data.peaks.length > 0) ? [data.peaks] : null;

      await new Promise((resolve, reject) => {
        wavesurfer.on('ready', resolve);
        wavesurfer.on('error', reject);
        if (peaks) {
          wavesurfer.load(`/audio/${filename}`, peaks, durationSec);
        } else {
          wavesurfer.load(`/audio/${filename}`);
        }
      });

      drawChordRegions();
      drawChordTimeline();
      renderTable();

      document.getElementById('statsRow').style.display = 'flex';
      document.getElementById('tableWrapper').style.display = 'block';
      document.getElementById('statTotal').textContent = chords.length;
      document.getElementById('statInferTime').textContent = data.elapsed_ms + 'ms';
      updateStats();

      progressFill.style.width = '100%';
      overlay.classList.add('hidden');
    });

    es.addEventListener('error', (e) => {
      es.close();
      overlay.classList.add('hidden');
      try {
        const d = JSON.parse(e.data);
        alert('Lỗi: ' + (d.error || 'Unknown'));
      } catch {
        alert('Lỗi khi phân tích. Xem console để biết thêm.');
      }
      console.error('SSE error', e);
    });
  });

  // ========== Draw chord regions on waveform (clean, no text) ==========
  const REGION_COLORS = [
    'rgba(108,99,255,0.18)',
    'rgba(255,107,157,0.18)',
    'rgba(34,197,94,0.18)',
    'rgba(234,179,8,0.18)',
    'rgba(99,220,255,0.18)',
    'rgba(255,159,64,0.18)',
  ];

  // Richer block colors for chord timeline
  const BLOCK_COLORS = [
    '#4f46e5','#7c3aed','#db2777','#059669','#d97706',
    '#0891b2','#dc2626','#16a34a','#9333ea','#c2410c',
    '#0284c7','#65a30d','#b45309','#6d28d9','#047857',
  ];

  function drawChordRegions() {
    if (!regionsPlugin) return;
    regionsPlugin.clearRegions();

    chords.forEach((c, i) => {
      regionsPlugin.addRegion({
        start: c.start_ms / 1000,
        end: c.end_ms / 1000,
        color: REGION_COLORS[i % REGION_COLORS.length],
        drag: false,
        resize: false,
      });
    });
  }

  // ========== Chord timeline strip (scrollable, pixel-based) ==========
  let totalDurationMs = 0;
  const MIN_BLOCK_PX = 72;   // minimum px per chord block so text is readable
  let timelineWidthPx = 0;

  function drawChordTimeline() {
    const scrollEl = document.getElementById('chordTimelineScroll');
    const container = document.getElementById('chordTimeline');
    const playhead = document.getElementById('chordPlayhead');
    container.innerHTML = '';
    container.appendChild(playhead);

    if (!chords.length) return;
    totalDurationMs = chords[chords.length - 1].end_ms;

    // Timeline width: at least container visible width, at least N * MIN_BLOCK_PX
    const viewW = scrollEl.clientWidth || 800;
    timelineWidthPx = Math.max(viewW, chords.length * MIN_BLOCK_PX);
    container.style.width = timelineWidthPx + 'px';

    chords.forEach((c, i) => {
      const leftPx  = (c.start_ms / totalDurationMs) * timelineWidthPx;
      const widthPx = Math.max(2, ((c.end_ms - c.start_ms) / totalDurationMs) * timelineWidthPx);
      const color   = BLOCK_COLORS[i % BLOCK_COLORS.length];

      const block = document.createElement('div');
      block.className = 'chord-block';
      block.style.left   = leftPx + 'px';
      block.style.width  = widthPx + 'px';
      block.style.background = color;
      block.dataset.idx = i;
      block.title = `${c.chord}  ${(c.start_ms/1000).toFixed(2)}s – ${(c.end_ms/1000).toFixed(2)}s  conf: ${(c.confidence*100).toFixed(1)}%`;

      const label = document.createElement('span');
      label.textContent = c.chord;
      block.appendChild(label);

      block.addEventListener('click', () => seekTo(c.start_ms));
      container.appendChild(block);
    });

    document.getElementById('chordTimelineInfo').textContent =
      `${chords.length} hợp âm · scroll ngang để xem toàn bộ`;
    document.getElementById('chordTimelineWrap').style.display = 'block';
  }

  function updateChordTimelinePlayhead(currentMs) {
    if (!totalDurationMs || !timelineWidthPx) return;
    const px = Math.min(timelineWidthPx, (currentMs / totalDurationMs) * timelineWidthPx);
    const playhead = document.getElementById('chordPlayhead');
    if (playhead) playhead.style.left = px + 'px';

    // Highlight active block + auto-scroll to keep it visible
    const scrollEl = document.getElementById('chordTimelineScroll');
    document.querySelectorAll('.chord-block').forEach((el, i) => {
      const c = chords[i];
      const isActive = currentMs >= c.start_ms && currentMs < c.end_ms;
      el.classList.toggle('active-block', isActive);
      if (isActive && scrollEl) {
        const elLeft = parseFloat(el.style.left);
        const viewLeft = scrollEl.scrollLeft;
        const viewRight = viewLeft + scrollEl.clientWidth;
        // Scroll to keep active block centered, but only if out of view
        if (elLeft < viewLeft + 40 || elLeft > viewRight - 100) {
          scrollEl.scrollTo({ left: Math.max(0, elLeft - scrollEl.clientWidth / 2), behavior: 'smooth' });
        }
      }
    });
  }

  // ========== Render table ==========
  function renderTable() {
    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';

    chords.forEach((c, i) => {
      const tr = document.createElement('tr');
      tr.dataset.idx = i;

      const startS = (c.start_ms / 1000).toFixed(2);
      const endS = (c.end_ms / 1000).toFixed(2);
      const confPct = (c.confidence * 100).toFixed(1);
      const barWidth = Math.max(4, Math.round(c.confidence * 100));

      tr.innerHTML = `
        <td style="color:var(--text-dim)">${i + 1}</td>
        <td><button class="seek-btn" onclick="seekTo(${c.start_ms})">${startS}s</button></td>
        <td>${endS}s</td>
        <td class="chord-cell">${c.chord}</td>
        <td>
          <span style="margin-right:6px">${confPct}%</span>
          <span class="conf-bar" style="width:${barWidth}px;opacity:${0.4 + c.confidence * 0.6}"></span>
        </td>
        <td>
          <div class="verdict-btns">
            <button class="v-correct" onclick="setVerdict(${i},'correct')" title="Đúng">✓</button>
            <button class="v-incorrect" onclick="setVerdict(${i},'incorrect')" title="Sai">✗</button>
          </div>
        </td>
        <td>
          <input class="feedback-input correction-input" placeholder="Hợp âm đúng..."
                 data-idx="${i}" oninput="setCorrection(${i}, this.value)"
                 style="display:none;max-width:120px">
        </td>
        <td>
          <input class="feedback-input" placeholder="Ghi chú..."
                 data-idx="${i}" oninput="setFeedback(${i}, this.value)">
        </td>
        <td><button class="seek-btn" onclick="seekTo(${c.start_ms})">▶</button></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // ========== Verdicts ==========
  function setVerdict(idx, verdict) {
    if (verdicts[idx] === verdict) {
      delete verdicts[idx];
    } else {
      verdicts[idx] = verdict;
    }

    const row = document.querySelectorAll('#resultsBody tr')[idx];
    if (!row) return;

    row.classList.remove('correct', 'incorrect');
    const btns = row.querySelectorAll('.verdict-btns button');
    btns.forEach(b => b.classList.remove('selected'));

    const correctionInput = row.querySelector('.correction-input');

    if (verdicts[idx] === 'correct') {
      row.classList.add('correct');
      btns[0].classList.add('selected');
      if (correctionInput) correctionInput.style.display = 'none';
    } else if (verdicts[idx] === 'incorrect') {
      row.classList.add('incorrect');
      btns[1].classList.add('selected');
      if (correctionInput) correctionInput.style.display = 'block';
    } else {
      if (correctionInput) correctionInput.style.display = 'none';
    }

    updateStats();
  }

  function setCorrection(idx, value) {
    corrections[idx] = value;
  }

  function setFeedback(idx, value) {
    feedbacks[idx] = value;
  }

  function markAllCorrect() {
    chords.forEach((_, i) => setVerdict(i, 'correct'));
  }

  function clearAllVerdicts() {
    chords.forEach((_, i) => {
      delete verdicts[i];
      delete corrections[i];
      delete feedbacks[i];
    });
    renderTable();
    updateStats();
  }

  function updateStats() {
    const total = chords.length;
    const reviewed = Object.keys(verdicts).length;
    const correct = Object.values(verdicts).filter(v => v === 'correct').length;
    const incorrect = Object.values(verdicts).filter(v => v === 'incorrect').length;

    document.getElementById('statReviewed').textContent = `${reviewed} / ${total}`;
    document.getElementById('statCorrect').textContent = correct;
    document.getElementById('statIncorrect').textContent = incorrect;

    if (reviewed > 0) {
      const acc = ((correct / reviewed) * 100).toFixed(1);
      document.getElementById('statAccuracy').textContent = acc + '%';
    } else {
      document.getElementById('statAccuracy').textContent = '—';
    }
  }

  // ========== Export ==========
  function exportResults() {
    const songName = document.getElementById('songSelect').value;
    const result = {
      song: songName,
      model: document.getElementById('modelBadge').textContent,
      timestamp: new Date().toISOString(),
      total_segments: chords.length,
      reviewed: Object.keys(verdicts).length,
      correct: Object.values(verdicts).filter(v => v === 'correct').length,
      incorrect: Object.values(verdicts).filter(v => v === 'incorrect').length,
      segments: chords.map((c, i) => ({
        ...c,
        verdict: verdicts[i] || null,
        correction: corrections[i] || null,
        feedback: feedbacks[i] || null,
      })),
    };

    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `test_results_${songName.replace(/\.[^.]+$/, '')}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ========== Keyboard shortcuts ==========
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    switch (e.code) {
      case 'Space':
        e.preventDefault();
        togglePlay();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        skipBack();
        break;
      case 'ArrowRight':
        e.preventDefault();
        skipForward();
        break;
      case 'KeyC':
        if (currentSegIdx >= 0) setVerdict(currentSegIdx, 'correct');
        break;
      case 'KeyX':
        if (currentSegIdx >= 0) setVerdict(currentSegIdx, 'incorrect');
        break;
    }
  });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    songs = list_songs()
    model_name = Path(_model_path).name if _model_path else "unknown"
    return render_template_string(HTML_TEMPLATE, songs=songs, model_name=model_name)


@app.route("/audio/<filename>")
def serve_audio(filename: str):
    """Serve audio file from AI/songs/."""
    filepath = SONGS_DIR / filename
    if not filepath.exists():
        return "File not found", 404
    # Determine mimetype
    ext = filepath.suffix.lower()
    mimes = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}
    return send_file(str(filepath), mimetype=mimes.get(ext, "application/octet-stream"))


def _compute_peaks(filepath: Path, n_peaks: int = 1000) -> list[float]:
    """Compute waveform peak values for fast WaveSurfer rendering (no full client download needed)."""
    import struct, wave, math

    ext = filepath.suffix.lower()
    samples = None
    sr = 1

    # Fast path for WAV
    if ext == ".wav":
        try:
            with wave.open(str(filepath), "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
            fmt = {1: "b", 2: "h", 4: "i"}.get(sampwidth, "h")
            arr = struct.unpack(f"{len(raw)//sampwidth}{fmt}", raw)
            if n_channels > 1:
                arr = arr[::n_channels]  # take left channel
            max_val = float(2 ** (sampwidth * 8 - 1))
            samples = [x / max_val for x in arr]
        except Exception:
            samples = None

    if samples is None:
        try:
            import librosa
            y, sr = librosa.load(str(filepath), sr=None, mono=True)
            samples = y.tolist()
        except Exception:
            return [0.0] * n_peaks

    # Downsample to n_peaks by taking max abs in each bucket
    total = len(samples)
    if total == 0:
        return [0.0] * n_peaks
    bucket = max(1, total // n_peaks)
    peaks = []
    for i in range(n_peaks):
        start = i * bucket
        end = min(start + bucket, total)
        chunk = samples[start:end]
        if chunk:
            peaks.append(max(abs(x) for x in chunk))
        else:
            peaks.append(0.0)
    return peaks


@app.route("/peaks/<filename>")
def get_peaks(filename: str):
    """Return pre-computed waveform peaks for fast WaveSurfer rendering."""
    filepath = SONGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    n = int(request.args.get("n", 1000))
    peaks = _compute_peaks(filepath, n_peaks=n)
    return jsonify({"peaks": peaks, "length": len(peaks)})


@app.route("/analyze/<filename>")
def analyze_start(filename: str):
    """Start a background inference job, return job_id immediately."""
    filepath = SONGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    use_demucs = request.args.get("demucs", "true").lower() in ("true", "1", "yes")

    job = worker.create_job(filename)
    t = threading.Thread(
        target=worker.run_inference_job,
        args=(job, filepath, use_demucs, get_predictor(), _separator_ref, _separator_lock),
        daemon=True,
    )
    t.start()
    job.thread = t

    return jsonify({"job_id": job.id})


@app.route("/analyze/stream/<job_id>")
def analyze_stream(job_id: str):
    """SSE endpoint: stream progress from background job queue."""
    job = worker.get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        while True:
            try:
                msg = job.q.get(timeout=15)  # check every 15s
            except Exception:
                # Send heartbeat to keep SSE connection alive (browser/proxy might drop idle connections)
                yield ": heartbeat\n\n"
                # Check if worker thread is still alive
                if job.thread and not job.thread.is_alive():
                    yield f"event: error\ndata: {json.dumps({'error': 'Worker exited unexpectedly'})}\n\n"
                    return
                continue

            if "error" in msg:
                yield f"event: error\ndata: {json.dumps(msg)}\n\n"
                return
            elif msg.get("done"):
                yield f"event: done\ndata: {json.dumps(msg['result'])}\n\n"
                return
            else:
                yield f"event: progress\ndata: {json.dumps(msg)}\n\n"

    return Response(
        generate(),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global predictor, _model_path

    parser = argparse.ArgumentParser(description="ChordSensePro — Interactive Test UI")
    parser.add_argument("--model", default=None, help="Path to .onnx model (default: latest in results/)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--classes-json", default=CLASSES_JSON)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _model_path = args.model or resolve_latest_model()
    print(f"🎵 ChordSensePro Test UI")
    print(f"   Model : {_model_path}")
    print(f"   Songs : {SONGS_DIR}")
    print(f"   Classes: {args.classes_json}")
    print(f"   Server : http://{args.host}:{args.port}")
    print()

    print("Loading ONNX model...")
    predictor = ONNXPredictor(_model_path, args.classes_json)
    if predictor.is_v5:
        print("   ✓ Model v5 (2-stage Full-Song Context)")
    elif predictor.is_frame_level:
        print("   ✓ Model v4 (frame-level)")
    else:
        print("   ✓ Model (clip-level)")

    print(f"\nSongs available: {len(list_songs())}")
    for s in list_songs():
        print(f"   • {s['name']}")

    print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
    print("   Shortcuts: Space=Play/Pause, ←→=Seek, C=Correct, X=Incorrect\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

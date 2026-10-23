"""Listen to training segments next to their labels — builds a small static HTML page.

What it does
  1. Picks N random segments from ``training_samples`` that match your filters.
  2. Cuts each 2 s segment out of its ``.npy`` file and writes it as a WAV.
  3. Writes ``index.html``: one card per segment with an audio player, the raw label (Harte),
     the decoded chord, source/song/time, and ✅ / ❌ / ❓ buttons + a note box.
  4. Verdicts are kept in the browser (localStorage); the "Download CSV" button saves them.

Run inside the container (DB + .npy files live there). Output is bind-mounted to the host:
    docker exec -w /opt/airflow chordsense_airflow env PYTHONPATH=/opt/airflow \\
        python scripts/listen_samples.py --source hooktheory --ext "#11" --n 30
    # → DE/data/export/listen/<name>/index.html   (open it in a browser)

Examples
    --ext nat9 --quality dominant            # dominant 9th chords
    --quality sus4 --split test --n 50       # sus4 chords from the test split
    --label-like "b13" --source jaah         # raw label contains b13
    --context                                # also add the previous / next segment of each song

Filters (all optional, combined with AND)
    --source S        hooktheory | jaah | guitarset | local | ...
    --split S         train | val | test
    --status LIST     comma list of label_status (default: ok)  e.g. ok,partial
    --root R          C, C#, D ... (value or enum name: Cs)
    --quality Q       major | minor | dominant | diminished (or dim) | sus2 | sus4
    --ext E           none | b5 | #5 | maj7 | min7 | dom7 | add9 | b9 | nat9 | #9 | nat11 | #11 | nat13 | b13 | alt
    --label-like TXT  case-insensitive substring of the raw Harte label
Other
    --n N             number of segments (default 30)
    --seed TXT        same seed + same filters = same sample (default "0")
    --name NAME       output sub-folder name (default: timestamp)
    --out DIR         output root (default /data/export/listen)
    --raw-gain        do NOT peak-normalise the WAVs (default: normalise to -1 dBFS so quiet stems are audible;
                      the training data itself is NOT changed)

Caveat: the label is the chord at the *midpoint* of the segment. A chord change inside the 2 s window,
or a faint extension note (#11, b13...), can be hard to hear — treat this as a spot check, not ground truth.
"""
from __future__ import annotations
import argparse
import html
import json
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/opt/airflow")

import numpy as np  # noqa: E402

from src.shared.domain.value_objects import ChordExtension, ChordLabel, ChordQuality, RootNote  # noqa: E402
from src.shared.infrastructure.postgres.client import get_connection                            # noqa: E402
from src.data_processing.application.pipeline_handlers.sample_writer_handler import (            # noqa: E402
    training_dir,
)

SAMPLE_RATE = 24_000
ROOTS, QUALITIES, EXTS = list(RootNote), list(ChordQuality), list(ChordExtension)
QUALITY_ALIASES = {"dim": "diminished"}


def resolve(options: list, text: str, kind: str) -> int:
    """Map a user string (enum value or enum name) to its class id."""
    text = QUALITY_ALIASES.get(text, text) if kind == "quality" else text
    for i, m in enumerate(options):
        if text in (m.name, m.value):
            return i
    valid = ", ".join(m.value for m in options)
    raise SystemExit(f"Unknown {kind} {text!r}. Valid: {valid}")


def decode(root_id, quality_id, ext_id) -> str:
    if root_id is None:
        return "—"
    root = ROOTS[root_id]
    if quality_id is None or ext_id is None:
        q = QUALITIES[quality_id].value if quality_id is not None else "?"
        e = EXTS[ext_id].value if ext_id is not None else "?"
        return f"{root.value} · {q} · {e}  (partial label)"
    q, e = QUALITIES[quality_id], EXTS[ext_id]
    label = ChordLabel(root, q, e)
    warn = "" if label.is_valid else "  ⚠ invalid combo"
    return f"{label.display_name}  ({root.value} · {q.value} · {e.value}){warn}"


def write_wav(path: Path, seg: np.ndarray, normalise: bool) -> None:
    a = np.asarray(seg, dtype=np.int16)
    if normalise:
        peak = float(np.abs(a.astype(np.float32)).max())
        if peak > 0:
            a = (a.astype(np.float32) * (0.89 * 32767 / peak)).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(a.tobytes())


def query_samples(args) -> tuple[int, list[dict]]:
    params: list[Any] = [[s.strip() for s in args.status.split(",")]]
    where = ["label_status = ANY(%s)"]
    for column, value in (("source_type", args.source), ("split", args.split)):
        if value:
            where.append(f"{column} = %s")
            params.append(value)
    if args.root:
        where.append("root_id = %s")
        params.append(resolve(ROOTS, args.root, "root"))
    if args.quality:
        where.append("quality_id = %s")
        params.append(resolve(QUALITIES, args.quality, "quality"))
    if args.ext:
        where.append("ext_id = %s")
        params.append(resolve(EXTS, args.ext, "extension"))
    if args.label_like:
        where.append("label_raw ILIKE %s")
        params.append(f"%{args.label_like}%")
    clause = " AND ".join(where)
    cols = ("id, song_id, source_type, source_url, segment_idx, start_ms, end_ms, audio_path, audio_idx, "
            "label_raw, label_status, root_id, quality_id, ext_id, split")
    keys = [c.strip() for c in cols.split(",")]
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM training_samples WHERE {clause}", params)
        total = (cur.fetchone() or (0,))[0]
        cur.execute(
            f"SELECT {cols} FROM training_samples WHERE {clause} ORDER BY md5(id::text || %s) LIMIT %s",
            params + [args.seed, args.n],
        )
        rows = [dict(zip(keys, r)) for r in cur.fetchall()]
        if args.context:
            for r in rows:
                cur.execute(
                    "SELECT segment_idx, start_ms, audio_path, audio_idx, label_raw FROM training_samples "
                    "WHERE song_id = %s AND segment_idx IN (%s, %s) ORDER BY segment_idx",
                    (r["song_id"], r["segment_idx"] - 1, r["segment_idx"] + 1),
                )
                r["neighbours"] = [
                    dict(zip(("segment_idx", "start_ms", "audio_path", "audio_idx", "label_raw"), n))
                    for n in cur.fetchall()
                ]
    return total, rows


def render_card(r: dict, n: int, base: Path, audio_dir: Path, normalise: bool, root: Path) -> str:
    def clip(row: dict, tag: str) -> str:
        arr = np.load(root / row["audio_path"], mmap_mode="r")
        name = f"{tag}.wav"
        write_wav(audio_dir / name, arr[row["audio_idx"]], normalise)
        return f"audio/{name}"

    main = clip(r, f"{r['id']}")
    nbr = ""
    for nb in r.get("neighbours", []):
        rel = "previous" if nb["segment_idx"] < r["segment_idx"] else "next"
        src = clip(nb, f"{r['id']}_{rel}")
        nbr += (f'<div class="nb"><span>{rel} · seg {nb["segment_idx"]} · {nb["start_ms"] / 1000:.1f}s · '
                f'<code>{html.escape(nb["label_raw"])}</code></span>'
                f'<audio controls preload="none" src="{src}"></audio></div>')
    link = html.escape(r["source_url"].split("#")[0])
    return f"""
<article class="card" data-id="{r['id']}" data-song="{html.escape(r['song_id'])}" data-seg="{r['segment_idx']}"
         data-label="{html.escape(r['label_raw'])}" data-source="{html.escape(r['source_type'])}" data-split="{r['split']}">
  <header><span class="num">#{n}</span>
    <span class="decoded">{html.escape(decode(r['root_id'], r['quality_id'], r['ext_id']))}</span></header>
  <div class="meta"><code>{html.escape(r['label_raw'])}</code> · {r['label_status']} · {r['source_type']} · {r['split']}
    · seg {r['segment_idx']} · {r['start_ms'] / 1000:.1f}–{r['end_ms'] / 1000:.1f}s
    · <a href="{link}" target="_blank" rel="noopener">source</a></div>
  <audio controls preload="none" src="{main}"></audio>
  {nbr}
  <div class="actions">
    <button data-v="ok" title="label matches what I hear">✅ đúng</button>
    <button data-v="bad" title="label does not match">❌ sai</button>
    <button data-v="unsure" title="cannot tell">❓ không chắc</button>
    <input class="note" placeholder="ghi chú (vd: đổi hợp âm giữa chừng, nghe không rõ #11)">
  </div>
</article>"""


PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChordSense — listen &amp; verify</title>
<style>
:root{color-scheme:dark;--bg:#0f1117;--card:#181b24;--line:#2a2f3d;--fg:#e8eaf0;--mut:#8b93a7;--acc:#7c9cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}
.top{position:sticky;top:0;z-index:5;background:#0f1117f2;backdrop-filter:blur(6px);border-bottom:1px solid var(--line);
 padding:12px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0}.filters{color:var(--mut);font-size:13px}.stats{margin-left:auto;font-size:13px}
button{background:#232838;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}.dl{background:var(--acc);color:#0b1020;border:0;font-weight:600}
main{max-width:860px;margin:18px auto;padding:0 16px;display:grid;gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;border-left:4px solid var(--line)}
.card.ok{border-left-color:#3ecf8e}.card.bad{border-left-color:#ff6b6b}.card.unsure{border-left-color:#f5b942}
header{display:flex;gap:12px;align-items:baseline}.num{color:var(--mut)}.decoded{font-size:20px;font-weight:600}
.meta{color:var(--mut);font-size:13px;margin:4px 0 8px}code{background:#232838;padding:1px 6px;border-radius:5px;color:#c9d3ff}
a{color:var(--acc)}audio{width:100%;height:36px}.nb{margin-top:8px;font-size:12px;color:var(--mut)}
.actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}.actions .note{flex:1;min-width:200px;background:#10131b;
 border:1px solid var(--line);border-radius:8px;color:var(--fg);padding:6px 10px;font:inherit}
.card.ok [data-v=ok],.card.bad [data-v=bad],.card.unsure [data-v=unsure]{border-color:var(--acc);background:#2b3350}
</style></head><body>
<div class="top"><h1>🎧 Listen &amp; verify</h1><span class="filters">__FILTERS__</span>
<span class="stats" id="stats"></span><button class="dl" id="dl">Download CSV</button></div>
<main>__CARDS__</main>
<script>
const KEY="listen:__NAME__", cards=[...document.querySelectorAll(".card")];
let st=JSON.parse(localStorage.getItem(KEY)||"{}");
function paint(){let c={ok:0,bad:0,unsure:0};
 cards.forEach(el=>{const s=st[el.dataset.id]||{};el.classList.remove("ok","bad","unsure");
  if(s.v){el.classList.add(s.v);c[s.v]++}el.querySelector(".note").value=s.note||""});
 document.getElementById("stats").textContent=`✅ ${c.ok}  ❌ ${c.bad}  ❓ ${c.unsure}  / ${cards.length}`}
function save(){localStorage.setItem(KEY,JSON.stringify(st));paint()}
cards.forEach(el=>{const id=el.dataset.id;
 el.querySelectorAll("[data-v]").forEach(b=>b.onclick=()=>{st[id]=Object.assign(st[id]||{},{v:b.dataset.v});save()});
 el.querySelector(".note").oninput=e=>{st[id]=Object.assign(st[id]||{},{note:e.target.value});localStorage.setItem(KEY,JSON.stringify(st))}});
document.getElementById("dl").onclick=()=>{const q=s=>`"${String(s??"").replace(/"/g,'""')}"`;
 const rows=[["id","song_id","segment_idx","source","split","label_raw","verdict","note"]];
 cards.forEach(el=>{const s=st[el.dataset.id]||{};rows.push([el.dataset.id,el.dataset.song,el.dataset.seg,el.dataset.source,el.dataset.split,el.dataset.label,s.v||"",s.note||""])});
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\\ufeff"+rows.map(r=>r.map(q).join(",")).join("\\n")],{type:"text/csv"}));
 a.download="listen___NAME__.csv";a.click()};
paint();
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="Build an HTML page to listen to labelled training segments",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source")
    p.add_argument("--split", choices=["train", "val", "test"])
    p.add_argument("--status", default="ok")
    p.add_argument("--root")
    p.add_argument("--quality")
    p.add_argument("--ext")
    p.add_argument("--label-like", dest="label_like")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", default="0")
    p.add_argument("--context", action="store_true", help="add previous/next segment players")
    p.add_argument("--name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--out", default="/data/export/listen")
    p.add_argument("--raw-gain", action="store_true")
    args = p.parse_args()

    total, rows = query_samples(args)
    if not rows:
        raise SystemExit("No segment matches these filters.")

    out = Path(args.out) / args.name
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    root = training_dir()
    cards = "".join(
        render_card(r, i, out, audio_dir, not args.raw_gain, root) for i, r in enumerate(rows, 1)
    )
    shown = {k: v for k, v in vars(args).items()
             if k in ("source", "split", "status", "root", "quality", "ext", "label_like") and v}
    filters = html.escape(f"{json.dumps(shown, ensure_ascii=False)} — {len(rows)} of {total} matching segments")
    (out / "index.html").write_text(
        PAGE.replace("__FILTERS__", filters).replace("__CARDS__", cards).replace("__NAME__", args.name),
        encoding="utf-8",
    )
    print(f"{len(rows)} of {total} matching segments → {out / 'index.html'}")
    print("On the host: DE/data/export/listen/" + args.name + "/index.html")


if __name__ == "__main__":
    main()

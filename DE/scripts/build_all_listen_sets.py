"""Batch-build listening sample sets for all chord categories in ChordSense Pro.

Creates:
  - 22 dedicated sample sets in DE/data/export/listen/<category_id>/
  - Master Dashboard in DE/data/export/listen/index.html with live localStorage progress tracking
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/opt/airflow")

import numpy as np

from scripts import listen_samples
from src.data_processing.application.pipeline_handlers.sample_writer_handler import training_dir

CATEGORIES = [
    # 1. QUALITIES (6 Tính chất cơ bản)
    {
        "id": "qual_major",
        "name": "Major Triad (Trưởng cơ bản)",
        "desc": "Tam hợp âm trưởng thuần (C, G, D, F...). Không có nốt mở rộng (1-3-5).",
        "group": "qualities",
        "group_name": "6 Tính chất cơ bản (Qualities)",
        "quality": "major",
        "ext": "none",
        "n": 20,
    },
    {
        "id": "qual_minor",
        "name": "Minor Triad (Thứ cơ bản)",
        "desc": "Tam hợp âm thứ thuần (Am, Em, Dm, Bm...). Không có nốt mở rộng (1-b3-5).",
        "group": "qualities",
        "group_name": "6 Tính chất cơ bản (Qualities)",
        "quality": "minor",
        "ext": "none",
        "n": 20,
    },
    {
        "id": "qual_dim",
        "name": "Diminished / Half-dim (Giảm / Bán giảm)",
        "desc": "Hợp âm giảm (Bdim, F#dim, m7b5, dim7). Âm hưởng căng thẳng, kịch tính.",
        "group": "qualities",
        "group_name": "6 Tính chất cơ bản (Qualities)",
        "quality": "dim",
        "n": 20,
    },
    {
        "id": "qual_sus2",
        "name": "Sus2 (Treo bậc 2)",
        "desc": "Hợp âm treo 2 (Csus2, Dsus2...). Nốt 1-2-5, sáng và thoáng.",
        "group": "qualities",
        "group_name": "6 Tính chất cơ bản (Qualities)",
        "quality": "sus2",
        "n": 20,
    },
    {
        "id": "qual_sus4",
        "name": "Sus4 (Treo bậc 4)",
        "desc": "Hợp âm treo 4 (Csus4, Gsus4...). Nốt 1-4-5, có xu hướng hút về trưởng.",
        "group": "qualities",
        "group_name": "6 Tính chất cơ bản (Qualities)",
        "quality": "sus4",
        "n": 20,
    },
    {
        "id": "qual_dominant",
        "name": "Dominant 7th (Hợp âm Bảy át)",
        "desc": "Hợp âm 7 át cơ bản (G7, C7, E7, A7...). Nốt 1-3-5-b7, dẫn về chủ âm.",
        "group": "qualities",
        "group_name": "6 Tính chất cơ bản (Qualities)",
        "quality": "dominant",
        "ext": "dom7",
        "n": 20,
    },

    # 2. STANDARD EXTENSIONS (Các bậc mở rộng phổ biến)
    {
        "id": "ext_maj7",
        "name": "Major 7th (7 Trưởng)",
        "desc": "Hợp âm 7 trưởng (Cmaj7, Fmaj7, Gmaj7...). Âm hưởng êm đềm, jazz/pop.",
        "group": "std_ext",
        "group_name": "Bậc mở rộng phổ biến (Standard Extensions)",
        "ext": "maj7",
        "n": 20,
    },
    {
        "id": "ext_min7",
        "name": "Minor 7th (7 Thứ)",
        "desc": "Hợp âm 7 thứ (Am7, Dm7, Em7...). Phổ biến trong Ballad, Pop, R&B.",
        "group": "std_ext",
        "group_name": "Bậc mở rộng phổ biến (Standard Extensions)",
        "ext": "min7",
        "n": 20,
    },
    {
        "id": "ext_dom7",
        "name": "Dominant 7th Extension (Bậc 7 át mở rộng)",
        "desc": "Các hợp âm mang bậc 7 át (gồm cả 7sus2, 7sus4, dim7).",
        "group": "std_ext",
        "group_name": "Bậc mở rộng phổ biến (Standard Extensions)",
        "ext": "dom7",
        "n": 20,
    },
    {
        "id": "ext_add9",
        "name": "Add9 (Thêm bậc 9)",
        "desc": "Hợp âm trưởng/thứ thêm nốt 9 (Cadd9, Gadd9...). Không có nốt 7.",
        "group": "std_ext",
        "group_name": "Bậc mở rộng phổ biến (Standard Extensions)",
        "ext": "add9",
        "n": 20,
    },
    {
        "id": "ext_nat9",
        "name": "Natural 9 (Hợp âm 9 đầy đủ)",
        "desc": "Hợp âm 9 (C9, D9, Cmaj9, Dm9...). Có nốt 7 + nốt 9.",
        "group": "std_ext",
        "group_name": "Bậc mở rộng phổ biến (Standard Extensions)",
        "ext": "nat9",
        "n": 20,
    },
    {
        "id": "ext_nat11",
        "name": "Natural 11 (Bậc 11)",
        "desc": "Hợp âm 11 (Dm11, Am11, C11...). Thường gặp ở hợp âm thứ.",
        "group": "std_ext",
        "group_name": "Bậc mở rộng phổ biến (Standard Extensions)",
        "ext": "nat11",
        "n": 20,
    },
    {
        "id": "ext_nat13",
        "name": "Natural 13 (Bậc 13)",
        "desc": "Hợp âm 13 (C13, G13, Fmaj13...). Âm sắc jazz phong phú, lộng lẫy.",
        "group": "std_ext",
        "group_name": "Bậc mở rộng phổ biến (Standard Extensions)",
        "ext": "nat13",
        "n": 20,
    },

    # 3. ALTERED & RARE TENSIONS (Biến đổi & nốt màu hiếm)
    {
        "id": "ext_b9",
        "name": "Flat 9 (Giảm 9 / b9)",
        "desc": "Hợp âm 7b9 (C7b9, G7b9...). Sức căng hút về át mạnh, âm hưởng cổ điển/jazz.",
        "group": "rare_ext",
        "group_name": "Nốt màu & Hợp âm biến đổi (Rare & Altered)",
        "ext": "b9",
        "n": 20,
    },
    {
        "id": "ext_sharp9",
        "name": "Sharp 9 (Tăng 9 / #9)",
        "desc": "Hợp âm 7#9 (E7#9 'Hendrix chord'...). Âm hưởng blues/rock/fusion độc đáo.",
        "group": "rare_ext",
        "group_name": "Nốt màu & Hợp âm biến đổi (Rare & Altered)",
        "ext": "#9",
        "n": 20,
    },
    {
        "id": "ext_sharp11",
        "name": "Sharp 11 (Tăng 11 / #11)",
        "desc": "Hợp âm 7#11, maj7#11. Âm hưởng Lydian bay bổng, bí ẩn.",
        "group": "rare_ext",
        "group_name": "Nốt màu & Hợp âm biến đổi (Rare & Altered)",
        "ext": "#11",
        "n": 20,
    },
    {
        "id": "ext_b13",
        "name": "Flat 13 (Giảm 13 / b13)",
        "desc": "Hợp âm 7b13 (C7b13...). Màu sắc hòa âm altered cổ điển.",
        "group": "rare_ext",
        "group_name": "Nốt màu & Hợp âm biến đổi (Rare & Altered)",
        "ext": "b13",
        "n": 20,
    },
    {
        "id": "ext_b5",
        "name": "Flat 5 (Giảm 5 / b5)",
        "desc": "Hợp âm 7b5 (C7b5...). Quãng tritone đặc trưng.",
        "group": "rare_ext",
        "group_name": "Nốt màu & Hợp âm biến đổi (Rare & Altered)",
        "ext": "b5",
        "n": 20,
    },
    {
        "id": "ext_sharp5",
        "name": "Sharp 5 / Augmented (Tăng 5 / #5)",
        "desc": "Hợp âm tăng (Caug, G7#5...). Nốt bậc 5 nâng nửa cung.",
        "group": "rare_ext",
        "group_name": "Nốt màu & Hợp âm biến đổi (Rare & Altered)",
        "ext": "#5",
        "n": 20,
    },
    {
        "id": "ext_alt",
        "name": "Altered (Hợp âm biến đổi alt)",
        "desc": "Hợp âm 7alt (thang âm altered jazz). Chứa đồng thời b9/#9 và b5/#5.",
        "group": "rare_ext",
        "group_name": "Nốt màu & Hợp âm biến đổi (Rare & Altered)",
        "ext": "alt",
        "n": 20,
    },

    # 4. SPECIAL SOURCE SPOT-CHECKS (Kiểm tra nguồn dữ liệu)
    {
        "id": "src_jaah",
        "name": "JAAH (Jazz Recordings Check)",
        "desc": "Mẫu ngẫu nhiên từ JAAH để nghe kiểm tra độ lệch canh thời gian.",
        "group": "sources",
        "group_name": "Kiểm tra theo nguồn dữ liệu (Sources Spot-check)",
        "source": "jaah",
        "n": 30,
    },
    {
        "id": "src_piano",
        "name": "Piano Local (Domain Check)",
        "desc": "Mẫu piano thuần (domain cốt lõi của ChordSense).",
        "group": "sources",
        "group_name": "Kiểm tra theo nguồn dữ liệu (Sources Spot-check)",
        "source": "local",
        "n": 20,
    },
]

DASHBOARD_HTML = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ChordSense Pro — Verification Hub</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0d13;
      --card: #151822;
      --card-hover: #1c202e;
      --border: #262b3d;
      --fg: #eef1f8;
      --mut: #8f98af;
      --acc: #6c8dff;
      --acc-glow: rgba(108, 141, 255, 0.15);
      --success: #34d399;
      --danger: #f87171;
      --warning: #fbbf24;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--fg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      padding: 32px 20px 60px;
      line-height: 1.5;
    }
    .container {
      max-width: 1100px;
      margin: 0 auto;
    }
    header {
      margin-bottom: 36px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
    }
    .badge-top {
      display: inline-block;
      padding: 4px 10px;
      background: var(--acc-glow);
      color: var(--acc);
      border: 1px solid rgba(108, 141, 255, 0.3);
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.5px;
      margin-bottom: 12px;
      text-transform: uppercase;
    }
    h1 {
      font-size: 28px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 8px;
    }
    p.subtitle {
      color: var(--mut);
      font-size: 15px;
      max-width: 720px;
    }
    .stats-bar {
      display: flex;
      gap: 20px;
      margin-top: 20px;
      flex-wrap: wrap;
    }
    .stat-pill {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 16px;
      font-size: 13px;
    }
    .stat-pill strong {
      color: #fff;
      font-size: 16px;
      margin-right: 4px;
    }
    
    .section-title {
      font-size: 18px;
      font-weight: 600;
      color: #fff;
      margin: 36px 0 16px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .section-title::before {
      content: "";
      width: 4px;
      height: 18px;
      background: var(--acc);
      border-radius: 2px;
      display: inline-block;
    }
    
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      text-decoration: none;
      color: inherit;
      display: flex;
      flex-direction: column;
      transition: all 0.2s ease;
      position: relative;
    }
    .card:hover {
      background: var(--card-hover);
      border-color: var(--acc);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .card-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 10px;
    }
    .card-title {
      font-size: 16px;
      font-weight: 600;
      color: #fff;
    }
    .card-count {
      font-size: 12px;
      background: #232838;
      color: var(--mut);
      padding: 2px 8px;
      border-radius: 6px;
      font-weight: 500;
    }
    .card-desc {
      font-size: 13px;
      color: var(--mut);
      flex: 1;
      margin-bottom: 16px;
      line-height: 1.45;
    }
    .card-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      padding-top: 12px;
      font-size: 12px;
    }
    .progress-text {
      color: var(--mut);
      font-weight: 500;
    }
    .progress-text.done {
      color: var(--success);
    }
    .open-btn {
      color: var(--acc);
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .card:hover .open-btn {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge-top">ChordSense Pro • Data Quality Hub</div>
      <h1>🎧 Verification Hub — Kiểm thử thính giác toàn bộ hợp âm</h1>
      <p class="subtitle">
        Tập trung toàn bộ 22 danh mục hợp âm (6 Qualities, 14 Extensions, và 2 tập kiểm tra JAAH/Piano).
        Mở từng danh mục để nghe đoạn audio 2s, xem nhãn Harte và bấm xác thực (✅ Đúng / ❌ Sai / ❓ Không chắc).
      </p>
      <div class="stats-bar">
        <div class="stat-pill"><strong id="total-cats">22</strong> Danh mục</div>
        <div class="stat-pill"><strong id="total-samples">450+</strong> Đoạn âm thanh</div>
        <div class="stat-pill"><strong id="total-verified">0</strong> Đã xác thực</div>
      </div>
    </header>

    __SECTIONS__

  </div>

  <script>
    // Live update status from localStorage
    let totalVerified = 0;
    document.querySelectorAll(".card").forEach(card => {
      const id = card.dataset.id;
      const count = parseInt(card.dataset.count, 10);
      const key = "listen:" + id;
      try {
        const data = JSON.parse(localStorage.getItem(key) || "{}");
        const voted = Object.keys(data).filter(k => data[k].v).length;
        const pText = card.querySelector(".progress-text");
        if (voted > 0) {
          totalVerified += voted;
          let ok = 0, bad = 0, unsure = 0;
          Object.values(data).forEach(s => {
            if (s.v === 'ok') ok++;
            else if (s.v === 'bad') bad++;
            else if (s.v === 'unsure') unsure++;
          });
          pText.innerHTML = `<span style="color:var(--success)">✅ ${ok}</span> <span style="color:var(--danger)">❌ ${bad}</span> <span style="color:var(--warning)">❓ ${unsure}</span> (${voted}/${count})`;
          if (voted >= count) pText.classList.add("done");
        } else {
          pText.textContent = `Chưa nghe (0/${count})`;
        }
      } catch (e) {}
    });
    document.getElementById("total-verified").textContent = totalVerified;
  </script>
</body>
</html>
"""


def build_all(root_out: Path = Path("/data/export/listen")):
    root_out.mkdir(parents=True, exist_ok=True)
    training_root = training_dir()

    print(f"=== Starting batch generation for {len(CATEGORIES)} categories ===")

    # Custom modified PAGE with a link back to the hub
    page_template = listen_samples.PAGE.replace(
        '<div class="top"><h1>🎧 Listen &amp; verify</h1>',
        '<div class="top"><a href="../index.html" style="color:var(--acc);text-decoration:none;font-weight:600;margin-right:12px;background:#232838;padding:6px 12px;border-radius:8px;border:1px solid var(--line);font-size:13px">← Hub Tổng</a><h1>🎧 Listen &amp; verify</h1>'
    )

    cat_counts = {}

    for cat in CATEGORIES:
        cid = cat["id"]
        out = root_out / cid
        audio_dir = out / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        args = SimpleNamespace(
            source=cat.get("source"),
            split=None,
            status="ok",
            root=None,
            quality=cat.get("quality"),
            ext=cat.get("ext"),
            label_like=None,
            n=cat.get("n", 20),
            seed="0",
            context=True,
            name=cid,
            out=str(root_out),
            raw_gain=False,
        )

        total, rows = listen_samples.query_samples(args)
        cat_counts[cid] = len(rows)

        if not rows:
            print(f"[{cid}] Warning: No rows matched! Skipping.")
            continue

        cards = "".join(
            listen_samples.render_card(r, i, out, audio_dir, not args.raw_gain, training_root)
            for i, r in enumerate(rows, 1)
        )

        shown = {k: v for k, v in vars(args).items()
                 if k in ("source", "split", "status", "root", "quality", "ext", "label_like") and v}
        filters = html.escape(f"{json.dumps(shown, ensure_ascii=False)} — {len(rows)} of {total} matching segments")
        
        page_content = page_template.replace("__FILTERS__", filters).replace("__CARDS__", cards).replace("__NAME__", cid)
        (out / "index.html").write_text(page_content, encoding="utf-8")
        print(f"[{cid}] Generated {len(rows)} / {total} samples -> {out / 'index.html'}")

    # Build Master Dashboard
    groups = {
        "qualities": "6 Tính chất cơ bản (Chord Qualities & Triads)",
        "std_ext": "Bậc mở rộng tiêu chuẩn (Standard 7th & Extensions)",
        "rare_ext": "Nốt màu & Hợp âm biến đổi (Rare & Altered Tensions)",
        "sources": "Kiểm tra theo nguồn dữ liệu (Sources Spot-check)"
    }

    sections_html = []
    for gid, gtitle in groups.items():
        cards_html = []
        for cat in CATEGORIES:
            if cat.get("group") != gid:
                continue
            cid = cat["id"]
            cnt = cat_counts.get(cid, cat.get("n", 20))
            cards_html.append(f"""
        <a class="card" href="./{cid}/index.html" data-id="{cid}" data-count="{cnt}">
          <div class="card-head">
            <span class="card-title">{html.escape(cat['name'])}</span>
            <span class="card-count">{cnt} mẫu</span>
          </div>
          <p class="card-desc">{html.escape(cat['desc'])}</p>
          <div class="card-footer">
            <span class="progress-text">Đang tải...</span>
            <span class="open-btn">Mở nghe →</span>
          </div>
        </a>""")

        grid_html = "\\n".join(cards_html)
        sections_html.append(f"""
    <div class="section-title">{html.escape(gtitle)}</div>
    <div class="grid">
{grid_html}
    </div>""")

    full_sections = "\\n".join(sections_html)
    dashboard_out = root_out / "index.html"
    dashboard_out.write_text(DASHBOARD_HTML.replace("__SECTIONS__", full_sections), encoding="utf-8")
    print(f"=== Master Dashboard generated at {dashboard_out} ===")


if __name__ == "__main__":
    build_all()

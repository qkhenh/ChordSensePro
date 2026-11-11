# AI — Model Results Log

Tư liệu so sánh kết quả giữa các phiên bản model.

---

## Bảng tổng hợp

> Experiment: v2 baseline → v3 clip-level (+ bass) → v4 frame-level head.

| Version | Crop | Aug | Phase | Epoch | val_loss | root_acc | qual_acc | ext_acc | bass_acc | ONNX (ms) | Ghi chú |
|---------|------|-----|-------|-------|----------|----------|----------|---------|----------|-----------|---------|
| v2 | 5s | ✗ | P1 (frozen) | E11/14 | 1.3791 | 45.8% | 53.1% | 0.4% | — | 26.3 (L4) | ext_weight=0.0 |
| v2 | 5s | ✗ | P2 (LoRA) | E14/19 | 1.0266 | 62.8% | 61.8% | 60.0% | — | 27.9 (L4) | baseline |
| **v3-A** | 5s | ✗ | P1 (frozen) | ~E11 | — | ~51.8% | ~53.5% | ~59.8% | ~97.1% | — | +bass head, ext train sớm |
| **v3-A** | 5s | ✗ | P2 (LoRA) | E10/15 | 0.9233 | 61.0% | 60.6% | 60.4% | 97.1% | — | r=32, +k_proj, early stop |
| **v4** | 5s | — | — | — | — | — | — | — | — | — | frame-level head (planned) |

---

## v2 — Phase 1 (MERT frozen, ~550K trainable params)

**Config:** `configs/default.yaml`
| Param | Value |
|-------|-------|
| crop_s | 5 |
| loss_weights | [0.5, 0.5, 0.0] |
| augment | off |
| batch_size | 16 |
| lr_head | 2e-4 |
| dropout | 0.3 |
| pooling | mean |
| early_stop_patience | 3 |

**Training curve:**
| Epoch | train_loss | val_loss | root_acc | qual_acc | ext_acc |
|-------|------------|----------|----------|----------|---------|
| E11 (best) | — | 1.3791 | 45.8% | 53.1% | 0.4% |
| E14 (stop) | — | — | — | — | — |

**Nhận xét:**
- ext_acc = 0.4% vì loss_weight ext = 0.0 → chỉ train root + quality
- root_acc 45.8% → random = 8.3%, tốt hơn 5.5× random
- quality_acc 53.1% → random = 16.7%, tốt hơn 3.2× random

---

## v2 — Phase 2 (LoRA, ~2.1M trainable params)

**Config:** `configs/phase2_lora.yaml`
| Param | Value |
|-------|-------|
| crop_s | 5 |
| loss_weights | [0.3, 0.3, 0.4] |
| augment | off (E1 có augment → tắt từ E2) |
| batch_size | 16 |
| lr_head | 2e-4 |
| lr_lora | 5e-5 |
| lora.r | 16 |
| lora.alpha | 32 |
| lora.target_modules | [q_proj, v_proj] |
| early_stop_patience | 5 |

**Training curve:**
| Epoch | train_loss | val_loss | root_acc | qual_acc | ext_acc |
|-------|------------|----------|----------|----------|---------|
| E14 (best) | — | 1.0266 | 62.8% | 61.8% | 60.0% |
| E19 (stop) | — | — | — | — | — |

**Nhận xét:**
- root +17pp so với P1 (45.8% → 62.8%)
- quality +8.7pp so với P1 (53.1% → 61.8%)
- ext 0.4% → 60.0% (now trained with weight 0.4)
- Augment E1 quá chậm (~4h/epoch) → tắt từ E2

---

## ONNX Benchmark

| Version | Platform | PyTorch (ms) | ONNX (ms) | Speedup |
|---------|----------|-------------|-----------|---------|
| v2 P1 | Colab L4 | 38.9 | 26.3 | 1.48× |
| v2 P2 | Colab L4 | 39.9 | 27.9 | 1.43× |
| v2 P2 | RTX 3050 (CPU) | — | ~3350 (67s/20seg) | — |
| v2 P2 | RTX 3050 (CUDA) | — | 89.2 (p95=89.7) | 37.5× (vs CPU) |

---

## Serve.py Test Results

### v2 P2 — `D:\Untitled.wav` (54.5s, CPU)

| Segment | Chord | Confidence |
|---------|-------|------------|
| [0s-5s] | C# | 0.72 |
| [1s-9s] | F# | 0.59 |
| [4s-14s] | B | 0.68 |
| [9s-19s] | G# | 0.51 |
| ... | ... | ... |

**Nhận xét:** 20 segments, 67s CPU. Chords trong key C# major (consistent). Confidence 0.4–0.8.

---


## v3-A — Phase 2 (LoRA, ~5.4M trainable params)

**Config:** `configs/v3_5s_p2.yaml`
| Param | Value |
|-------|-------|
| crop_s | 5 |
| loss_weights | [0.3, 0.25, 0.25, 0.2] |
| augment | off |
| batch_size | 16 |
| lr_head | 2e-4 |
| lr_lora | 5e-5 |
| lora.r | 32 |
| lora.alpha | 64 |
| lora.target_modules | [q_proj, v_proj, k_proj] |
| n_bass | 13 |
| bass_head_hidden | 128 |
| early_stop_patience | 5 |

**Training curve (selected epochs):**
| Epoch | train_loss | val_loss | root_acc | qual_acc | ext_acc | bass_acc |
|-------|------------|----------|----------|----------|---------|----------|
| E1 | 0.9804 | 1.0013 | 51.8% | 53.5% | 59.8% | 97.1% |
| E2 | 0.8921 | 0.9775 | 55.8% | 58.0% | 59.8% | 97.1% |
| E3 | — | 0.9694 | 55.9% | 52.5% | 59.8% | 97.1% |
| E5 | — | 0.9795 | 57.9% | 53.1% | 59.5% | 97.1% |
| E6 | 0.8469 | 0.9410 | 59.6% | 54.0% | 59.8% | 97.1% |
| E7 | 0.8233 | 0.9511 | 60.7% | 54.8% | 59.8% | 97.1% |
| E8 | 0.8101 | 0.9441 | 60.0% | 60.6% | 60.5% | 97.1% |
| E10 (best) | 0.8265 | 0.9233 | 61.0% | 55.5% | 60.4% | 97.1% |
| E15 (stop) | 0.7577 | 0.9068 | 58.8% | 54.0% | 59.7% | 97.1% |

**Nhận xét:**
- root không cải thiện so với v2 P2 (61.0% vs 62.8%)
- quality giảm nhẹ (60.6% peak vs 61.8%)
- ext không đổi (~60%)
- bass_acc=97.1% nhưng là **majority class memorization** (91.5% data = bass_id=12)
- **Kết luận: thêm bass head không cải thiện root/quality/ext**

**Root cause analysis (2026-11-11):**
- Mean pooling nén 375 MERT frames → 1 vector → mất temporal info
- 56.4% windows 5s chứa 4+ chord → model nghe hỗn hợp predict 1 label
- Passing chords (<1s, 33.5% data) bị chord dài lấn át → bỏ qua
- → Cần chuyển sang frame-level classification (v4)

---

## Vấn đề phát hiện trong v2

| # | Vấn đề | Mức độ | Nguyên nhân |
|---|--------|--------|-------------|
| 1 | Output overlap — cùng 1 timestamp 2 chord khác nhau | 🔴 | Window 5s + step 1s, hiển thị range window |
| 2 | Chord nhảy loạn (C# → C#m → C#) | 🟡 | Quality dao động, merge chỉ merge exact match |
| 3 | Quality/ext kém — toàn ra trưởng/thứ, miss dim7, m7 | 🔴 | qual_acc 61.8%, ext_acc 60.0%, dataset bias |
| 4 | Slash chord không detect được | 🔴 | Không có bass_id trong dataset, không có bass head |
| 5 | Chord ngắn (<2s) bị crop 5s nuốt | 🟡 | 75% chord < 2s nhưng crop 5s |

### Fix đã áp dụng (serve.py)
- ✅ Vấn đề 1: Frame-level voting thay vì window-level output
- ✅ Vấn đề 2: Temporal smoothing (`_smooth_jitter`) + chuẩn hóa `CHORD_SUFFIX` + hỗ trợ 4-output bass
- ❌ Vấn đề 3–5: v3 trained — **không cải thiện** do kiến trúc clip-level (mean pool) → cần v4 frame-level

---

## Target v3

| Metric | v2 hiện tại | Target v3 | Δ |
|--------|-------------|-----------|---|
| root_acc | 62.8% | ≥ 70% | +7.2pp |
| qual_acc | 61.8% | ≥ 70% | +8.2pp |
| ext_acc | 60.0% | ≥ 65% | +5pp |
| bass_acc | — | ≥ 75% | new |
| ONNX latency | 27.9ms (L4) | ≤ 30ms (RTX 3050 CUDA) | — |

---

## Dataset Distribution (reference)

### Quality (6 classes)
| ID | Quality | Count | % |
|----|---------|-------|---|
| 0 | major | 82,888 | 55.3% |
| 1 | minor | 39,440 | 26.3% |
| 2 | dominant | 13,336 | 8.9% |
| 3 | diminished | 2,209 | 1.5% |
| 4 | sus2 | 3,057 | 2.0% |
| 5 | sus4 | 6,443 | 4.3% |

**Imbalance ratio:** major/diminished = 37.5×

### Extension (15 classes)
| ID | Extension | Count | % |
|----|-----------|-------|---|
| 0 | none | 99,668 | 66.5% |
| 4 | min7 | 14,251 | 9.5% |
| 5 | dom7 | 13,317 | 8.9% |
| 3 | maj7 | 6,301 | 4.2% |
| 6 | add9 | 4,121 | 2.7% |
| 8 | nat9 | 3,718 | 2.5% |
| 10 | nat11 | 1,743 | 1.2% |
| 12 | nat13 | 797 | 0.5% |
| 2 | #5 | 507 | 0.3% |
| 1 | b5 | 307 | 0.2% |
| 11 | #11 | 307 | 0.2% |
| 9 | #9 | 283 | 0.2% |
| 7 | b9 | 371 | 0.2% |
| 13 | b13 | 138 | 0.1% |
| 14 | alt | 65 | 0.04% |

**Imbalance ratio:** none/alt = 1,533×

### Slash chords
- **Total:** 12,809 / 149,850 = 8.5%
- Bass intervals phổ biến: /5 (3,233), /1 (2,798), /3 (2,022), /2 (1,036), /b7 (925)


### Chord overlap per window (measured)
| Window | Mean chords | 1 chord | 2 chords | 3+ chords |
|--------|-------------|---------|----------|-----------|
| 5s | 4.0 | 3.4% | 12.1% | **84.5%** |
| 3s | 2.8 | 8.7% | 35.3% | 54.3% |
| 1.5s | 1.8 | 27.8% | 54.0% | 14.7% |

**Kết luận:** crop 5s → model luôn nghe hỗn hợp 3-4 chord → predict 1 label cho cả window → bỏ qua passing chord.
### Chord duration
- Mean: 1,538ms
- Median: 1,410ms
- 75% < 2,000ms
- 33% < 1,000ms


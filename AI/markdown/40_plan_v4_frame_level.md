# Plan v4 — Frame-Level Chord Recognition

> Ngày tạo: 2026-11-11
> Tác giả: Đặng Quốc Khánh (AI) + Antigravity AI
> Branch: `qkhenh`

---

## Tóm tắt

Chuyển từ **clip-level** (1 label/window) sang **frame-level** (1 label/frame, 75Hz).

| | v3 (hiện tại) | v4 (frame-level) |
|---|---|---|
| MERT output | (batch, 375, 1024) | (batch, 375, 1024) |
| Pooling | mean → (batch, 1024) | **không pool** |
| Head input | 1 vector | 375 vectors |
| Head output | 1 label | **375 labels** |
| Resolution | 5s → 1 chord | 5s → **375 chords** (13.3ms/frame) |
| Passing chord | ❌ bỏ qua | ✅ detect được |

---

## Phân tích dữ liệu hỗ trợ

```
Chord duration: median 1.41s (=106 frames @75Hz)
Chord overlap per 5s window: mean 4.0 chords → 56.4% có 4+ chords
MERT temporal resolution: 75 frames/second = 13.3ms/frame
→ Chord 1s = 75 frames → đủ dense để classify
```

---

## Kiến trúc mới

```
Audio 5s (24kHz)
    ↓
MERT encoder
    ↓
(batch, 375, 1024) — 25 layers weighted sum
    ↓
[KHÔNG POOL]
    ↓
Frame-level head (applied per-frame):
    ├─ head_root:    Linear(1024, 128) → ReLU → Linear(128, 12)   → (batch, 375, 12)
    ├─ head_quality: Linear(1024+12, 128) → ReLU → Linear(128, 6) → (batch, 375, 6)
    ├─ head_ext:     Linear(1024+12+6, 256) → ReLU → Drop → Linear(256, 64) → ReLU → Linear(64, 15) → (batch, 375, 15)
    └─ head_bass:    Linear(1024+12+6+15, 128) → ReLU → Linear(128, 13) → (batch, 375, 13)
```

Cascade giống v3 nhưng applied per-frame: mỗi frame t có input là `[emb_t, root_logits_t, quality_logits_t, ...]`.

---

## Thay đổi code

### Phase A — Model + Dataset (core)

#### A1. `model.py` — Bỏ pooling, classify per-frame

**Hiện tại:**
```python
emb = self._pool(h)             # (batch, 1024)
root_logits = self.head_root(emb)  # (batch, 12)
```

**Mới:**
```python
# h shape: (batch, T, 1024) — giữ nguyên temporal dim
root_logits = self.head_root(h)  # (batch, T, 12)

if self.cfg.detach_logits:
    root_input = root_logits.detach()
else:
    root_input = root_logits
quality_logits = self.head_quality(torch.cat([h, root_input], dim=-1))  # (batch, T, 6)
# ... tương tự cho ext, bass
```

**Return shape:** `(batch, T, 12), (batch, T, 6), (batch, T, 15), (batch, T, 13)`

**Config mới:**
```yaml
model:
  frame_level: true   # ← toggle giữa clip-level và frame-level
```

#### A2. `dataset.py` — Multi-label per window

**Hiện tại:**
```python
mid_ms = crop_start_ms + (crop_s * 500)
root_id, quality_id, ext_id, bass_id = self._lookup_chord(song_id, mid_ms)
```

**Mới:**
```python
n_frames = 375  # 75Hz × 5s
root_labels = torch.full((n_frames,), -100, dtype=torch.long)
quality_labels = torch.full((n_frames,), -100, dtype=torch.long)
ext_labels = torch.full((n_frames,), -100, dtype=torch.long)
bass_labels = torch.full((n_frames,), -100, dtype=torch.long)

for t in range(n_frames):
    frame_ms = crop_start_ms + t * (1000 / 75)
    r, q, e, b = self._lookup_chord(song_id, frame_ms)
    root_labels[t] = r
    quality_labels[t] = q
    ext_labels[t] = e
    bass_labels[t] = b
```

**Return:**
```python
return {
    "wav": crop,
    "root": root_labels,      # (375,)
    "quality": quality_labels, # (375,)
    "ext": ext_labels,         # (375,)
    "bass": bass_labels,       # (375,)
}
```

**Tối ưu:** `_lookup_chord` gọi 375 lần quá chậm → dùng binary search / vectorized lookup:
```python
frame_times = crop_start_ms + np.arange(n_frames) * (1000 / 75)
# Vectorized: tìm chord index cho tất cả frame cùng lúc
chord_starts = chords["start_ms"].values
chord_ends = chords["end_ms"].values
indices = np.searchsorted(chord_starts, frame_times, side="right") - 1
# ... fill labels from chords.iloc[indices]
```

#### A3. `losses.py` — Per-frame loss

**Hiện tại:**
```python
l_root = self.ce_root(root_logits, root_true)  # scalar
```

**Mới:**
```python
# root_logits: (batch, T, 12), root_true: (batch, T)
B, T, C = root_logits.shape
l_root = F.cross_entropy(
    root_logits.reshape(B*T, C),
    root_true.reshape(B*T),
    ignore_index=-100,
)
```

Loss tự động skip frames có label=-100 (silence, gap).

#### A4. `train.py` — Per-frame metrics

**Hiện tại:**
```python
root_pred = root_logits.argmax(dim=-1)  # (batch,)
```

**Mới:**
```python
root_pred = root_logits.argmax(dim=-1)  # (batch, T)
mask = root_true != -100
root_correct = (root_pred[mask] == root_true[mask]).sum()
root_total = mask.sum()
```

### Phase B — Inference + Serve

#### B1. `serve.py` — Simplified inference

**Hiện tại:** sliding window → vote → smooth → merge (phức tạp)

**Mới:** sliding window → per-frame predictions → align → merge
```python
# Mỗi window 5s → 375 frame predictions
# Frame t trong window bắt đầu tại pos → timestamp = pos/sr + t/75

for pos, crop in windows:
    root_logits, qual_logits, ext_logits, bass_logits = predictor.predict(wav)
    # root_logits shape: (375, 12)
    for t in range(375):
        time_s = pos / sample_rate + t / 75.0
        frame_idx = int(time_s * 75)
        # Vote tại frame_idx
        root_id = root_logits[t].argmax()
        ...
```

Voting trở nên simple hơn: mỗi global frame chỉ nhận vote từ 1-2 window (overlap rìa), không phải 5 window như clip-level.

Hoặc tốt hơn: **dùng window lớn hơn (10s) với stride 5s, chỉ lấy center 5s predictions** → không overlap → không cần vote.

#### B2. `export_onnx.py` — Output shape mới

```python
# Output names giữ nguyên nhưng shape thay đổi:
# root_logits:    (1, T, 12)  thay vì (1, 12)
# quality_logits: (1, T, 6)
# ext_logits:     (1, T, 15)
# bass_logits:    (1, T, 13)
```

### Phase C — Config + Test

#### C1. Configs mới
```yaml
# configs/v4_5s_p1.yaml
model:
  frame_level: true
  # ... giữ nguyên head dims

train:
  phase: 1
  epochs: 20
  batch_size: 8     # ← giảm vì output lớn hơn
  loss_weights: [0.4, 0.3, 0.2, 0.1]

augment:
  enabled: false
```

```yaml
# configs/v4_5s_p2.yaml
model:
  frame_level: true

train:
  phase: 2
  epochs: 25
  batch_size: 8

lora:
  enabled: true
  r: 32

augment:
  enabled: true
  pitch_shift_range: [-5, 6]
  gain_db: 6.0
  noise_prob: 0.5
```

#### C2. Tests mới

- `test_model.py`: verify output shape `(batch, T, n_classes)` khi `frame_level=true`
- `test_dataset.py`: verify labels shape `(T,)` per sample
- `test_losses.py`: verify per-frame loss với mask
- `test_serve.py`: verify frame-level inference pipeline

---

## Execution order

| # | Task | File(s) | Est. |
|---|------|---------|------|
| 1 | Model frame-level forward | `model.py` | 1h |
| 2 | Dataset multi-label | `dataset.py` | 2h |
| 3 | Loss per-frame | `losses.py` | 30m |
| 4 | Train per-frame metrics | `train.py` | 1h |
| 5 | Tests Phase A | `tests/` | 1h |
| 6 | **Checkpoint: tests pass → commit** | | |
| 7 | Config v4 | `configs/` | 15m |
| 8 | Export ONNX | `export_onnx.py` | 30m |
| 9 | Serve frame-level | `serve.py` | 2h |
| 10 | Evaluate per-frame | `evaluate.py` | 1h |
| 11 | Tests Phase B | `tests/` | 1h |
| 12 | **Checkpoint: full tests pass → commit** | | |
| 13 | Colab notebook update | `notebooks/` | 30m |
| 14 | **Train P1 on Colab** | | ~2h |
| 15 | **Train P2 on Colab** | | ~3h |
| 16 | Export ONNX v4 + benchmark | | 30m |
| 17 | Compare v3 vs v4 → update results log | | 30m |

**Total dev time:** ~11h (code) + ~5h (train)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| VRAM tăng (375 labels/sample vs 1) | Batch size giảm → train chậm hơn | Giảm batch_size 16→8, dùng gradient accumulation nếu cần |
| Dataset `_lookup_chord` 375× chậm | Training loop bottleneck | Vectorized lookup với np.searchsorted |
| ONNX output lớn hơn | Inference chậm hơn | Output chỉ thêm dim T, compute head vẫn nhẹ |
| Backward compat bị phá | Serve.py cũ không chạy với v4 model | Giữ `frame_level` toggle, serve.py check output shape |

---

## Success criteria

| Metric | v3-A hiện tại | Target v4 |
|--------|---------------|-----------|
| root_acc | 61.0% | ≥ 70% |
| qual_acc | 60.6% | ≥ 68% |
| ext_acc | 60.4% | ≥ 65% |
| bass_acc (excl. none) | N/A (97.1% = majority) | ≥ 60% (chỉ slash chords) |
| Temporal resolution | 1s (STEP_S) | **13.3ms** (75Hz) |
| Passing chord detection | ❌ | ✅ |


# AI — Change Log

---

## 2026-11-09 — Kết quả training Phase 1 + Phase 2

### Phase 1 (MERT frozen, ~550K params)
- Config: `default.yaml` — loss_weights [0.5, 0.5, 0.0], augment off, dropout 0.3
- Early stopped at epoch 14 (patience 3)
- **Best (E11):** val_loss=1.3791, root_acc=45.8%, qual_acc=53.1%, ext_acc=0.4%
- ext_acc=0.4% vì loss_weight ext = 0.0 (intentional — chỉ train root + quality)

### Phase 2 (LoRA, ~2.1M params)
- Config: `phase2_lora.yaml` — loss_weights [0.3, 0.3, 0.4], augment off, LoRA r=16 α=32
- Init từ Phase 1 best checkpoint (`--phase1_ckpt`)
- E1 chạy WITH augment (pitch-shift) → ~4h/epoch → tắt augment từ E2
- Early stopped at epoch 19 (patience 5)
- **Best (E14):** val_loss=1.0266, root_acc=62.8%, qual_acc=61.8%, ext_acc=60.0%

### ONNX Export (Phase 1)
- PyTorch L4: mean=38.9ms per 5s clip
- ONNX L4: mean=26.3ms per 5s clip

### Quyết định: Tắt augmentation
- **Lý do 1:** Root distribution đã balanced (max/min = 2.3×) → pitch-shift không cần
- **Lý do 2:** `torchaudio.functional.pitch_shift` trên CPU ~1-2s/clip → ~4h/epoch quá chậm
- **Lý do 3:** Quality imbalance (37×) và ext imbalance (1533×) xử lý bằng Focal Loss, pitch-shift không giúp
- Code augment giữ nguyên (`augment.py`), `augment.enabled: false` trong config

---

## 2026-11-09 — Viết toàn bộ AI source code

### Files tạo mới (session 2)
| File | Mô tả |
|------|--------|
| `src/evaluate.py` | `compute_metrics()` 3-head accuracy + F1, `chord_acc`, `per_class_recall`, confusion matrix |
| `src/extract_embeddings.py` | Cache MERT embeddings (25 layers pooled) cho baseline B1/B2, multi-crop train |
| `src/infer.py` | Sliding window + RMS gate + `valid_combo_mask` + merge consecutive, CLI |
| `src/export_onnx.py` | ONNX export wrapper + PyTorch/ONNX benchmark (warmup + p95) |
| `tests/test_dataset.py` | Crop shape, label range, short song exclusion, val deterministic, missing song |
| `tests/test_augment.py` | `pitch_shift_root` 12×11, augmenter disabled/enabled, gain, noise |
| `tests/test_losses.py` | FocalLoss basic/all-masked/partial/weighted, ChordLoss shapes/nan/backward |
| `tests/test_model.py` | Head shapes, full forward (FakeMERT mock), center pooling, trainable params |
| `notebooks/colab_train.ipynb` | Mount Drive, clone, pip install, copy data, train, resume, infer, ONNX |
| `requirements.txt` | torch, torchaudio, transformers, peft, pandas, sklearn, onnx, pytest |
| `src/__init__.py` | Package init |
| `tests/__init__.py` | Package init |

### Files tạo trước đó (session 1)
| File | Mô tả |
|------|--------|
| `configs/default.yaml` | Hyperparameters (phase 1: freeze MERT, no augment, loss [0.5,0.5,0]) |
| `src/config.py` | Dataclass config + `load_config()` |
| `src/dataset.py` | `ChordDataset` — per-song .npy, random crop 5s, midpoint chord lookup |
| `src/augment.py` | Pitch-shift, gain, noise, `pitch_shift_root()` |
| `src/losses.py` | `FocalLoss` + `ChordLoss` (3-head combined, ignore_index=-100) |
| `src/model.py` | `ChordModel`: MERT frozen + weighted layer sum + cascaded heads |
| `src/train.py` | Full training loop: fp16 AMP, checkpoint/resume, early stopping, CSV log |

### Tổng kết
- **22 files** trong `AI/`
- **Source**: 8 modules (`src/`)
- **Tests**: 4 files (`tests/`)
- **Config**: 1 YAML (`configs/`)
- **Notebook**: 1 Colab notebook
- Code sẵn sàng chạy trên Colab T4 16GB (phase 1: freeze MERT, train heads only)

---

## 2026-11-09 — Cập nhật planning docs cho pipeline v2

### Context
Pipeline DE v2 hoàn tất (2026-11-04). Data đã export tại `DE/data/export/v2/`.
- **6.222 bài** (train 4.324 / val 1.251 / test 647)
- **222.816 chord labels** trong chord timeline
- Audio: `.npy` 24 kHz int16 mono, chiều dài biến thiên (whole section)
- Labels: `chords.parquet` với `(song_id, start_ms, end_ms, label_raw, root_id, quality_id, ext_id)`
- Crawl queue: 6.220 done, 1.436 failed (video unavailable/alignment rejected), 740 skipped

### Thay đổi v1 → v2 (ảnh hưởng đến AI)

| Aspect | v1 | v2 |
|---|---|---|
| Audio format | Segment cố định 2 s, (N, 48000) concatenated | Whole section, per-song `.npy`, chiều dài biến thiên |
| Label format | 1 nhãn/segment tại midpoint | Chord timeline `(start_ms, end_ms)`, nhiều nhãn/bài |
| Demucs model | htdemucs_6s | htdemucs_ft |
| Harmonic stems | piano + guitar + other | **other + bass** |
| Alignment | Chỉ check duration | Chroma offset alignment (`align_offset_ms`, `align_score`) |
| Data scale | ~48K segments / 2.535 bài | **222K chord labels / 6.222 bài** |
| Export | Concatenated arrays + meta parquet | Per-song `.npy` + `chords.parquet` + `config.yaml` + `classes.json` |

### Files cập nhật

| File | Nội dung |
|---|---|
| `AI/markdown/20_DE_training_plan.md` | §0–§15: 2s→5s crop, midpoint→timeline, data stats v2, dataset code, architecture 375 frames, cache strategy, experiments E10, crawl done |
| `AI/markdown/ai_checklist.md` | Export ✅, format v2, htdemucs_ft, sampler trọng số theo bài, MERT 375 frames |

### Quyết định thiết kế mới

1. **Segment-level baseline**: random crop 5 s, gán nhãn tại midpoint của crop từ chord timeline, pooling mean → 1 embedding → 3 head. Đơn giản, tương thích kiến trúc cũ.
2. **Frame-level experiment** (E10): per-MERT-frame label từ timeline, head dự đoán per-frame. Chính xác hơn nhưng phức tạp hơn.
3. **Pooling P1 (mean đều)** làm baseline thay vì P2 (center) — vì chord timeline cho biết chính xác ranh giới, không cần ưu tiên midpoint.
4. **Sampler trọng số theo bài** `∝ 1/√(duration_s)` — bài JAAH dài không chi phối.

---

## 2026-11-04 — Pipeline v2 crawl hoàn tất

- `crawl_queue`: 6.220 done, 1.436 failed, 740 skipped, **0 pending**
- `training_sections`: 6.222 rows, `training_chords`: 222.816 rows
- Disk: `/data/training_v2` 17 GB, `/data/raw` 31 GB
- Export chạy thành công: `DE/data/export/v2/` (~15 GB)
- 1.434 failed chủ yếu do video unavailable (394), alignment rejected (~50), duration mismatch (~20)

### DE changes (ảnh hưởng pipeline)

| File | Thay đổi |
|---|---|
| `DE/dags/dag_ingest_raw.py` | Schedule 2h → **30 phút** |
| `DE/dags/dag_process_audio.py` | Thêm task `reset_stale_jobs` đầu chain |
| `DE/dags/wrapper/process_audio_wrapper.py` | Thêm function `reset_stale_jobs()` — reset job stuck > 30 phút |
| `DE/src/.../crawl_queue_repository.py` | Thêm method `reset_stale_running(cutoff)` |


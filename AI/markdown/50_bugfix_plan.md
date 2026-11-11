# 50 — Master Fix & Retrain Plan (Model v4 Priority)

> Bản cập nhật sau khi review thực tế trên toàn bộ source code và dữ liệu `Hooktheory.json.gz`.
> **Định hướng ưu tiên:** Dồn toàn lực cứu dữ liệu `bass_id` + sửa bug core AI + nâng chất lượng **Model v4 Frame-Level (~75 Hz)** + khớp nối **Demucs `htdemucs_ft`** lúc inference. Các hạng mục BE/FE và dọn code legacy DE dời sang Stage cuối sau khi có model tốt.

**Ngày cập nhật:** 2026-11-11  
**Trạng thái:** 🟢 Đã hoàn tất Stage 1 → Stage 3 (Sẵn sàng Stage 4 Retrain trên Colab L4)

---

## Lộ Trình Thực Thi Mới (Đảo lại thứ tự để không phải train 2 lần)

| Stage | Hạng mục | Mục tiêu cốt lõi | Thời gian ước tính |
|-------|----------|------------------|--------------------|
| **Stage 1** | Bug #1 + Bug #4 | Cứu nhãn `bass_id` cho 78% data Hooktheory (chỉ sửa DB + `chords.parquet` ~3.5MB, **0 GB upload lại audio**) | ~25 phút |
| **Stage 2** | Bug #2 + #6a + #6b + Model v4 Upgrades | Chặn crash `374 vs 375`, sửa LR warmup+cosine theo batch, multi-crop train / full-tile val, FocalLoss cho bass, Temporal Conv1d + Softmax Cascade | ~2 giờ code |
| **Stage 3** | Bug #3 + Bug #7 + Eval Test | Tích hợp Demucs **`htdemucs_ft` (`other + bass`)** vào `serve.py`, sửa `avg_conf`, smoothing v4, cập nhật `colab_train.ipynb` + script chấm 647 bài `test` | ~1 giờ code |
| **Stage 4** | Retrain Colab L4 | Sync 3 file parquet lên Drive → Train `v4_5s_p1` → `v4_5s_p2` → Export ONNX → Benchmark trên `test` split & nhạc thực tế | ~4–5 giờ GPU |
| **Stage 5** *(Hoãn)* | Bug #5 + #8 + #9 | Chuyển volume Docker an toàn, dọn code v1 DE, đồng bộ schema BE/FE & BPM/Key | Sau khi chốt model |

---

## 🚨 STAGE 1 — CỨU DỮ LIỆU `bass_id` TRƯỚC KHI TRAIN (DE → Parquet)

### 1.1. Bug #1: Thêm `bass_id` vào `TrainingChord` & PostgreSQL `training_chords`

**Mức độ:** 🔴 Blocker DB schema  
**File cần sửa:**
- `DE/src/training_data/domain/models/training_section.py` (thêm `bass_id: int = 12` vào `TrainingChord`)
- `DE/src/training_data/infrastructure/repositories/training_section_repository.py` (thêm `bass_id` vào `DDL`, `_CHORD_COLUMNS`, và `replace_song()`)
- `DE/init/init.sql` (thêm `bass_id SMALLINT DEFAULT 12` vào `training_chords`)

**Migration chạy trực tiếp khi bật container `chordsense_postgres`:**
```sql
ALTER TABLE training_chords ADD COLUMN IF NOT EXISTS bass_id SMALLINT DEFAULT 12;
```

---

### 1.2. Bug #4: Khôi phục `inversion` cho Hooktheory (~78% dataset) & Backfill `chords.parquet`

**Mức độ:** 🔴 Nguyên nhân gốc làm `head_bass` học vẹt class `12` (`bass_acc = 97.1%`)  
**Đã kiểm chứng trực tiếp trên `DE/data/datasets/hooktheory/Hooktheory.json.gz`:**
- Tổng hợp âm trong `Hooktheory.json.gz`: `421,298` chords, trong đó **`61,027` chords (`14.49%`) có `inversion > 0`** (`inv=1: 31,619`, `inv=2: 21,970`, `inv=3: 7,438`).
- Có `124` dòng dị biệt `inversion > len(root_position_intervals)` → cần chặn biên `0 < inversion <= len(semitones)`.
- Lưu ý trong `hooktheory.py` L68 có lệnh `return` sớm cho `maj6` / `min6` → phải gom `base_label` trước rồi mới nối `/bass_note` ở cuối hàm để không rớt hợp âm 6 đảo bass.

**Giải pháp code trong `DE/src/data_processing/application/annotation_parsers/hooktheory.py`:**
```python
def harmony_label(root_pitch_class: int, intervals: list[int], inversion: int = 0) -> str:
    root = _ROOTS[root_pitch_class % 12]
    semitones: list[int] = []
    total = 0
    for step in intervals:
        total += step
        semitones.append(total)

    degrees = [d for d in (_degree(s) for s in semitones) if d]
    if not degrees:
        base_label = f"{root}:(1)"
    elif "6" in degrees:
        has_b3, has_b5 = "b3" in degrees, "b5" in degrees
        if has_b3 and has_b5:
            degrees = ["bb7" if d == "6" else d for d in degrees]
            base_label = f"{root}:({','.join(degrees)})"
        elif "b7" not in degrees and "7" not in degrees:
            base_label = f"{root}:{'min6' if has_b3 else 'maj6'}"
        else:
            degrees = ["13" if d == "6" else d for d in degrees]
            base_label = f"{root}:({','.join(degrees)})"
    else:
        base_label = f"{root}:({','.join(degrees)})"

    if 0 < inversion <= len(semitones):
        bass_pc = (root_pitch_class + semitones[inversion - 1]) % 12
        if bass_pc != (root_pitch_class % 12):
            return f"{base_label}/{_ROOTS[bass_pc]}"
    return base_label
```

**Giải pháp cập nhật Dataset KHÔNG cần crawl lại audio (`DE/scripts/backfill_hooktheory_bass.py`):**
1. Bật riêng DB: `docker compose -f DE/docker-compose.yml up -d postgres`.
2. Script đọc `Hooktheory.json.gz`, dựng lại timeline có `/bass` theo `clip_id`.
3. Map qua `training_sections` (`source_url` chứa `#ht=<clip_id>` + `align_offset_ms`) để cập nhật `label_raw` và `bass_id` cho cả:
   - Bảng `training_chords` trong Postgres (đồng bộ Single Source of Truth).
   - 3 file `DE/data/export/v3/{train,val,test}/chords.parquet` và `DE/data/export/v3/classes.json`.
4. Chỉ cần upload đè 3 file `chords.parquet` + `classes.json` (~3.5 MB) lên Google Drive (`MyDrive/ChordSensePro/DE/data/export/v3/`), giữ nguyên 15 GB `.npy` đã có.

---

## 🔥 STAGE 2 — SỬA BUG CORE AI & NÂNG CẤP MODEL V4 (`AI/src/`)

### 2.1. Bug #2: Sửa lệch chiều `374` vs `375` frames + Căn tâm frame + Vector hóa Lookup

**Mức độ:** 🔴 Crash ngay batch 1 khi train v4  
**Giải pháp:**
1. Thêm vào `AI/src/model.py`:
   - `mert_output_length(input_length: int) -> int`: tính chính xác qua `conv_kernel=(10,3,3,3,3,2,2)`, `conv_stride=(5,2,2,2,2,2,2)` (`120,000 → 374`, `72,000 → 224`).
   - `mert_frame_times_ms(crop_start_ms: float, n_frames: int, sr: int = 24000) -> np.ndarray`: tính tâm receptive field từng frame `crop_start_ms + (np.arange(n_frames) * 320 + 200) / sr * 1000.0`.
2. Cập nhật đủ **4 file** (bản cũ thiếu `test_dataset.py`):
   - `AI/src/dataset.py`: dùng `mert_output_length(self.crop_samples)` + `mert_frame_times_ms(...)` và đổi `_lookup_chord_frames()` sang `np.searchsorted` (nhanh hơn ~20× so với vòng for qua từng chord).
   - `AI/src/serve.py`: dùng `mert_output_length(len(crop))` và căn đúng timestamp của từng frame.
   - `AI/tests/test_model.py`: `FakeMERT` trả `T = mert_output_length(wav.shape[-1])` thay vì hardcode `375`.
   - `AI/tests/test_dataset.py`: sửa `expected_frames = mert_output_length(sr * crop_s)`.
3. Thêm guard an toàn trong `AI/src/losses.py`: nếu `logits.shape[1] != targets.shape[1]` thì cắt `T = min(...)` trước khi reshape.

---

### 2.2. Bug #6a: Sửa LR Scheduler có Warmup + Cosine Decay theo từng Batch

**Mức độ:** 🔴 Ảnh hưởng hội tụ Phase 1 & Phase 2 LoRA  
**Vấn đề:** Code cũ tính `T_max` theo batch (~6000) nhưng gọi `scheduler.step()` theo epoch (25 lần → chỉ đi 0.4% chu kỳ) và bỏ quên `warmup_steps`. Nếu chỉ đổi `T_max = epochs` mà bỏ warmup thì LoRA trên MERT-330M dễ bị giật gradient ở epoch đầu.

**Giải pháp trong `AI/src/train.py`:**
- Dùng `SequentialLR` nối `LinearLR(start_factor=0.1, total_iters=warmup_steps)` với `CosineAnnealingLR(T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)`.
- Chuyển `scheduler.step()` vào bên trong `train_one_epoch()` ngay sau mỗi optimizer step hợp lệ (kiểm tra `scaler.get_scale()` không giảm khi dùng fp16 AMP để tránh step lúc skip gradient `inf/nan`).

---

### 2.3. Bug #6b: Tắt CPU Pitch-Shift + Nâng cấp Multi-Crop Train & Full-Tile Validation

**Mức độ:** 🟠 Tốc độ train & chất lượng khai thác dữ liệu Jazz/Extended  
*(Lưu ý đính chính: DE pipeline v2 KHÔNG chạy `AugmentHandler`, nên dữ liệu `.npy` chỉ có 1 bản gốc; tuy nhiên phân bố 12 root tự nhiên đã cân bằng `2.3×` nên không cần CPU pitch-shift tốn 4h/epoch).*

**Giải pháp:**
1. **Augmentation nhanh (~0ms CPU) trong `AI/src/augment.py` & `v4_5s_p2.yaml`:**
   - Đặt `pitch_shift_range: [0, 0]` trong `v4_5s_p2.yaml`.
   - Giữ Random Gain (`±6 dB`) + SNR Gaussian Noise (`15–30 dB`) + thêm **Short Time-Masking** (mask ngẫu nhiên 1–2 đoạn ngắn `80–150ms` về `0` với xác suất `0.3` để ép model dùng ngữ cảnh xung quanh thay vì học vẹt cục bộ). Xóa field thừa `reverb_prob` hoặc để `0.0`.
2. **Khai thác hết dữ liệu bài dài & sửa Validation mù >85% bài trong `AI/src/dataset.py`:**
   - Giới hạn vùng lấy mẫu của mỗi bài trong khoảng có nhãn `[first_chord_ms, last_chord_ms]` (loại bỏ việc crop nhầm vào ±3s đệm không nhãn của Hooktheory).
   - **Split `train`:** Mỗi bài có thời lượng vùng nhãn `D` giây được sinh `n_crops = min(6, max(1, round(D / 15.0)))` cửa sổ ngẫu nhiên mỗi epoch → các bài Jazz dài (JAAH, ChoCo giàu hợp âm `m7b5, dim7, maj9, 13, alt`) được học tương xứng thay vì chỉ 1 đoạn 5s/epoch như clip ngắn.
   - **Split `val` & `test`:** Trải đều các cửa sổ 5s không chồng lấp (`stride = crop_samples`) phủ kín toàn bộ vùng có nhãn của bài → đánh giá **100% hợp âm toàn bài** một cách tất định, giúp `val_loss` phản ánh đúng chất lượng thực tế.

---

### 2.4. Nâng Cấp Kiến Trúc Head, Cascade & Imbalance Loss Cho v4 (Mới — Bổ sung để kéo Accuracy)

1. **Temporal Context Block & Softmax Cascade trong `AI/src/model.py`:**
   - Khi `frame_level=True`, đặt 1 lớp ngữ cảnh thời gian nhẹ trên `(B, T, 1024)` trước 4 head:
     `Conv1d(1024, 1024, kernel_size=5, padding=2, groups=16)` → `GELU` → `Dropout` → residual + `LayerNorm(1024)` (~330K params, xuất ONNX `opset=17` hoàn toàn native, làm mượt nhiễu 13.3ms giữa các frame liền kề).
   - Trong chuỗi cascade `root → quality → ext → bass`: truyền xác suất **`F.softmax(logits, dim=-1)`** (có tôn trọng `detach_logits`) sang head tiếp theo thay vì raw logits chưa chuẩn hóa biên độ.
2. **Chống lệch lớp cho `bass`, `quality`, `ext` trong `AI/src/losses.py`:**
   - Kể cả sau khi cứu Hooktheory, `bass_id = 12` (`none`) vẫn chiếm ~82–84% tổng số frame. Đổi `ce_bass` từ `CrossEntropyLoss` thường sang **`FocalLoss(gamma=2.0, weight=bass_weight)`** với `bass_weight = [1.0]*12 + [0.25]` để chấm dứt tình trạng đoán toàn `12`.
   - Hỗ trợ truyền `quality_class_weight` và `ext_class_weight` làm mượt theo `1 / (count ** 0.3)` (clip `[0.5, 4.0]`, đọc tự động từ `train/chords.parquet`) để kéo recall các hợp âm hiếm (`dim`, `sus2`, `maj7`, `9`, `11`, `13`, `alt`).
3. **Bổ sung Metric thực chất cho Slash Chord trong `AI/src/evaluate.py`:**
   - Giữ `bass_acc`, đồng thời thêm **`bass_slash_acc`** và **`bass_slash_f1`** (chỉ tính trên các frame có `bass_true < 12`, tức hợp âm đảo thật sự) và in trực tiếp ra log mỗi epoch trong `train.py`.

---

## 🎧 STAGE 3 — ĐỒNG BỘ INFERENCE, DEMUCS `htdemucs_ft` & BENCHMARK TEST

### 3.1. Bug #3: Tích hợp đúng Model Demucs `htdemucs_ft` (`other + bass`) vào `serve.py`

**Mức độ:** 🔴 Khớp 100% miền âm thanh giữa Train và Serve  
**Lưu ý kỹ thuật quan trọng (sửa lại từ bản nháp cũ):**
- Model chuẩn mà DE đã dùng để tạo toàn bộ dữ liệu train v2/v3 là **`htdemucs_ft`** (trộn 2 stem **`other + bass`**, bỏ `drums + vocals`, chuẩn hóa đỉnh `0.99`), **không phải** `htdemucs`.
- Gói `demucs==4.0.1` trên PyPI **không có module `demucs.api`**. Phải dùng API chuẩn `demucs.pretrained.get_model("htdemucs_ft")` + `demucs.apply.apply_model(..., segment=6.0, split=True)` (`segment=6.0` giúp chạy mượt trên GPU 4GB VRAM như RTX 3050 mà không OOM).

**Giải pháp trong `AI/src/serve.py` (và thêm `demucs>=4.0.0,<5` vào `AI/requirements.txt`):**
- Thêm singleton `HarmonicSeparator(model_name="htdemucs_ft", segment=6.0)` tự chọn `cuda` / `cpu`.
- Mặc định bật tách `other + bass` cho endpoint `POST /predict` (có query param `separate: bool = True` và biến môi trường `SERVE_USE_DEMUCS=1` để bật/tắt linh hoạt; riêng `/predict_npy` mặc định tắt vì `.npy` từ tập test đã tách sẵn).
- Cập nhật `AI/test_v3.py` (hoặc đổi thành `test_infer_real.py`) chạy qua bước tách `htdemucs_ft` này trước khi đưa vào `ONNXPredictor`.

---

### 3.2. Bug #7 + Hậu xử lý mượt cho v4 Frame-Level trong `serve.py`

1. **Sửa công thức `avg_conf` (loại `bass_conf`):**
   - `avg_conf = 0.4 * root_conf + 0.35 * qual_conf + 0.25 * ext_conf` ở cả nhánh frame-level và clip-level.
   - Xét `reliable = avg_conf >= confidence_threshold` (mặc định `0.30`, giữ khả năng tinh chỉnh qua tham số API; **không** khóa cứng `root_conf >= 0.6` ở mức frame vì trên hợp âm mở rộng xác suất softmax đúng của 12 lớp thường nằm vùng `0.35–0.55`).
2. **Overlap-Add Softmax & Khử nháy frame ngắn cho `_infer_frame_level()`:**
   - Trượt cửa sổ `crop_s=5.0, step_s=2.5`, cộng dồn xác suất softmax có trọng số tam giác (ưu tiên trung tâm cửa sổ hơn rìa) tại mỗi global frame, sau đó mới áp `valid_combo_mask` và `argmax`.
   - Thêm bước gộp/lọc đoạn hợp âm ngắn hơn `min_duration_ms = 160ms` (~12 frames) trước khi `_merge_consecutive()` để output sạch, không nháy hợp âm 13ms.

---

### 3.3. Cập nhật `colab_train.ipynb`, Sửa `infer.py` & Chấm điểm Tập `test` (647 bài)

- Cập nhật `AI/src/evaluate.py` / thêm CLI `python -m src.evaluate --config ... --checkpoint ... --split test` (hoặc `--onnx ...`) chạy tiled windows trên toàn bộ **647 bài tập `test`** để xuất bảng kết quả chuẩn cho báo cáo (`root_acc`, `quality_acc`, `ext_acc`, `bass_slash_acc`, `chord_acc`, macro-F1).
- Sửa `AI/src/infer.py` hỗ trợ LoRA + frame-level (ủy quyền qua logic chuẩn) để tránh crash ở cell cuối Colab.
- Cập nhật `AI/notebooks/colab_train.ipynb`: trỏ mặc định sang `v4_5s_p1` / `v4_5s_p2`, kiểm tra tự động `chords.parquet` mới, train P1 → P2 → Export ONNX → Chấm điểm tập `test`.

---

## ⚪ STAGE 5 — HOÃN LẠI SAU KHI CHỐT MODEL (DE Legacy & BE/FE Integration)

> *Không làm bây giờ để tránh phân tán thời gian. Giữ checklist để xử lý khi ghép hệ thống.*

- **Bug #5 (Bind-mount `/data/training_v2` & `/data/raw` trong `DE/docker-compose.yml`):**
  > ⚠️ **Cảnh báo an toàn:** Tuyệt đối **không** thêm bind-mount `./data/training_v2:/data/training_v2` khi chưa `docker cp` dữ liệu từ named volume `chordsense_data` ra máy host, nếu không thư mục rỗng trên host sẽ che khuất 48 GB dữ liệu trong container. Trong lúc làm Stage 1–4, giữ nguyên `docker-compose.yml` và tuyệt đối không chạy `docker compose down -v`.
- **Bug #8 (Dọn legacy handlers v1 trong DE):** Deprecate `FeatureHandler` (152-dim), `AugmentHandler` DE, sửa nhánh không file nhãn của `AnnotationHandler` nếu cần seed thêm Kaggle local.
- **Bug #9 (Khớp nối BE/FE & Analytics):** Bổ sung trả `tempo_bpm`, `beat_times`, `detected_key` trong `serve.py` khi ghép FE Smart Grid; sửa PK bảng `user_chord_mastery` khi làm phần Practice Analytics.

---

## Checklist Thực Thi (Stage 1 → Stage 3)

- [x] **1. [DE + Data]** Sửa `TrainingChord` + repo + `init.sql` (`bass_id`) và sửa `hooktheory.py` đọc `inversion` (kể cả nhánh `maj6`/`min6`).
- [x] **2. [Data Rescue]** Viết & chạy `DE/scripts/backfill_hooktheory_bass.py` (cập nhật 238,422 dòng trong `chordsense_postgres`, làm mới 6,067 JSON, xuất lại `DE/data/export/v3/{train,val,test}/chords.parquet` & `classes.json` → cứu 15,255 slash chords của Hooktheory, tổng slash chords v3 đạt **29,606 = 13.69%**).
- [x] **3. [AI Bug #2]** Thêm `mert_output_length()` & `mert_frame_times_ms()`, vector hóa `_lookup_chord_frames()`, sửa `dataset.py`, `serve.py`, `test_model.py`, `test_dataset.py`.
- [x] **4. [AI Model & Loss v4]** Thêm `TemporalContextBlock` + Softmax Cascade trong `model.py`; đổi `head_bass` sang `FocalLoss` có trọng số + smoothed class weights cho `quality`/`ext` trong `losses.py`; thêm `bass_slash_acc`/`bass_slash_f1` trong `evaluate.py`.
- [x] **5. [AI Train & Sampling]** Sửa `SequentialLR` (warmup + cosine per-batch) trong `train.py`; multi-crop vùng có nhãn cho `train` + full-tile cho `val`/`test` trong `dataset.py`; tắt CPU pitch-shift + thêm short time-masking trong `augment.py` & `v4_5s_p2.yaml`.
- [x] **6. [AI Serve + Demucs]** Tích hợp `htdemucs_ft` (`other + bass`) vào `serve.py` & `test_v3.py`; sửa `avg_conf`, overlap-add smoothing + `min_duration_ms` cho v4; thêm CLI chấm tập `test` (647 bài) & cập nhật `colab_train.ipynb`.
- [x] **7. [Verify]** Chạy toàn bộ unit tests (`AI/tests` + `DE/tests/unit`) xanh 100%, sẵn sàng sync 3 file parquet + `classes.json` (~3.5 MB) lên Drive để bấm train trên Colab L4.

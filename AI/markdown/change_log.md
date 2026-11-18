# AI — Change Log

---

## 2026-11-18 — Xây dựng Toàn Bộ Kiến Trúc & Pipeline Model v5 (Jazz-Ready: Joint Conditional Heads, 20s–60s Bi-Conformer, LocalKey, Sustain Tension & 145-State Skeleton CRF)

### 1. Chẩn đoán thực nghiệm trên `D:\TestPIANO.wav` & Thiết kế `70_plan_v5_jazz_context.md`
- Soi trực tiếp phổ Chroma CQT, RMS và xác suất thô của `model_v4_5s_p2.onnx` trên `D:\TestPIANO.wav` (cả có và không Demucs):
  - Xác định nguyên nhân toán học gây ảo giác `Cm` trong bài Tone `C Major`: **Marginal Factorized Heads Bug** (4 head biên độc lập lấy `Root = C` của `Cmaj9` ghép với `Quality = min` cộng dồn từ upper-structure `Em7 + Am7`).
  - Xác định nguyên nhân đoạn im lặng đầu bài (`0.00s–1.12s`, `-68.6 dBFS`) ra `C`: `_rms_gate` xét cả cửa sổ 5s, z-score normalize khuếch đại nhiễu `-68 dB` lên `2,700×`, và thiếu lớp im lặng `N`.
  - Xác định nguyên nhân nhảy `Gsus2/F` khi chạy giai điệu `G-D-B-C-D-G` trên nền `F`: thiếu phân tách dải tần và thiếu bộ lọc thời lượng ngân (sustain) của nốt màu.
- Lập bản đặc tả [`70_plan_v5_jazz_context.md`](file:///d:/qkhenh/Project/ChordSensePro/AI/markdown/70_plan_v5_jazz_context.md) giữ nguyên 100% từ vựng 41 `VALID_COMBOS`, dùng thẳng `DE/data/export/v3` (0% đụng tới DE/Docker, 0 GB upload lại data).

### 2. Triển khai trọn bộ Source Code, Configs, Tests & Colab Notebook cho Model v5
- **Nền tảng Nhạc lý & Kiến trúc Mạng:**
  - [`AI/src/music_theory.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/music_theory.py): Ánh xạ song ánh 41 `VALID_COMBOS` ↔ 12 Họ Sườn Guide-Tone (`145` trạng thái tính cả `N`), bảng xoay tròn tương đối `REL_PITCH_INDEX`, trích xuất `shell_chroma` / `full_chroma`, và ước lượng `local_key_id (0..23)`.
  - [`AI/src/conformer.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/conformer.py): `BiConformerEncoder` (4 lớp Macaron + RoPE Multi-Head Attention + Depthwise GLU Conv1d `k=15`) và `MultiScaleSustainLayer` (`80ms` vs `520ms` geometric sustain filter phân biệt nốt giai điệu đi nhanh với nốt màu ngân dài).
  - [`AI/src/crf.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/crf.py): `MarginalLinearChainCRF(num_tags=145)` hỗ trợ marginalize chính xác qua các khoảng khuyết nhãn `-100` + `viterbi_crf_numpy` có `Onset-Decay Locking`.
  - [`AI/src/model.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/model.py) & [`AI/src/losses.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/losses.py): `ChordModelV5` (Chunked 5s MERT + DualChroma + Sustain + Bi-Conformer + `LocalKeyHead` + `Roll(-r) RootConditionedTypeHead` + `145-state CRF` + `merge_lora()`) và `ChordLossV5` (6 thành phần phân tầng).
- **Data Pipeline `20s`, `MelodyOverlayAugmenter` & Huấn luyện 3 Pha:**
  - [`AI/src/augment.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/augment.py): Thêm `apply_v5_musical_augment` (tổng hợp câu giai điệu đi nhanh giữ nguyên nhãn hợp âm + chế độ nốt bậc 9 ngân dài tự nâng nhãn `maj → add9` + chèn intro/outro im lặng `N`).
  - [`AI/src/dataset.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/dataset.py) & [`AI/src/train.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/train.py): Hỗ trợ cửa sổ `20s` (`4 × 5s` chunks, giữ nguyên bài ngắn `< 20s`), Soft-Clamped Normalization (`-55 dBFS` floor), gán `root = 12 (N)` cho frame `< -45 dBFS`, `grad_accum_steps`, và **Phase 3 Full-Song (`60s`) Sequence Polish** trên cached `float16` embeddings.
- **ONNX 2-Stage Export, Serving Ẩn `N` & Colab Notebook:**
  - [`AI/src/export_onnx.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/export_onnx.py): Gộp LoRA `merge_and_unload()`, xuất cặp `encoder.onnx` + `decoder.onnx` + `.meta.json`.
  - [`AI/src/serve.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/serve.py), [`AI/src/infer.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/infer.py), [`AI/test_v3.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/test_v3.py): Tự động phát hiện nhạc cụ độc tấu (`detect_solo_clean_audio`) để bypass Demucs, chạy giải mã 2-stage ONNX toàn bài, và **mặc định lọc bỏ hoàn toàn các đoạn `N` (im lặng) khỏi kết quả output**.
  - Thêm configs [`v5_20s_p1.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v5_20s_p1.yaml), [`v5_20s_p2.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v5_20s_p2.yaml), [`v5_fullsong_p3.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v5_fullsong_p3.yaml), cập nhật [`colab_train.ipynb`](file:///d:/qkhenh/Project/ChordSensePro/AI/notebooks/colab_train.ipynb) và bộ kiểm thử [`test_v5_model.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/tests/test_v5_model.py).

---

## 2026-11-11 — Hoàn tất Model v4 Frame-Level (~75 Hz), Rescue Slash-Chord Data, Retrain L4 & Benchmark Test

### 1. Buổi sáng sớm (`75f5920`, `3541e1f`) — Khởi tạo Kiến trúc v4 Frame-Level
- Chuyển đổi kiến trúc từ clip-level mean-pooling sang **frame-level classification (~75 Hz, ~13.3ms/frame)**:
  - [`AI/src/model.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/model.py): Thêm cờ `frame_level: bool`, bỏ bước `_pool()` để dự đoán chuỗi `(B, T, C)` trực tiếp trên từng frame MERT.
  - [`AI/src/dataset.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/dataset.py): Thêm `_lookup_chord_frames()` gán nhãn chuỗi thời gian cho từng cửa sổ 5s.
  - Tạo configs [`v4_5s_p1.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v4_5s_p1.yaml), [`v4_5s_p2.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v4_5s_p2.yaml) và tài liệu [`40_plan_v4_frame_level.md`](file:///d:/qkhenh/Project/ChordSensePro/AI/markdown/40_plan_v4_frame_level.md).
  - Làm gọn cell Setup tự thân (self-contained) trong [`AI/notebooks/colab_train.ipynb`](file:///d:/qkhenh/Project/ChordSensePro/AI/notebooks/colab_train.ipynb).

### 2. Buổi sáng (`cd68166`) — Triển khai Master Plan `50_bugfix_plan.md` (Stages 1 → 3)

#### Stage 1 — DE: Sửa DB Schema & Khôi phục `bass_id` cho Hooktheory (~78% dataset)
- Cập nhật [`training_section.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/training_data/domain/models/training_section.py), [`training_section_repository.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/training_data/infrastructure/repositories/training_section_repository.py) và [`init.sql`](file:///d:/qkhenh/Project/ChordSensePro/DE/init/init.sql) bổ sung cột `bass_id SMALLINT DEFAULT 12`.
- Sửa [`hooktheory.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_processing/application/annotation_parsers/hooktheory.py) (`harmony_label` & `build_annotation`) đọc trường `inversion` cho mọi nhánh (bao gồm cả `maj6`/`min6`, chặn biên an toàn `0 < inversion <= len(semitones)`).
- Viết và chạy [`DE/scripts/backfill_hooktheory_bass.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/scripts/backfill_hooktheory_bass.py):
  - Khớp **4.806 / 4.806** bài Hooktheory trong PostgreSQL, cập nhật **238.422** dòng trong `training_chords` và **6.067** file annotation JSON.
  - Xuất lại `chords.parquet` + `classes.json` cho `export/v2` & `export/v3` (giữ nguyên 15 GB `.npy`, không cần crawl lại audio): tỷ lệ slash chords thực sự tăng gấp đôi từ `14.467` (`6.69%`) lên **`29.606 / 216.216` (`13.69%`)** (`train: 13.66%`, `val: 13.65%`, `test: 13.96%`).

#### Stage 2 — AI Core: Sửa lỗi căn thời gian 374 vs 375, Loss, Scheduler & Nâng cấp Model v4
- [`AI/src/model.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/model.py):
  - Tính chính xác hình học 7 lớp CNN của MERT qua `mert_output_length()` (`120.000 samples → 374 frames`, `72.000 → 224 frames`) và tâm thời gian `mert_frame_times_ms()`.
  - Thêm `TemporalContextBlock` (`Conv1d(1024, 1024, k=5, groups=16)` + `GELU` + `Linear` + residual `LayerNorm`).
  - Đổi truyền logits thô sang **Softmax Cascade** (`F.softmax` probabilities truyền qua `root → quality → ext → bass`).
- [`AI/src/dataset.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/dataset.py):
  - Vector hóa `_lookup_chord_frames` bằng `np.searchsorted`, khóa biên cắt `[first_ms, last_ms]` để tránh cắt trúng vùng không nhãn.
  - Multi-crop theo độ dài bài cho `train` (**10.198 crops/epoch** trên 4.324 bài), giới hạn tối đa 6 tiles cách đều cho `val` (**6.232 tiles**), và lát gạch kín 100% bài không trùng lắp cho `test` (**7.812 tiles** trên 647 bài, dùng `min_eval_ms` che phần lẻ cuối bài).
- [`AI/src/losses.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/losses.py):
  - Đổi `head_bass` sang `FocalLoss(gamma=2.0)` với `default_bass_class_weight` (`[1.0]*12 + [0.25]` cho lớp `none`) chống học vẹt lớp đa số; thêm `compute_smoothed_class_weights` cho `quality` và `ext`.
- [`AI/src/train.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/train.py) & [`AI/src/augment.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/augment.py):
  - Thêm `build_scheduler()` (`SequentialLR`: Linear warmup + CosineAnnealingLR cập nhật theo từng batch hợp lệ dưới AMP).
  - Tắt CPU pitch-shift (`[0, 0]`), bổ sung waveform `time_mask_prob: 0.3` siêu nhẹ.
- [`AI/src/export_onnx.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/export_onnx.py): Truyền `dynamo=False` vào `torch.onnx.export` tương thích PyTorch 2.9+ với forward hooks 25 lớp của MERT.

#### Stage 3 — Inference, Demucs `htdemucs_ft` & Evaluation CLI
- [`AI/src/serve.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/serve.py) & [`AI/test_v3.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/test_v3.py):
  - Tích hợp `HarmonicSeparator(model_name="htdemucs_ft", segment=6.0)` tách stereo 44.1 kHz và trộn đúng `other + bass` đồng bộ 100% với pipeline DE.
  - Sửa công thức `weighted_chord_conf` (`0.40*root + 0.35*qual + 0.25*ext`, loại bỏ `bass=none` khỏi điểm tự tin).
  - Bổ sung cộng dồn xác suất chồng lấp trọng số tam giác (**Triangle-weighted Overlap-Add**) trong `_infer_frame_level`.
- [`AI/src/evaluate.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/evaluate.py): Thêm CLI `python -m src.evaluate` chấm toàn bộ split kèm các chỉ số chuyên biệt `bass_slash_acc`, `bass_slash_f1`, `chord_no_bass_acc`.
- **Unit tests:** Pass tuyệt đối **288 / 288 tests** (`47 passed` trong `AI/tests`, `241 passed` trong `DE/tests/unit`).

### 3. Buổi chiều (`08dfd20` + Uncommitted local CLI updates) — Retrain Colab L4, Benchmark 647 bài Test & Nghe kiểm thử thực tế
- **Retrain thành công trên Colab L4:** Huấn luyện `v4_5s_p1` → `v4_5s_p2`, xuất và tải model [`AI/results/model_v4_5s_p2.onnx`](file:///d:/qkhenh/Project/ChordSensePro/AI/results/model_v4_5s_p2.onnx) (`1.28 GB`) về máy local.
- **Fix lỗi tràn số FP16 khi eval (`08dfd20`):** Sửa `e.masked_fill(~valid_ext, -1e9)` trong [`AI/src/evaluate.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/evaluate.py#L219-L225) ép kiểu `.float()` và dùng `-1e4` để tránh `RuntimeError: value cannot be converted to type c10::Half without overflow`.
- **Kết quả Benchmark chính thức trên tập `test` (647 bài, 7.812 tiles phủ 100% bài hát ở ~75 Hz):**
  - `root_acc`: **77.84%** (`root_f1_macro`: **77.35%**) — tăng **+16.8pp** so với `v3-A` (`61.0%`).
  - `quality_acc`: **69.88%** (`quality_f1_macro`: **43.75%**) — tăng **+9.3pp** so với `v3-A`.
  - `ext_acc`: **68.97%** (`ext_f1_macro`: **16.91%**) — tăng **+8.6pp** so với `v3-A`.
  - `bass_acc`: **84.80%** | **`bass_slash_f1`: 50.28%** (`bass_slash_acc`: **37.79%**) — xóa bỏ hoàn toàn lỗi đoán mò `none` của `v3-A`.
  - `chord_no_bass_acc` (khớp 3 heads): **50.09%** | `chord_acc` (khớp cả 4 heads): **45.80%** | `test_loss`: **0.6425**.
- **Cải thiện tiện ích CLI local:**
  - Bật `progress=True` trong `HarmonicSeparator.separate_waveform` (`AI/src/serve.py`) để hiển thị thanh tiến độ `%` khi tách nhạc Demucs.
  - Cập nhật [`AI/test_v3.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/test_v3.py): Hỗ trợ biến môi trường `TEST_AUDIO`/`AUDIO_PATH`, `TEST_MODEL`, cờ `--demucs-model` và ưu tiên mặc định `D:\Untitled.wav`.
- **Kiểm thử thực tế trên `D:\Untitled.wav` & Lập kế hoạch xử lý tiếng tỉa rải dây:**
  - Model bắt chuẩn xác vòng hòa thanh giọng `Dm` ở đầu các ô nhịp (`Dm7 → Cdim7 → F7/C → F7 → Bbmaj7 → Am → Abdim → Gm7 → Edim7/E7 → A7#5 → Dm7`).
  - Phát hiện 4 nguyên nhân khiến output bị cắt vụn (`0.01s`) khi nhạc công đánh tỉa/arpeggio: (1) `_smooth_jitter` bỏ lọt nhiễu bậc thang `A → B → C`, (2) chưa gộp `bass_id == root_id` về `12` làm xẻ nhỏ `Am`/`Gm`, (3) Greedy Argmax từng 13.3ms thiếu quán tính chuyển trạng thái, (4) chưa bám lưới phách (beat grid).
  - Lập kế hoạch khắc phục chi tiết tại [`60_postprocess_arpeggio_plan.md`](file:///d:/qkhenh/Project/ChordSensePro/AI/markdown/60_postprocess_arpeggio_plan.md) (Sticky Viterbi + Beat-Sync + Canonical Root-Bass + Iterative Short-Segment Pruning).

---

## 2026-11-10 — Triển khai Improvement Plan v3 (Phase A, B, C1, D) + Benchmark CUDA

### Phase A — DE: Parse `bass_id` & Export Dataset v3
- Thêm [`parse_bass_id()`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_processing/application/mappers/harte_mapper.py#L41-L60) vào `harte_mapper.py`:
  - Map 13 classes: `0..11` (`C..B`), `12` (`none` = không đảo bass / bass trùng root).
  - `/1` (4.570 dòng) và trường hợp bass trùng root (`C:maj/C`) được gộp thống nhất về `12` (`none`).
- Viết [`DE/scripts/export_dataset_v3.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/scripts/export_dataset_v3.py) tạo `DE/data/export/v3/` từ `v2` (giữ nguyên `v2/` làm backup an toàn, dùng hardlink cho 6.222 file `.npy` tốn 0 GB dung lượng phụ):
  - Tổng: **216.216** chord rows | **14.467** slash chords thật sự (`bass != root`, **6.69%**), `none`: 201.749 (**93.31%**).
  - Cập nhật `classes.json` (thêm `"bass"` 13 classes + `"bass_distribution"`) và `config.yaml` (`pipeline_version: 3`).

### Phase B — AI: Nâng cấp Model v3 lên 4 Cascaded Heads (`root → quality → ext → bass`)
- [`AI/src/config.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/config.py): Thêm `n_bass: int = 13`, `bass_head_hidden: int = 128` vào `ModelCfg`; `loss_weights` 4 phần tử `[0.4, 0.3, 0.2, 0.1]` vào `TrainCfg`; `data_dir` mặc định trỏ tới `../DE/data/export/v3`.
- [`AI/src/model.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/model.py): Thêm `self.head_bass = ChordHead(1057, 128, 13)` vào `ChordModel`; `forward()` trả 4 logits `(root, quality, ext, bass)`.
- [`AI/src/dataset.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/dataset.py): Đọc `bass_id` từ `chords.parquet` (fallback `12` nếu chạy trên data v2), hỗ trợ dịch `bass_id` khi bật pitch-shift (`0 <= bass_id < 12`).
- [`AI/src/losses.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/losses.py): Thêm `ce_bass = nn.CrossEntropyLoss(ignore_index=-100)` vào `ChordLoss`, chống `NaN` khi cả batch bị mask.
- [`AI/src/train.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/train.py) & [`AI/src/evaluate.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/evaluate.py): Xử lý 4 heads trong `train_one_epoch` / `validate`, tính `bass_acc` và `bass_f1_macro`.
- [`AI/src/export_onnx.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/export_onnx.py), [`AI/src/infer.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/infer.py), [`AI/src/extract_embeddings.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/extract_embeddings.py): Cập nhật xuất và suy luận 4 outputs (`bass_logits`).

### Phase C1 — Tạo 4 Config YAML & Cập nhật Colab Notebook
- Tạo 4 file config tại `AI/configs/`:
  - [`v3_5s_p1.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v3_5s_p1.yaml), [`v3_5s_p2.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v3_5s_p2.yaml) (`crop_s: 5`, `batch_size: 16`, LoRA `r=32`, `[q_proj, v_proj, k_proj]`)
  - [`v3_3s_p1.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v3_3s_p1.yaml), [`v3_3s_p2.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v3_3s_p2.yaml) (`crop_s: 3`, `batch_size: 24`, LoRA `r=32`, `[q_proj, v_proj, k_proj]`)
- Cập nhật [`AI/notebooks/colab_train.ipynb`](file:///d:/qkhenh/Project/ChordSensePro/AI/notebooks/colab_train.ipynb): Tự động chạy `export_dataset_v3.py` trên Colab từ `/content/data_v2` sang `/content/data_v3` trong ~3s (không cần upload lại 15 GB audio lên Drive).

### Phase D — Cải thiện `serve.py` + Fix bug tên hợp âm + Benchmark RTX 3050 CUDA
- Thêm `_smooth_jitter()` (temporal smoothing) khử nhiễu nhảy hợp âm 1 frame (`A → B → A` thành `A`).
- Cập nhật `ONNXPredictor.predict()` tự động tương thích cả model ONNX 3-output (v2) và 4-output (v3).
- Cập nhật `chord_name()` hỗ trợ slash chord (`Cmaj7/E`, `Bm7/F#`) và đồng bộ bảng `CHORD_SUFFIX` chuẩn từ `value_objects.py` (fix lỗi ghép chuỗi `min + m7 -> Bmm7` thành `Bm7`, `maj + nat9 -> C9` thành `Cmaj9`).
- **Benchmark ONNX v2 P2 trên RTX 3050 (CUDAExecutionProvider):** `mean = 89.2ms`, `p95 = 89.7ms` (nhanh hơn 37.5× so với CPU).
- **Unit tests:** `DE/tests/unit` (**231 passed**), `AI/tests` (**32 passed**, thêm `test_serve.py`).

### Fix các lỗi từ Code Review (v3)
1. **LR Phase 2 (`v3_5s_p2.yaml`, `v3_3s_p2.yaml`, `phase2_lora.yaml`):** Sửa `lr_head: 2.0e-4`, `lr_lora: 5.0e-5` (đảm bảo `lr_lora < lr_head` tránh overfit backbone).
2. **`detach_logits` Phase 2:** Chuyển `detach_logits: true` trong cả `v3_5s_p2.yaml`, `v3_3s_p2.yaml`, `phase2_lora.yaml` để tránh gradient cascade gây nhiễu giữa các head khi fine-tune LoRA.
3. **`evaluate.py` (`chord_acc`):** Cập nhật `chord_acc` kiểm tra đủ cả 4 head (`root == quality == ext == bass`) khi truyền `bass_pred` và `bass_true`.
4. **`infer.py` (Confidence check):** Đồng bộ kiểm tra `if avg_conf < confidence_threshold:` giống `serve.py` thay vì chỉ kiểm tra `root_conf`.
5. **`serve.py` & `config.py` (Backward compatibility):** Giữ default `CLASSES_JSON` và `Config.data_dir` về `v2` kèm danh sách fallback tự động tìm cả `v3` và `v2`; đưa `from collections import Counter` lên đầu file `serve.py`.
6. **`augment.py` & `losses.py`:** Thêm hàm `pitch_shift_bass(bass_id, n_steps)` (kèm unit test) và bổ sung `torch.isnan(l_ext)` guard trong `ChordLoss`.

---

## 2026-11-09 (tối) — Deploy serve.py + phát hiện issues + lập plan v3

### Deploy ONNX inference server (`serve.py`)

Viết [`AI/src/serve.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/serve.py) — FastAPI + ONNX Runtime:
- `GET /health`, `POST /predict` (audio file), `POST /predict_npy` (raw int16)
- Sliding window 5s / step 1s, RMS gate, valid_combo_mask, merge consecutive
- `chord_name()` → human-readable chord string

### Bugs & fixes trong quá trình deploy

| # | Lỗi | Nguyên nhân | Fix | Prod? |
|---|------|-------------|-----|-------|
| 1 | ORT GPU 1.31.1 không dùng CUDA | Yêu cầu CUDA 13, máy có 12.6 | Downgrade `onnxruntime-gpu==1.20.0` | ✅ |
| 2 | ORT 1.20.0 vẫn không tìm cuBLAS DLL | DLL nằm trong `torch/lib` | Prepend `torch/lib` vào `os.environ["PATH"]` | ✅ |
| 3 | `cannot access local variable 'torchaudio'` | `import` trong try block tạo local var | Dùng `importlib.import_module()` | ✅ |
| 4 | `TorchCodec is required` khi load MP3 | torchaudio 2.11 đổi backend sang torchcodec | Thay bằng `librosa.load()` | ✅ |
| 5 | PowerShell `-c` inline SyntaxError | Dấu `"` lồng nhau | Dùng script `.py` riêng | ❌ |

### Test kết quả v2 — phát hiện 5 vấn đề

Test `D:\Untitled.wav` (54.5s, CPU): 20 segments, 67s. Test `D:\full.mp3`.

| # | Vấn đề | Mức độ |
|---|--------|--------|
| 1 | Output overlap — cùng timestamp 2 chord | 🔴 |
| 2 | Chord nhảy loạn (C# → C#m → C#) | 🟡 |
| 3 | Quality/ext kém — toàn ra trưởng/thứ | 🔴 |
| 4 | Slash chord không detect | 🔴 |
| 5 | Chord ngắn (<2s) bị crop 5s nuốt | 🟡 |

### Fix đã áp dụng

- ✅ **Frame-level voting** trong `serve.py` — thay window-level output bằng per-frame majority vote → merge → output clean, không overlap

### Lập plan v3

Xem [`30_improvement_plan_v3.md`](file:///d:/qkhenh/Project/ChordSensePro/AI/markdown/30_improvement_plan_v3.md):
- **Phase A:** DE parse `bass_id` từ `label_raw` (12,809 slash chord có sẵn), re-export dataset v3
- **Phase B:** AI model thêm `head_bass` (13 classes), sửa 7 files
- **Phase C:** Train 4 runs: 5s P1+P2, 3s P1+P2 trên Colab L4
- **Phase D:** serve.py temporal smoothing + bass output
- **Phase E:** Manual test + benchmark

### Files mới / sửa

| File | Thay đổi |
|------|---------|
| `AI/src/serve.py` | Viết mới → fix 5 bugs → frame-level voting |
| `AI/markdown/30_improvement_plan_v3.md` | Plan cải thiện v3 |
| `AI/markdown/model_results_log.md` | Log kết quả model, tư liệu so sánh |

### Commit

- `55bf5e9` — "fix(AI): fix ONNX export for LoRA checkpoint + add serve.py + update docs"
- Các fix sau commit chưa push (serve.py librosa + voting)

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


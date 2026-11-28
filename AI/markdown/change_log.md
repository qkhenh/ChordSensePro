# AI — Change Log

---
## 2026-11-28 — Hoàn tất triển khai toàn bộ Codebase `Model v6` & `Dataset v4` (Sẵn sàng Train 8 Epochs/Phase)

Triển khai hoàn chỉnh toàn bộ hệ thống theo đặc tả [`80_plan_v6_full_retrain.md`](file:///d:/qkhenh/Project/ChordSensePro/AI/markdown/80_plan_v6_full_retrain.md) với cấu hình rút gọn **8 epochs/phase** theo yêu cầu và đầy đủ 6 chốt chặn chống overfit:

### 1. Data Engineering — Dataset `v4` (`DE/data/export/v4`)
- [`harte_mapper.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_processing/application/mappers/harte_mapper.py): Thêm parser `parse_v6_chord_targets()` giữ nguyên 100% tương thích ngược `map_harte()`, cứu toàn bộ họ hợp âm `6` (`MAJ6_SHELL = 1,842`, `MIN6_SHELL = 456`, tổng **2,298** mẫu thực tế trên `v4`!), phân rã 7-bit Multi-Label Tension (`b9, 9, #9, 4_11, #11, b13_#5, 6_13`), 5 lớp thế đảo (`root_pos, inv1_3rd, inv2_5th, inv3_7th, other_slash`), tách `missing_third` (`4,311` power chords) và loại bỏ hoàn toàn 7 hợp âm lỗi lý thuyết.
- [`export_dataset_v4.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/scripts/export_dataset_v4.py): Đã chạy thực tế xuất thành công `DE/data/export/v4` (`4,324 train / 1,251 val / 647 test`, `216,216` dòng hợp âm, hardlink 0 GB trùng lặp audio).
- [`build_v4_piano_and_rare_pack.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/scripts/build_v4_piano_and_rare_pack.py): Script render song-split `POP909` Solo Piano (`train/val/test_piano_pop`) và gói 9 hợp âm cực hiếm (`is_synthetic_rare=True`) kèm fallback synth piano thuần NumPy.

### 2. Kiến trúc `ChordModelV6` & Huấn luyện Chống Overfit
- [`music_theory.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/music_theory.py): Bổ sung 14 Skeletons / 169 CRF states, ma trận **6-D Tonnetz** (0 tham số), prior hòa thanh `build_functional_crf_prior_v6()` (`V->I` = `+0.55`, `ii->V` = `+0.40`, `ivm6->I` = `+0.40`, Tritone Sub = `+0.35`), và `assemble_v6_chord_name()` đảm bảo 0% hợp âm lỗi.
- [`conformer.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/conformer.py): Thêm `DropPath`, `DualScalePassingSustainLayer` (`120ms` gated passing env vs `520ms` sustain env), `ParallelKernelConvModule` (`k=3 || k=15`), `BiConformerEncoderV6` và `FullSongCrossSectionAttention (stride=15)` giúp Intro Piano nhìn thẳng sang Full-Band/Outro.
- [`crf.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/crf.py): Thêm `DynamicBoundaryCRF` (điều biến động `stay_bonus(t)` theo `p_chord_chg`, L2 anchor về prior nhạc lý) + `viterbi_dynamic_crf_numpy()` khớp 100% PyTorch.
- [`model.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/model.py): Thêm `HarmonicCQTFilterbankV6` (**96 bins**, 24 bins/oct ở `C1..B3`), `BassPedalHoldLayerV6` (**4.00s horizon + Onset-Reset Gate** song song 100% ONNX opset-17), `OctavePitchBassModuleV6` (`E_below` + harmonic-suppressed onset flux), `ChordModelV6` (Confidence-Gated Global/Local Key, Factorized `TrueBass -> Root + Skel14 + Inv5 + Tension7D + SlashBass`, LoRA khóa tầng `[8, 21]`).
- [`losses.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/losses.py), [`augment.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/augment.py), [`dataset.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/dataset.py), [`train.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/train.py), [`evaluate.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/evaluate.py): Thêm `ChordLossV6` (7 thành phần + `1.5×` trọng số passing chord `≤ 800ms` + `label_smoothing=0.08`), `HighRegisterPianoAugment` (`HPF 180-260Hz`), `CQT SpecAugment`, chặn cứng `0%` synthetic ở Phase 2 & `≤ 4%` ở Phase 1/3, **`ModelEMA`** (`0.995/0.998`), và chọn `best.pt` theo **Composite Score $S_{\text{val}}$** (`skel14_acc`, `short_passing_chord_f1`, `bass_slash_f1`, `tension_macro_f1`, `sixth_chord_f1`).

### 3. ONNX Export, Serving Động, Config 8 Epochs & Colab Notebook
- [`export_onnx.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/export_onnx.py): Xuất 2-stage ONNX v6 (`encoder + cqt96` & `full-song decoder + dynamic CRF params`).
- [`serve.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/src/serve.py): Hỗ trợ tự động `arch_version == 6`, thêm **Windowed Solo-Piano Stem Router** (`route_audio_stems_dynamic` cửa sổ `4s` crossfade `250ms`), và **Harmonic-Gated Passing Filter** (`_prune_short_segments_v6`) bảo toàn bậc `V` trong `ii-V-I` và đảo bass lướt `G -> D/F# -> Em7` mà không sinh hợp âm rác.
- Config huấn luyện **8 epochs / phase**: [`v6_20s_p1.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v6_20s_p1.yaml), [`v6_20s_p2.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v6_20s_p2.yaml), [`v6_fullsong_p3.yaml`](file:///d:/qkhenh/Project/ChordSensePro/AI/configs/v6_fullsong_p3.yaml), cập nhật [`colab_train.ipynb`](file:///d:/qkhenh/Project/ChordSensePro/AI/notebooks/colab_train.ipynb) và thêm [`eval_golden_songs.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/scripts/eval_golden_songs.py) + [`test_v6_model.py`](file:///d:/qkhenh/Project/ChordSensePro/AI/tests/test_v6_model.py).

---
## 2026-11-26 — Chẩn đoán chuyên sâu lỗi thực nghiệm `v5.1` & Lập Kế Hoạch Huấn Luyện Toàn Diện `Model v6` (`80_plan_v6_full_retrain.md`)

### 1. Chẩn đoán pháp y (Forensic Analysis) trên kết quả kiểm thử thực tế & Code `v5.1`
Đối chiếu chi tiết kết quả review thực tế trên 2 bài *Rồi Em Sẽ Gặp Một Chàng Trai Khác* (`150/185 = 81.1%`) và *Một Ngàn Nỗi Đau* (`74/94 = 78.7%`), bài *Phép Màu* (Only Piano / Ballad), cùng thống kê trực tiếp trên `DE/data/export/v3` (`216,216` dòng nhãn):

1. **Nguyên nhân mất hợp âm bậc `V` trong vòng `2-5-1` sau commit `2271ad0`:**
   - Rule `weak_glitch` trong `_prune_short_segments(prune_low_conf=True)` (`AI/src/serve.py`) xóa mọi phân đoạn `< 820ms` có `confidence < 0.44` nằm kẹp giữa hai hợp âm có `confidence` cao hơn `+0.08`. Trong thực tế, bậc `V` màu Jazz (`Ab9(13)`, `F7b9`, `Eb9sus4`) chia xác suất giữa nhiều họ nên `confidence` tự nhiên chỉ đạt `0.35–0.43` (ví dụ `F7b9` dài `747ms`, `conf=0.372` kẹp giữa `Cm7b5 (0.563)` và `A#m7 (0.558)`), dẫn tới bị cắt mất hoàn toàn.
   - Đồng thời, rule `m_no_b7` trong `_calibrate_v5_crf_and_bass()` dựa trên cửa sổ trung bình nhân `sustain_env` dài `520ms`, làm hợp âm bậc `V` ngắn 1 phách bị phạt `-0.65` điểm emission; còn `D#7#9` bị nhầm thành `D#m7` (3 lần liên tiếp) vì nốt `#9` trùng cao độ với `b3` và thiếu prior tiến trình quãng 4 (`A#m7 → D#7#9 → G#m7`).
2. **Nguyên nhân thực nghiệm trên bài *Phép Màu* (`257s`, ~71 BPM, 1 ô nhịp = `3.40s`) — Tại sao Intro/Verse 1 Only Piano sai & vỡ vụn, khi Guitar Bass vào (`45s+`) thì chuẩn, và Outro Only Piano (`218s+`) lại bắt đúng `G → D/F# → Em`:**
   - **Thủ phạm #1 — Hết hạn bộ nhớ `BassPedalHoldLayer` ngay giữa ô nhịp (`1.70s` < `3.40s`):** `BassPedalHoldLayer` cũ chỉ có tầm nhìn tối đa `8 × 16 = 128 frames = 1.70s`. Trên *Phép Màu*, tay trái Piano gẩy `G` ở phách 1 (`2.67s` và `16.69s`), đúng **`+1.87s` sau (`4.59s` và `18.56s`)** bộ nhớ Bass rơi thẳng về `0`, khiến nửa sau ô nhịp bị rải tỉa tay phải quãng cao lừa thành `Em → C` và `Cm` (ngay cả `D/F#` ở `48.64s` tới `50.24s` = `+1.60s` cũng bị tuột thành `Bm/F# → Bm`).
   - **Thủ phạm #2 — Chênh lệch âm vực tay trái giữa Intro (`Octave 3`) và Outro (`Octave 1–2`) + Cắt khúc `60s`:** Đo trực tiếp `CQT` trên *Phép Màu* cho thấy `LocalKeyHead` đã nhận ra `G major (0.76–0.86)` ngay từ đầu bài, NHƯNG ở Intro/Verse 1 (`8–28s`), Piano đánh treo cao ở Octave 3–4 (`cqt_oct1 = 0.044`, không có năng lượng Octave 1) làm `TrueBassHead` mất phương hướng (`tbass` chỉ đạt `0.26–0.36`); ngược lại ở Outro (`218–228s`), tay trái Piano đánh trầm xuống Octave 1–2 (`cqt_oct1 = 0.219` — mạnh gấp `5×` Intro!) và nằm chung cửa sổ `60s` ngay sau Điệp khúc nên bắt gọn `G (218.5s) → D/F# (221.9s) → Em (225.2s)`.
   - **Thủ phạm #3 — Biến dạng hài âm do `htdemucs_ft` trên đoạn Piano + Vocal mỏng:** Tại `18.56s–20.16s`, audio Raw đoán `C` (thuộc giọng `G major`), nhưng khi qua `htdemucs_ft` bị cắt mất hài bậc 3 dính theo vocal nên biến thành ảo giác **`Cm`** và băm đoạn `20.16s–26.88s` thành 6 mảnh vụn; cộng thêm `BassPedalHoldLayer` thiếu **Onset-Reset** và CRF khóa cứng `(Root, Skeleton)` trước `SlashBass`.
3. **Nguyên nhân bỏ sót Passing Chords đánh nhanh (`250ms–800ms`):**
   - Đo trên `DE/data/export/v3`: có tới **`35,986 / 149,850` hợp âm (`24.0%` tập train)** có thời lượng `≤ 800ms` (`13.8% ≤ 500ms`), nhưng đang bị đè bẹp đồng thời bởi `MultiScaleSustainLayer (520ms)`, `Conformer (conv_kernel=15)`, `crf_stride=4 (53.3ms)` có `stay_bonus` tĩnh, và khóa `< 1000ms` trong `serve.py`.
4. **Nguyên nhân mù họ hợp âm `6` (`Bmaj7 → Bm6`, `Ebm6`, `6/9`) & thiếu tổ hợp màu hiếm:**
   - Trong `harte_mapper.py`, toàn bộ **~2,107 mẫu thật** thuộc họ hợp âm `6` (`:maj6 = 1,306`, `:min6 = 386`, `:maj6(9) = 246`, `:(3,5,13) = 129`...) đang bị gán `ext_id = None` (`PARTIAL`, bị mask `-100` khi train `ChordTypeHead`).

### 2. Lập bản đặc tả [`80_plan_v6_full_retrain.md`](file:///d:/qkhenh/Project/ChordSensePro/AI/markdown/80_plan_v6_full_retrain.md) — Train lại từ đầu tới cuối `Model v6` (Kèm 6 Chốt Chặn Chống Overfit)
Kết hợp kiến trúc từ 3 bài báo nghiên cứu trong `paper/` (*Harmony Transformer* ISMIR 2019, *BTC* ISMIR 2019, *MERT* ICLR 2024):
- **Giai đoạn 1 — Dataset `DE/data/export/v4` + Gói `POP909` Solo Piano Bridge:** Hardlink 6,222 bài từ `v3`, cứu trọn vẹn ~2,107 mẫu họ `6` vào 2 họ sườn mới **`MAJ6_SHELL` & `MIN6_SHELL`** (nâng tổng lên **14 Skeletons / 169 CRF states**), tách nhãn 7-bit Multi-Label Tension + 5-class `inversion_id`, kết hợp render song-split `POP909` Solo Piano (`50% Raw / 50% Demucs`) từ `DE/data/datasets/`.
- **Giai đoạn 2 — Kiến trúc `ChordModelV6`:**
  - `HarmonicCQTFilterbankV6 (96 bins: 24 bins/oct cho C1..B3)` + **`Onset-Reset BassPedalHoldLayerV6`** xóa bóng ma bass cũ trong `13.3ms` khi bước lùi `G2 → F#2 → E2` nhưng giữ vững qua rải dây arpeggio `E2 → B2`.
  - **Cổng `DualScalePassingSustainLayer` (`520ms` + `120ms` có `Polyphonic/Bass Onset Gate`):** Mở khóa cho Passing Chord (`≥ 2` nốt mới đồng thời ở `C2..G4` hoặc đổi đáy Bass) nhưng chặn đứng 1 nốt giai điệu chạy nhanh ở `>= C4`.
  - **`BoundaryHead` (`p_chord_chg`, `p_bass_chg`) + `DynamicBoundaryCRF` (`crf_stride=2 = 26.7ms`)** điều biến động `stay_bonus(t)` và tích hợp prior hòa thanh (`ii–V–I`, Tritone Sub, `IVmaj7 → ivm6`), giải mã đồng thời **Joint `(Root, Skel14, Inversion)`** + **`7-Dim Multi-Label Tension Head (BCE)`** + đặc trưng **6-D Tonnetz** (0 tham số).
- **Giai đoạn 3 & 4 — 6 Kỷ luật Chống Overfit, Huấn luyện 3 Pha (`EMA=0.998`) & Serving Động:** Giới hạn LoRA `r=16` ở tầng `8..21` của MERT, cấm 100% dữ liệu tổng hợp chạm vào MERT ở Phase 2, bật `DropPath=0.15` + `Label Smoothing=0.08` + `CQT SpecAugment` + `Raw/Demucs Mixup`, chọn `best.pt` theo Composite Score $S_{\text{val}}$, và nâng cấp `serve.py` lên **Windowed Solo-Piano Stem Routing (`4s` sliding crossfade)** không xóa nhầm bậc `V` / passing chords.

---
## 2026-11-24 — 
# ChordSensePro — Change Log & Kế Hoạch Chuẩn Hóa Nhạc Lý (Jazz / Pop)

Tài liệu này lưu vết toàn bộ thay đổi về kiến trúc giải mã hợp âm, luật nhạc lý bậc mở rộng (`6 / 7 / 9 / 11 / 13 / alt`), xử lý thế đảo, luật Tone bài hát và lộ trình mở rộng dữ liệu cho các phiên bản tiếp theo.

---

## Phần 1. Change Log

### `[v5.1.2]` — 2026-11-24: Chuẩn hóa Taxonomy Jazz/Pop (Phương án 1 — Zero-Retrain)
Cập nhật trực tiếp trên tầng giải mã Guide-Tone + `sustain_env` + `key_logits` của `model_v5_1_fullsong_p3.onnx` (**không cần train lại**):

- **Loại bỏ 7 hợp âm không tồn tại / sai chuẩn trong nhạc lý thực hành:**
  - `DOM7_NAT`: bỏ **`dom11`** (`11`) — vì bậc `3` trưởng đụng quãng `b9` với bậc `11` tự nhiên; tự động quy đổi về `9sus4` / `7sus4` (khi khuyết bậc 3) hoặc `9` / `7` (khi có bậc 3).
  - `MAJ7_SHELL`: bỏ **`maj11`** — tự động quy đổi về `add4` (khi không có `maj7`), `maj9#11` (khi có `#11`), hoặc `maj9` / `maj7`.
  - `MIN7_SHELL`: khóa xác suất (`DEPRECATED_TYPE_MASK`) đối với **`m7b9`**, **`m7#11`**, **`m7b13`** $\to$ đưa về `m7` / `m6` / `madd4`.
  - `AUG_SHELL`: khóa xác suất **`b5`** (`maj b5`) $\to$ đưa về `aug` hoặc `maj`.
  - `SUS2_SHELL`: khóa xác suất **`7sus2`** $\to$ đưa về `sus2` hoặc `9sus4`.
- **Bổ sung các màu hợp âm mới đã có nền dữ liệu trong `DE/data/export/v3`:**
  - `MAJ_SHELL`: bổ sung **`add4`** (`majadd4`, ví dụ `Cadd4`: `{1, 3, 4, 5}`, vắng `7`).
  - `MIN_SHELL`: bổ sung **`madd4`** (`minor add 4`, ví dụ `Cmadd4`: `{1, b3, 4, 5}`, vắng `b7`).
  - `MAJ7_SHELL`: bổ sung **`maj9#11`** (`{1, 3, 7, 9, #11}`).
  - **Gia đình hợp âm 6**: bổ sung **`6`** (`Cmaj6` / `C6`), **`m6`** (`Cm6` trực tiếp khi có `1 + b3 + 6` vắng `b7`, bên cạnh quy đổi `Xm7b5/b3 → Ym6`), và **`6/9`** (`{1, 3, 5, 6, 9}`). *(Riêng `Cm6/9` mới có 3 mẫu trong dataset nên đưa vào Backlog chờ bổ sung data).*
  - `DOM7_ALT`: bổ sung 3 màu đã có mẫu nền trong `v3` là **`9#11`**, **`13#11`**, **`13b9`**. *(Các màu đa biến âm hiếm `< 10` mẫu được ghi vào Backlog bên dưới).*
- **Bổ sung Luật Tone bài hát (`Diatonic Key Fallback`) cho hợp âm khuyết bậc 3 (`1 + 5` Power Chord):**
  - Kết hợp xác suất giọng toàn bài (`global_song_key`) và cục bộ (`local_key`) từ `key_logits` (24 tone Trưởng/Thứ).
  - Khi một hợp âm chỉ vang bậc `1` và `5` (cả bậc `3` trưởng lẫn `b3` thứ đều vắng `< 0.18`): tra bậc âm giai của tone hiện hành để tự chốt Trưởng/Thứ (ví dụ bài tone **C Major** đánh `C + G` khuyết bậc 3 $\to$ bắt buộc ra **`C`**, không nhảy sang `Cm`; đánh `A + E` $\to$ ra **`Am`**).

---

### `[v5.1.1]` — 2026-11-24: Hiệu chỉnh Guide-Tone CRF & Khử Ảo Giác (Bước A + Bước B)
- **Bước A (`AI/src/music_theory.py`):**
  - Bỏ bậc `3` trưởng khỏi mặt nạ hợp âm `11` tự nhiên; chỉnh `tension_only` của bậc `13` tập trung vào nốt `13` (`9` bán cung).
  - Bổ sung `resolve_jazz_chord_theory()`: quy đổi `Xm7b5/b3 → Ym6` (`Cm7b5/Eb → Ebm6`), `R7b9/3 → (R+4)dim7` (`E7b9/Ab → Abdim7`), `9(13)` (`Ab9(13)`), nâng cấp `m11` / `m13`.
- **Bước B (`AI/src/serve.py`):**
  - Hiệu chỉnh emission CRF (`_calibrate_v5_crf_and_bass`): chống trễ giọng ở vòng `ii-V-I` (`Eb7 → Ebm7`), khôi phục thế đảo 1 (`Ab/C, Gb/Bb, Eb/G, F/A`) và thế đảo 3 (`Eb/Db`), phân biệt `Ab9(13)` với `Fm7/Ab`.
  - Xử lý kẹt `stay_bonus` sau `sus4` (`C9sus4 → C7` giải sạch về `C9sus4 → C`), rút gọn `R7/b7 → R/b7`, gộp phân đoạn nhiễu ngắn độ tin cậy thấp.

---

### `[v5.1.0]` — 2026-11-19..23: Kiến trúc `Model v5.1 (Bass-First)` & Tối ưu Demucs GPU
- **Model `v5.1`:** Thêm CQT 5 quãng tám (`C1..B5`, 60 bins), `BassPedalHoldLayer` (`tau = 1.15s`), chuỗi phân cấp `TrueBassHead → RootHead → ChordTypeHead → SlashBassHead`.
- **Tối ưu `htdemucs_ft` trên GPU VRAM thấp (`<= 6.5 GB`):** Tự động giải phóng VRAM tạm thời trước khi tách nhạc, stream cửa sổ `6s` lên CUDA (`peak VRAM ~1.04 GB`), và lưu cache stem tại `AI/results/.demucs_cache/`.

---

## Phần 2. Kế Hoạch Đầy Đủ (Master Plan)

### Giai đoạn 1 — Đã triển khai ngay (Phương án 1: Zero-Retrain trên `model_v5_1`)
1. Khóa cứng & quy đổi toàn bộ 7 hợp âm sai chuẩn (`dom11, maj11, m7b9, m7#11, m7b13, b5, 7sus2`).
2. Kích hoạt nhóm hợp âm mới có sẵn nền âm học trong `v3`: `add4`, `madd4`, `maj9#11`, `6`, `m6`, `6/9`, `9#11`, `13#11`, `13b9`.
3. Kích hoạt luật **Diatonic Key Fallback** bằng `key_logits` cho các đoạn khuyết bậc 3 (`1 + 5`).

---

### Danh mục Hợp Âm Chưa Đủ Dataset — Lưu Trữ Chờ Xử Lý Sau (Backlog)
Các hợp âm dưới đây có số mẫu gốc cực thấp hoặc bằng `0` trong `DE/data/export/v3` (`149,850` đoạn train). **Chưa mở vội ở giai đoạn hiện tại** để tránh hiện tượng đoán mò (hallucination), sẽ đưa vào cùng đợt bổ sung dữ liệu ở Giai đoạn 2:

| Họ Skeleton | Hợp âm cần bổ sung sau | Số mẫu hiện có (`train` / `tổng v3`) | Trạng thái |
| :--- | :--- | :---: | :--- |
| **Gia đình hợp âm 6** | `Cm6/9` (`min6/9`) | `3` / `3` | Chờ bổ sung data |
| **`DOM7_ALT`** | `7b5b9` | `9` / `20` | Chờ bổ sung data |
| **`DOM7_ALT`** | `7b5#9` | `3` / `6` | Chờ bổ sung data |
| **`DOM7_ALT`** | `7#5#9` | `4` / `6` | Chờ bổ sung data |
| **`DOM7_ALT`** | `7#5b9` | `1` / `11` | Chờ bổ sung data |
| **`HDIM_SHELL`** | `m11b5` (`hdim11`) | `0` / `0` | Chưa có mẫu (`0`) — chờ sinh data |
| **`SUS4_SHELL`** | `7b9sus4` | `0` / `0` | Chưa có mẫu (`0`) — chờ sinh data |
| **`SUS4_SHELL`** | `13b9sus4` | `0` / `0` | Chưa có mẫu (`0`) — chờ sinh data |

*(Lưu ý thêm: màu đơn `7#5` hiện cũng chỉ có `43` mẫu train, nên cần được nhân bản/bổ sung cùng đợt).*

---

### Giai đoạn 2 — Kế Hoạch Train Lại `Model v6` (Để sau khi sẵn sàng)

#### Bước 2.1. Chuẩn hóa lại Data Pipeline (`DE`)
- Cập nhật `DE/src/data_processing/application/mappers/harte_mapper.py` và `classes.json`:
  - Cứu **`1,398` mẫu thật** của họ hợp âm 6 (`:maj6` = `927`, `:min6` = `270`, `:maj6(9)` = `201`) đang bị đánh dấu `PARTIAL` (`ext = None`) sang `OK` (tạo thêm họ `SIXTH_SHELL` hoặc extension riêng).
  - Tách nhãn tường minh cho `add4`, `madd4`, `maj9#11`, `9#11`, `13#11`, `13b9`.
  - Xóa vĩnh viễn 7 tổ hợp không hợp lệ (`dom11`, `maj11`, `m7b9`, `m7#11`, `m7b13`, `b5`, `7sus2`) khỏi `VALID_COMBO_TUPLES` ngay từ bước export dataset `v4`.
  - Áp dụng tone bài hát (`estimate_local_keys`) để gán nhãn Trưởng/Thứ chuẩn cho các đoạn `:5` / `:(1,5)` đủ điều kiện hoặc giữ cờ `missing_third` riêng.

#### Bước 2.2. Bổ sung Dataset cho 8 hợp âm hiếm trong Backlog
- Sử dụng kho SoundFont có sẵn tại `DE/data/datasets/soundfonts` (Grand Piano, Rhodes EP, Acoustic/Jazz Guitar, Strings + Bass):
  - Render đa thế bấm (root position, drop-2, rootless shell voicing, đảo bass) × 12 giọng × nhiều tempo/velocity cho 8 loại hợp âm thiếu mẫu (`m6/9`, `7b5b9`, `7b5#9`, `7#5#9`, `7#5b9`, `m11b5`, `7b9sus4`, `13b9sus4`) và `7#5` (~`500–1,000` đoạn mỗi màu).
  - Cho chạy qua cùng bộ lọc `htdemucs_ft` (`other + bass`) hoặc trộn nhiễu phòng thu nhẹ để đồng nhất miền âm học (domain matching) với nhạc thật.

#### Bước 2.3. Nâng cấp Kiến trúc & Huấn luyện `Model v6`
- Thay vì nhồi hàng chục tổ hợp màu hiếm vào một lớp `Softmax` đơn nhãn dễ bị lệch phân phối:
  - Giữ **CRF Skeleton Head** cho khung nền (`MAJ`, `MAJ6`, `MAJ7`, `DOM7`, `MIN`, `MIN6`, `MIN7`, `MIN_MAJ7`, `HDIM`, `DIM`, `AUG`, `SUS2`, `SUS4`).
  - Thay tầng chọn màu đơn lẻ bằng **Multi-Label Tension Head** (dự đoán độc lập xác suất có mặt của từng bậc `b9, 9, #9, add4/11, #11/b5, #5/b13, 6/13` bằng `BCEWithLogitsLoss` + `sustain_env`), giúp model ghép tổ hợp tự nhiên mọi màu (`7#5b9`, `13b9sus4`, `maj9#11`, `m6/9`...) mà không sợ thiếu tổ hợp chéo.
  - Khởi tạo ấm (`--warm_ckpt`) từ `model_v5_1` để giữ nguyên toàn bộ bộ trích xuất MERT + CQT Bass-First và rút ngắn thời gian train trên GPU.


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


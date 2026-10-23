# AI — Checklist bàn giao từ DE sang AI

> Tạo 2026-10-23. Gom mọi điều cần nhớ khi bắt đầu phần AI, rút ra từ việc kiểm tra data và code phía DE.
> Kế hoạch train chi tiết: [`20_DE_training_plan.md`](20_DE_training_plan.md). Từ điển data: [`DE/markdown/21_DE_data_dictionary.md`](../../DE/markdown/21_DE_data_dictionary.md).
> Tick `[x]` khi xong. Mục **[Quyết định]** cần chốt trước khi chạy.

---

## A. Đã chốt

- [x] Giữ taxonomy 12 root × **6** quality × 15 extension (sus tách thành sus2/sus4, 2026-10-23). **Không có lớp N** trong model (cách "a" ở mục E.3).
- [x] Code train đặt ở `AI/`; theo dõi thí nghiệm bằng `results.csv` + `config.yaml` mỗi lần chạy (không cần TensorBoard/W&B).
- [x] Train trên Colab T4, deploy web app chạy trên PC (RTX 3050 4 GB) → cần model nhỏ/nhanh.
- [x] Tên hợp âm hiển thị lấy từ `CHORD_SUFFIX` trong `value_objects.py` (đã sửa lỗi `Cmaj9` = `C9`).

---

## B. Chuẩn bị data (trước mỗi lần train)

- [ ] **Export lại data mới** (`export_dataset.py`) — data còn tăng (Hooktheory còn ~3.300 clip chờ crawl). Ghi **ngày export + số segment mỗi split** vào `config.yaml` để biết kết quả ứng với data nào.
- [ ] `classes.json` khớp bảng lớp (id theo **thứ tự enum**: quality 0 major, 1 minor, 2 dominant, 3 dim, 4 sus2, 5 sus4; ext 0 none … 14 alt). Đầu vào head extension = 1024 + 12 + **6** = 1042; có `valid_combo_mask` (6 × 15) trong `classes.json` để mask lúc inference.
- [ ] Assert: không `song_id` nào nằm ở hai split; số dòng audio = số dòng meta; không segment toàn 0.
- [ ] **Luôn đọc đủ 3 cột** `root_id / quality_id / ext_id`: `ext_id = 5 (dom7)` còn gồm cả 7sus2/7sus4 và dim7. `dim7` = quality dim + ext dom7 (không có lớp riêng).
- [ ] **Nhãn `partial` (~1.250 hàng):** hợp âm 6, 6/9, power chord → **mask head bị thiếu** khỏi loss; loss không được `nan` khi cả batch thiếu nhãn một head.

### Chuẩn hóa âm lượng (dễ sai, đọc kỹ)
- [ ] Audio trong data **chưa chuẩn hóa biên độ** (stem Demucs nhỏ, median −24 dBFS) → chuẩn hóa ở phía train.
- [ ] Chuẩn hóa **theo cả bài** (hoặc gain cố định), **KHÔNG theo từng segment**: tăng từng segment sẽ biến đoạn im lặng thành tiếng ồn to ngang đoạn có nhạc.
- [ ] Kiểm tra im lặng (RMS) **trước** khi chuẩn hóa, không phải sau.

### Lọc nhiễu
- [x] Segment gần như im lặng nhưng vẫn mang nhãn hợp âm (RMS < −50 dBFS, đo thực tế 2,31 % = 1.151/49.726 segment: Hooktheory 911, JAAH 240, hay do Demucs tách stem bị trống) → **`export_dataset.py` đã tự bỏ** ở cả train/val/test (đổi ngưỡng `--min-rms-dbfs`, giữ lại `--keep-silent`; số bị bỏ nằm trong `classes.json`). **Đừng lọc thêm lần nữa** ở phía train.
- [ ] Meta export có cột `rms_dbfs` (đo trên audio gốc, trước chuẩn hóa): dùng để chỉnh ngưỡng cổng âm lượng ở mục E.3, khỏi đọc lại audio.
- [x] 233 hàng (0,47 %) có tổ hợp (quality, ext) **ngoài `VALID_COMBOS`** (dim+b9 61, sus4+b5 22, sus2+nat13 19, minor+#5 15…) → **`export_dataset.py` đã tự loại** (`--keep-invalid-combos` để giữ; số bị loại nằm trong `classes.json`). Lý do: lúc inference mask chặn các tổ hợp này, giữ lại chỉ dạy model thứ nó không được phép đoán. **Đừng lọc thêm** ở phía train.
- [ ] 39 nhãn `unmapped` (đều chứa bậc `b2`) và `no_chord` đã bị export loại sẵn — không cần xử lý.

### Cân bằng và thiên lệch
- [ ] Hooktheory ≈ 70 %, JAAH ≈ 30 % (jazz chỉ 102 bài). Hợp âm màu đa số đến từ **một nguồn crowd-sourced** → luôn báo metric **theo từng nguồn**, đừng chỉ báo một số tổng.
- [ ] Lớp altered còn mỏng (tổng toàn data): b13 133, b5 114, #9 142, alt 225, #11 323, #5 383. Dùng Focal Loss (γ=2) cho head extension + thử các cách cân bằng lớp (bảng thí nghiệm §9 của plan).
- [ ] Với lớp mỏng, tập test chỉ có vài chục mẫu → kết quả chỉ mang tính định tính, nói rõ trong báo cáo.

---

## C. Train

- [ ] Pitch-shift **kèm đổi nhãn root**, chỉ áp cho `train` (không bao giờ cho val/test). Viết **unit test** cho hàm đổi nhãn (`ChordLabel.pitch_shift` đã có, tái sử dụng).
- [ ] Chạy thử một batch qua MERT: shape `(B, 150, 1024)` và 25 tầng.
- [ ] Có resume checkpoint; lưu config + seed.
- [ ] Mỗi lần chạy ghi một dòng `results.csv` + `config.yaml`.
- [ ] Đánh giá theo **bài** (split đã chia theo bài), không theo segment.

---

## D. Đánh giá

- [ ] **Dựng tập test hợp âm màu kiểm tay** bằng `DE/scripts/listen_samples.py` (hướng dẫn: `DE/markdown/setup.md` mục 7). Vài trăm mẫu, mỗi lớp màu chính khoảng 30–50 mẫu (sus, add9, 9, 11, 13, #11, b9, #9, alt…), lấy **cả JAAH lẫn Hooktheory** để đo thiên lệch nguồn.
- [ ] **JAAH có tín hiệu nhãn–audio yếu nhất** (kiểm chroma 2026-10-23, chi tiết trong `DE/markdown/change_log.md`): nghe ~30 mẫu JAAH trước; nếu nhãn lệch thật thì huấn luyện thử **có/không JAAH** và so sánh trên tập test kiểm tay. Báo cáo kết quả theo **từng nguồn**, đừng chỉ báo số gộp.
- [ ] Chỉ số: F1 từng lớp và macro-F1 (không chỉ accuracy, vì lớp đa số áp đảo); confusion matrix riêng cho root, quality, extension.
- [ ] Báo cáo riêng: theo nguồn; theo lớp hiếm; tổ hợp (quality, ext) model hay đoán sai.
- [ ] Phân tích lỗi bắt buộc (plan §8.3). Nhớ: nhãn Hooktheory do cộng đồng gán, nốt mở rộng yếu (#11, b13) có thể không nghe rõ → một phần "lỗi" là lỗi nhãn.

---

## E. Inference (chạy model trên bài mới)

1. [ ] **Mask tổ hợp hợp lệ:** sau khi chọn quality, chặn các ext không hợp lệ trước khi lấy argmax. Dùng `valid_combo_mask()` (`mask[quality_id][ext_id]`, thứ tự enum) từ `value_objects.py`. Không tự viết lại bảng.
2. [ ] Chạy test: mỗi quality luôn còn ít nhất một ext hợp lệ (đã có test phía DE).
3. [ ] **Xử lý "không có hợp âm" (cách a, không thêm lớp N):**
   - **Bước 1 — cổng âm lượng:** segment có RMS dưới ngưỡng thì trả "không có hợp âm", **không hỏi model**. Tính RMS **trước** khi chuẩn hóa âm lượng.
   - **Bước 2 — độ tự tin:** model tự tin thấp thì đánh dấu "không rõ".
   - Ngưỡng của cả hai bước chỉnh trên **tập val**. Ngưỡng trong `ConfidenceScore.is_reliable` (root ≥ 0,6, quality ≥ 0,55) mới là số tạm.
   - Nếu báo động giả nhiều, mới cân nhắc thêm lớp N ở phiên bản sau (data N có sẵn ~2.100 segment).
4. [ ] Làm mượt theo thời gian (HMM / lọc theo nhịp) — plan S5.
5. [ ] Đo tốc độ trên PC RTX 3050 4 GB; thu nhỏ model nếu chậm (plan §0, S5).

---

## F. Giao diện với BE/FE

- [ ] Với mỗi đoạn, AI trả về: root, quality, ext, độ tự tin từng head, cờ **`reliable`** và cờ **`silent`** (dưới cổng âm lượng). **Hiện, làm mờ hay ẩn đoạn không chắc là việc của BE/FE**, không phải của AI.
- [ ] Báo BE/FE bảng tên mới (`CHORD_SUFFIX`): `Cmaj9` ≠ `C9`, `Csus2`/`Csus4`, `C7sus4`, `C9sus4`, `Cm(maj7)`; quality có **6** lớp (4 = sus2, 5 = sus4). BE đang dùng hàm `chord_symbol()` riêng — hai bên nên thống nhất một bảng.
- [ ] Nói rõ giới hạn vốn từ: sus2/sus4 đã tách nhưng **dễ nhầm nhau** (Csus2 và Gsus4 cùng nốt, khác root); không có lớp riêng cho hợp âm 6, 6/9 hay dim7 (gộp dim + dom7); sus2 chỉ có none/maj7/dom7, sus4 chỉ có none/maj7/dom7/9/13 (các tổ hợp khác bị loại).
- [ ] Còn chờ BE xác nhận trước khi dọn: `song_analysis`, `user_analytics`, `ingest_service` cũ.

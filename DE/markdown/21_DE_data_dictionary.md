# ChordSense Pro — DE: Từ điển dữ liệu (bảng, cột, file)

> Ngày: 2026-10-30. Giải thích **từng nơi lưu data** và **từng cột** của pipeline training, kèm ví dụ thật lấy từ DB.
> Xem tổng quan luồng ở [data_flow.md](data_flow.md); lệnh xem data ở [setup.md](setup.md) mục 7.

---

## 1. Bức tranh tổng thể: data nằm ở đâu?

```
data/datasets/...            File gốc (nhãn + audio local)         ← bạn tải về, không đổi
        │ seed_local_datasets.py
        ▼
Postgres  crawl_queue        "Danh sách việc cần làm" (mỗi dòng = 1 bài/clip)
        │ dag_ingest_raw (mỗi 2 giờ, lấy 100 dòng)
        ▼
MongoDB   raw_audio_jobs     Khu "chờ xử lý": trỏ tới file WAV tạm  (xóa được bất cứ lúc nào)
data/tmp/<id>/*.wav          WAV tạm (xóa sau khi xử lý xong)
        │ dag_process_audio (Demucs → gắn nhãn toàn bộ timeline)
        ▼
Postgres  training_sections  KẾT QUẢ v2: mỗi dòng = 1 bài/clip + metadata
          training_chords    N dòng/bài = chord timeline (đã áp offset)
data/training_v2/audio/*.npy Audio whole section (24 kHz int16 mono, 1D variable length)
data/raw/<song_id>.wav       Raw WAV gốc 44.1 kHz
        │ export_dataset_v2.py
        ▼
data/export/v2/              train/val/test/ (audio/ + chords.parquet) → đưa lên Colab
```

**Quy tắc nhớ:** `crawl_queue` = *việc phải làm*; `training_samples` = *thành phẩm*; Mongo + `data/tmp` = *đồ tạm*.

Các bảng khác trong Postgres (`song_analyses`, `chord_attempts`, `user_chord_mastery`) thuộc **hệ thống BE** (người dùng, phân tích bài hát) — **không liên quan** việc train, bạn có thể bỏ qua.

---

## 2. Bảng `crawl_queue` — danh sách việc cần làm

Mỗi dòng = một **bài hát** (JAAH, ChoCo, GuitarSet, SWD, file piano) hoặc một **clip** (Hooktheory) cần tải/xử lý.

| Cột | Kiểu | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `id` | UUID text | Mã của dòng việc. Dùng để nối với Mongo (`crawl_queue_id`). | `a7e21670-96bb-…` |
| `source_url` | text | **Nơi lấy audio**: URL YouTube hoặc đường dẫn file local. Với Hooktheory có thêm `#ht=<mã clip>` để phân biệt nhiều clip của cùng một video. **Duy nhất** (không trùng). | `https://www.youtube.com/watch?v=gBAcHwaY-0I#ht=wDpgvVAkgad` |
| `annotation_url` | text | **File nhãn** hợp âm đi kèm (đường dẫn trong container). | `/data/datasets/hooktheory/annotations/wDpgvVAkgad.json` |
| `source_type` | text | **Nguồn dữ liệu**: `jaah`, `choco`, `hooktheory`, `guitarset`, `swd`, `local` (piano Kaggle). Quyết định cách xử lý (xem data_flow.md mục 2). | `hooktheory` |
| `status` | text | Trạng thái: `pending` (chờ), `running` (đang chạy), `done` (xong), `failed` (lỗi), `skipped` (bỏ qua — có trong code, hiện chưa có dòng nào). | `done` |
| `priority` | int | **Độ ưu tiên**: số **lớn chạy trước**. Mỗi lượt lấy 100 dòng `pending` có priority cao nhất. Hợp âm hiếm được đẩy lên 8. | `7` |
| `error_message` | text | Lý do lỗi (rỗng nếu không lỗi). | `Video unavailable` |
| `created_at` | timestamp | Lúc seed vào hàng đợi. | `2026-10-19 06:58` |
| `processed_at` | timestamp | Lúc xử lý xong/lỗi. | `2026-10-19 10:08` |

**Cách đọc nhanh:** `GROUP BY source_type, status` cho biết còn bao nhiêu việc. (Lệnh: xem `data_report.sql`.)

---

## 3. Bảng `training_samples` — thành phẩm (cái dùng để train)

Mỗi dòng = **một đoạn audio dài 2 giây** + nhãn hợp âm của nó. Một bài hát tạo ra hàng chục–trăm dòng (cắt trượt, bước ~1,5 s, nên hai đoạn liền nhau **chồng nhau 0,5 s**).

Ví dụ một dòng thật:

```
song_id      b4312f53-4355-4dc2-9b7f-bcd7e09764a1
source_type  hooktheory
source_url   https://www.youtube.com/watch?v=_Q6Str9aDmw#ht=lamkVYrMoDM
segment_idx  3            ← đoạn thứ 3 của clip này
start_ms     4507         ← bắt đầu ở giây 4,507
end_ms       6507         ← kết thúc ở giây 6,507 (đúng 2000 ms)
audio_path   audio/b4312f53-….npy
audio_idx    1            ← nằm ở dòng số 1 trong file .npy đó
sample_rate  24000
label_raw    A#:(3,5,b7,9)  ← nhãn gốc (ký hiệu Harte)
label_status ok
root_id      10           ← A#
quality_id   2            ← dominant
ext_id       8            ← 9   (=> A#9)
split        test
align_status ok
```

| Cột | Ý nghĩa | Ghi chú |
|---|---|---|
| `id` | số thứ tự tự tăng của dòng | không có ý nghĩa nhạc |
| `song_id` | mã của **bài/clip** (UUID). Mọi đoạn của cùng một bài có cùng `song_id`. Cũng là tên file `.npy`. | **Dùng để nhóm theo bài** khi đánh giá |
| `source_type` | nguồn (jaah, hooktheory, …) | để báo cáo riêng theo nguồn |
| `source_url` | audio gốc lấy từ đâu | dùng để chia split |
| `segment_idx` | số thứ tự đoạn trong bài (0, 1, 2, …) | `(song_id, segment_idx)` là **duy nhất** |
| `start_ms`, `end_ms` | thời điểm đoạn trong bài (mili-giây) | luôn dài 2000 ms |
| `audio_path` | đường dẫn file `.npy` chứa audio, tính từ thư mục `data/training/` | mỗi **bài** một file |
| `audio_idx` | đoạn này là **dòng thứ mấy** trong file `.npy` | file có hình dạng `(số đoạn của bài, 48000)` |
| `sample_rate` | tần số lấy mẫu = 24000 Hz | MERT cần 24 kHz |
| `label_raw` | nhãn hợp âm **gốc** ở dạng chữ (Harte): `C:maj7`, `Bb:7(b9)`, `F:(3,5,b7,13)`, `N` (không hợp âm) | giữ lại để ánh xạ lại nếu đổi quy tắc |
| `label_status` | tình trạng ánh xạ nhãn: `ok` (đủ 3 nhãn), `partial` (thiếu quality/extension, vd hợp âm 6), `no_chord` (im lặng), `unmapped` (chưa hiểu nhãn) | export mặc định chỉ lấy `ok` + `partial` |
| `root_id` | **nốt gốc** 0..11 = C, C#, D, D#, E, F, F#, G, G#, A, A#, B | rỗng nếu `no_chord`/`unmapped` |
| `quality_id` | **loại hợp âm** 0 major, 1 minor, 2 dominant, 3 dim, 4 sus2, 5 sus4 | rỗng nếu `partial` kiểu `C5` |
| `ext_id` | **phần mở rộng/hợp âm màu** 0 none, 1 b5, 2 #5, 3 maj7, 4 min7, 5 dom7, 6 add9, 7 b9, 8 9, 9 #9, 10 11, 11 #11, 12 13, 13 b13, 14 alt | rỗng nếu `partial` (6th, 6/9) |
| `split` | `train` / `val` / `test` — chọn theo **hash của video/bài**, nên mọi đoạn cùng bài ở cùng một split | tránh rò rỉ giữa train và test |
| `align_status` | kết quả kiểm tra nhãn có khớp độ dài audio không: `ok`, hoặc `n/a` (nguồn không kiểm tra) | |
| `created_at` | lúc ghi vào DB | dùng xem tốc độ xử lý |

**Cách ghép 3 cột thành tên hợp âm:** `root_id`=10 (A#) + `quality_id`=2 (dominant) + `ext_id`=8 (9) → **A#9**.

**Ba trường hợp đặc biệt:**
- `dim7` = `quality_id` 3 (dim) + `ext_id` 5 (dom7).
- `aug` (tăng) = major + `#5` (ext 2). Không có quality "aug".
- Nhãn `label_status = 'partial'` chỉ thiếu **một phần** (có root nhưng thiếu quality/extension) — khi train phải **bỏ qua** phần thiếu trong loss.

**Nhãn của đoạn lấy ở đâu?** Là hợp âm đang vang tại **điểm giữa** đoạn (giây thứ 1,0 trong đoạn 2 s). Nếu hợp âm đổi giữa chừng thì nhãn chỉ đúng một nửa đoạn.

---

## 3a. Bảng `training_sections` — thành phẩm v2 (1 dòng/bài)

Mỗi dòng = **một bài hát / clip** đã xử lý qua pipeline v2.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `song_id` | UUID text, PK | Mã bài. Cũng là tên file `.npy` và là FK cho `training_chords`. |
| `source_type` | text | Nguồn (jaah, hooktheory, …) |
| `source_url` | text | URL gốc |
| `audio_path` | text | File `.npy` audio 24 kHz int16 1D (variable length), tính từ `training_v2/` |
| `raw_audio_path` | text | File WAV gốc 44.1 kHz, tính từ `data/` |
| `duration_ms` | int | Thời lượng audio đã resample (ms) |
| `align_offset_ms` | int | Chroma offset (ms). Dương = annotation chạy trễ so với audio. |
| `align_score` | real | Điểm cosine chroma tại offset tối ưu (0..1) |
| `align_status` | text | `ok` (≤200ms), `shifted` (200–2000ms, điểm đủ), `rejected` (>2000ms hoặc điểm thấp) |
| `split` | text | train/val/test (hash source_url) |
| `pipeline_version` | int | Luôn = 2 |
| `created_at` | timestamp | Lúc ghi DB |

## 3b. Bảng `training_chords` — chord timeline (N dòng/bài)

Mỗi dòng = **một hợp âm** trong timeline đã áp chroma offset.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `id` | bigserial PK | Số thứ tự tự tăng |
| `song_id` | UUID FK → training_sections | Bài nào |
| `start_ms` | int | Thời điểm bắt đầu (đã áp offset) |
| `end_ms` | int | Thời điểm kết thúc (đã áp offset) |
| `label_raw` | text | Nhãn Harte gốc |
| `label_status` | text | ok/partial/no_chord/unmapped |
| `root_id` | smallint | 0..11 |
| `quality_id` | smallint | 0..5 |
| `ext_id` | smallint | 0..14 |

---

## 4. MongoDB `chordsense_raw.raw_audio_jobs` — khu chờ xử lý

Chỉ là **vùng đệm** giữa hai bước. Có thể xóa hết mà không mất kết quả (kết quả ở Postgres). Mỗi document = một việc đã tải về, chờ `dag_process_audio` xử lý.

| Trường | Ý nghĩa | Ví dụ |
|---|---|---|
| `_id` | mã của lần xử lý (UUID); **chính là `song_id`** ở `training_samples` | `381fc786-5827-…` |
| `source_url` | audio gốc | `/data/datasets/kaggle/…/A_min_5_0.wav` |
| `source_type` | nguồn | `local` |
| `wav_path` | file WAV tạm đã tải/chuyển | `/data/tmp/381fc786-…/A_min_5_0.wav` |
| `annotation_path` | file nhãn đã sao về tạm (rỗng nếu nhãn nằm trong tên file) | `''` |
| `crawl_queue_id` | dòng tương ứng ở `crawl_queue` | `0a4dffd1-…` |
| `status` | `pending` / `processing` / `done` / `failed` | `done` |
| `error_message` | lý do lỗi | `null` |
| `created_at`, `updated_at` | thời điểm tạo/cập nhật | |

---

## 5. File trên ổ đĩa (thư mục `DE/data/`)

| Thư mục | Chứa gì | Có thể xóa không? |
|---|---|---|
| `datasets/` | dữ liệu nguồn (nhãn JAAH, ChoCo, Hooktheory, file piano, …) | **Không** — là nguồn gốc |
| `tmp/<id>/` | WAV tạm trong lúc xử lý | Có (tự xóa sau khi xong) |
| `training/audio/<song_id>.npy` | audio các đoạn 2 s của một bài: mảng int16, hình `(n, 48000)` | Chỉ khi chấp nhận mất audio (DB sẽ trỏ vào file không còn) |
| `export/` | bộ dữ liệu xuất ra để train (xem dưới) | Có, tạo lại được |

### File export (đưa lên Colab)

| File | Nội dung |
|---|---|
| `train_audio.npy`, `val_audio.npy`, `test_audio.npy` | mảng int16 `(N, 48000)`, mỗi dòng là một đoạn 2 s |
| `train_meta.parquet`, … | bảng thông tin, **thứ tự dòng khớp với dòng audio**: `song_id, source_type, source_url, segment_idx, start_ms, end_ms, label_raw, label_status, root_id, quality_id, ext_id, split` |
| `classes.json` | bảng tên lớp (id → tên) + số lượng từng split |

---

## 6. Ký hiệu nhãn gốc (`label_raw`) đọc thế nào

Dạng Harte: `Gốc:loại(độ)`. Ví dụ:

| `label_raw` | Nghĩa | → (root, quality, ext) |
|---|---|---|
| `C:maj` hoặc `C` | Đô trưởng | C, major, none |
| `A:min7` | La thứ 7 | A, minor, min7 |
| `G:7` | Sol bảy | G, dominant, dom7 |
| `Bb:7(b9)` | Si giáng 7 giáng 9 | A#, dominant, b9 |
| `A#:(3,5,b7,9)` | danh sách quãng: 3, 5, b7, 9 (= A#9) | A#, dominant, 9 |
| `F:(4,5,b7)` | quãng 4 thay quãng 3 → treo 4 (sus4) | F, sus4, dom7 |
| `Db:(3,6,9)/5` | 6/9, bass ở quãng 5 | C#, major, *(thiếu ext)* → `partial` |
| `N` | không hợp âm | `no_chord` |

- Quãng: `3` = quãng 3 trưởng, `b3` = quãng 3 thứ, `5` = quãng 5, `b7` = bảy thứ, `7` = bảy trưởng, `9/11/13` = các nốt mở rộng, `b9/#9/#11/b13` = biến âm.
- Phần sau dấu `/` là nốt bass; model hiện **không** dùng bass (bị bỏ khi ánh xạ).

---

## 7. Cách tự xem data

| Muốn biết | Làm gì |
|---|---|
| Tổng quan nhanh | `Get-Content DE\scripts\data_report.sql -Raw \| docker exec -i chordsense_postgres psql -U chordsense -d chordsense` |
| Xem dòng cụ thể | DBeaver (host `localhost`, port `5432`, db/user `chordsense`, mật khẩu `chordsense_dev`) → bảng `training_samples`, thêm điều kiện lọc (vd `WHERE ext_id = 13`) |
| Xem bài nào chứa hợp âm hiếm | `SELECT DISTINCT song_id, source_url FROM training_samples WHERE ext_id = 13;` |
| Nghe thử một đoạn | `scripts/listen_samples.py` (trang HTML có trình phát + nhãn + nút đúng/sai) — xem `setup.md` mục 7 |
| Xem còn bao nhiêu việc | `SELECT source_type, status, count(*) FROM crawl_queue GROUP BY 1,2;` |

---

## 8. Các nhầm lẫn thường gặp

1. **"Một dòng `training_samples` = một bài hát"** → sai. Một dòng = **một đoạn 2 s**; một bài có nhiều dòng.
2. **"`done` trong `crawl_queue` nghĩa là có data"** → `done` chỉ nghĩa là đã xử lý; có thể bài đó cho ra ít đoạn có nhãn.
3. **"Số segment = số mẫu độc lập"** → không; đoạn liền nhau chồng 0,5 s và cùng một hợp âm kéo dài nhiều đoạn.
4. **"`ext_id` = 5 (`dom7`) nghĩa là hợp âm 7 thường"** → đúng một phần: cùng `ext_id` này còn là dim7 (khi `quality` = dim) và 7sus4 (khi `quality` = sus4) và 7sus2 (khi `quality` = sus2). Luôn đọc **cả ba cột**.
5. **"`unmapped` là lỗi pipeline"** → chỉ là nhãn lạ chưa có quy tắc; sửa `harte_mapper.py` rồi chạy `remap_labels.py` là xong, không cần tải lại.

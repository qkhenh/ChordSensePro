# ChordSensePro DE — Setup Guide

> Hướng dẫn setup từ đầu đến khi crawl + xử lý data train chạy được.
> Phạm vi: **DE — chuẩn bị data để train AI** (chưa phải hệ thống BE/FE). Lịch sử thay đổi: [`change_log.md`](change_log.md).

---

## 1. Prerequisites

- Docker Desktop đang chạy (WSL2 backend)
- **GPU (khuyến nghị):** NVIDIA driver mới + Docker Desktop bật GPU support. Máy dev: RTX 3050 4 GB.
  Kiểm tra: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
  → Không có GPU vẫn chạy được (tự fallback CPU) nhưng Demucs chậm ~10–20×; xóa block `deploy:` trong `docker-compose.yml`.
- Git, Python 3.11+ (chạy test local: `pip install -r requirements-dev.txt` rồi `pytest` trong `DE/`; cấu hình ở `pytest.ini`)
- DBeaver (xem PostgreSQL), MongoDB Compass (optional)
- **Dung lượng đĩa:** mỗi segment 96 KB (48000 mẫu int16). JAAH + Kaggle ≈ 18–20k segment ≈ **2 GB**; thêm ~500 bài ChoCo ≈ +8 GB. Output ở `DE/data/training/` (ổ D) + Docker image (~10 GB, ổ C).

---

## 2. Chuẩn bị Dataset

Tải thủ công, bỏ vào đúng thư mục:

```
DE/data/datasets/
  jaah/JAAH-v0.1/MTG-JAAH-xxx/annotations/   ← 113 JSON files
  kaggle/archive/piano_triads/               ← 432 WAV files
  choco/choco/...                            ← clone smashub/choco (chỉ có 4 partition, dùng cho 'chordify')
  choco_release/v1.0.0/v1.0.0/jams/          ← ChoCo release v1.0.0.zip (20.080 JAMS) — audio partition
  guitarset/annotation/  guitarset/audio_mono-mic/   ← Zenodo 3371780
  Schubert_Winterreise/Schubert_Winterreise_Dataset_v2-0/   ← Zenodo 5139893
```

| Nguồn | Có audio? | Ghi chú |
|-------|-----------|---------|
| **JAAH** | ❌ chỉ annotation | Audio được tải từ YouTube bằng tìm kiếm `artist + title`; chọn bản có **thời lượng khớp** annotation |
| **Kaggle piano_triads** | ✅ | 432 WAV, nhãn lấy từ tên file (`A_dim_2_0.wav` → `A:dim`), **không qua Demucs** |
| **ChoCo clone** (`partitions/chordify`) | ❌ chỉ annotation | 50 bài, link YouTube trong `meta.csv` |
| **ChoCo release** | ❌ chỉ annotation | isophonics, robbie-williams, rwc-pop, uspop2002, billboard (1.309 bài): audio từ YouTube search `"<performer> <title>"`, chọn bản **khớp thời lượng** JAMS; bản sai bị loại (`duration_mismatch`) |
| **GuitarSet** | ✅ | 360 clip guitar ~30 s, nhãn JAMS (lấy track *performed*: có sus2/sus4), **không qua Demucs** |
| **Schubert Winterreise (SWD)** | ✅ (2/9 bản thu) | 48 file (HU33, SC06), nhãn CSV, có Demucs |
| McGill, RWC | — | Chưa có data (McGill Billboard đã nằm trong ChoCo release) |
| Pianoteq | — | Cần license |

Danh sách nguồn đã research + trạng thái tích hợp: [`dataset_sources.md`](dataset_sources.md).

**JAAH:** [zenodo.org/record/1290737](https://zenodo.org/record/1290737) → `JAAH-v0.1.zip` → giải nén vào `data/datasets/jaah/`.
**Kaggle:** tải `piano_triads` → `data/datasets/kaggle/archive/piano_triads/`.
**ChoCo (clone):** `cd DE/data/datasets/choco && git clone https://github.com/smashub/choco`
**ChoCo release:** [v1.0.0.zip](https://github.com/smashub/choco/releases/download/v1.0.0/v1.0.0.zip) (178 MB) → giải nén vào `data/datasets/choco_release/v1.0.0/`.
**GuitarSet:** [zenodo.org/records/3371780](https://zenodo.org/records/3371780) → `annotation.zip` + `audio_mono-mic.zip` → giải nén vào `data/datasets/guitarset/`.
**SWD:** [zenodo.org/records/5139893](https://zenodo.org/records/5139893) → `Schubert_Winterreise_Dataset_v2-0.zip` → `data/datasets/Schubert_Winterreise/`.

### (Tùy chọn) cookies YouTube

Nếu YouTube chặn (lỗi "Sign in to confirm you're not a bot", 403 dai dẳng):
1. Cài extension *Get cookies.txt LOCALLY*, đăng nhập YouTube bằng **tài khoản phụ**, export cookies.
2. Lưu thành `DE/config/cookies.txt` (đã `.gitignore`). Container tự dùng nếu file tồn tại (`YTDLP_COOKIES`).

---

## 3. Build & Start

```bash
cd DE
docker compose build airflow       # lần đầu ~10 phút; cài yt-dlp[default] + deno + pyarrow
docker compose up -d
docker compose ps                  # 5 container healthy
```

Kiểm tra GPU + JS runtime trong container:

```bash
docker exec chordsense_airflow python -c "import torch; print('cuda:', torch.cuda.is_available())"
docker exec chordsense_airflow deno --version
```

> `cuda: False` → Docker chưa thấy GPU (xem mục 1). Pipeline vẫn chạy nhưng chậm.

---

## 4. Seed Data vào crawl_queue

```bash
# tất cả nguồn (idempotent: ON CONFLICT DO NOTHING)
docker exec chordsense_airflow bash -c \
  "cd /opt/airflow && PYTHONPATH=/opt/airflow python3 scripts/seed_local_datasets.py"

# hoặc từng nguồn:  --source jaah | kaggle | choco | guitarset | swd | choco-release | hooktheory   (+ --dry-run)

# Hooktheory (cần data/datasets/hooktheory/Hooktheory.json.gz từ Sheet Sage):
#   --hooktheory-scope rare  (mặc định, ~6.000 clip có hợp âm hiếm)  |  all (~17.800 clip dùng được)
docker exec -w /opt/airflow chordsense_airflow env PYTHONPATH=/opt/airflow \
  python scripts/seed_local_datasets.py --source hooktheory --hooktheory-scope rare
```

Expected:
```
[jaah]          Found 113 tracks
[kaggle]        Found 432 files
[guitarset]     Found 360 clips
[swd]           Found 48 recordings
[choco-release] isophonics 225 · robbie-williams 62 · rwc-pop 100 · uspop2002 195 · billboard 727
[hooktheory]    6084 clips seeded (scope=rare)
```

> Hooktheory tải **một đoạn** của video YouTube (clip ~20 s, ~0.3–1.5 MB mỗi clip) → tốn mạng; giấy phép CC BY-NC-SA 3.0 (phi thương mại).

`crawl_queue.priority` quyết định thứ tự crawl (cao trước): hooktheory 7/6 (nhiều/ít hợp âm hiếm) → guitarset 5 → swd 4 → choco-release (pop/rock) 3 → billboard 1 → còn lại 0.

---

## 5. Chạy pipeline

```
dag_ingest_raw  (mỗi 2 giờ, hoặc Trigger tay)
  pick_urls → download_and_stage → send_summary → trigger_dag_process
        │  Tải annotation → đọc duration → tải audio (JAAH: chọn đúng bản YouTube) → Mongo raw_audio_jobs
        ▼
dag_process_audio  (được trigger tự động)
  pick_pending_jobs → process_and_save → send_summary
        AlignmentCheck → Separate (Demucs GPU) → Beat → Segment → Annotation → Label → SampleWriter
        │  ghi  /data/training/audio/{song_id}.npy   +   Postgres training_samples
```

1. Vào **http://localhost:8080** (`admin / admin`), bật 2 DAG `dag_ingest_raw`, `dag_process_audio`.
2. Bấm **▶ Trigger** `dag_ingest_raw`.
3. Mỗi lượt xử lý `INGEST_BATCH_SIZE` URL (hiện 100, đổi trong `docker-compose.yml`).

### Chạy lại từ đầu (sau khi sửa code / data hỏng)

```bash
# xem trước, chưa đổi gì
docker exec chordsense_airflow python /opt/airflow/scripts/reset_for_recrawl.py --dry-run
# thực hiện: xóa Mongo staging, đưa crawl_queue về pending
docker exec chordsense_airflow python /opt/airflow/scripts/reset_for_recrawl.py
# chỉ retry các URL lỗi (giữ nguyên các bài đã xong)
docker exec chordsense_airflow python /opt/airflow/scripts/reset_for_recrawl.py --only-failed
# làm sạch luôn training_samples + file .npy
docker exec chordsense_airflow python /opt/airflow/scripts/reset_for_recrawl.py --truncate-samples
# chỉ reset MỘT nguồn (vd. xử lý lại JAAH): xóa samples + .npy của nguồn đó, xếp lại queue và đẩy lên đầu (priority 8)
docker exec chordsense_airflow python /opt/airflow/scripts/reset_for_recrawl.py --source jaah --truncate-samples --priority 8 --dry-run
```

---

## 6. Output để train

Một dòng `training_samples` = một segment 2 s đã gán nhãn. Audio nằm trong `.npy` (int16, 24 kHz, `(N, 48000)`), trỏ bởi `audio_path` + `audio_idx`.

```sql
SELECT source_type, split, label_status, count(*)
FROM training_samples GROUP BY 1,2,3 ORDER BY 1,2,3;
```

`label_status`: `ok` (đủ 3 head) · `partial` (có root, thiếu quality/extension — 6th, power chord…) · `no_chord` (`N`) · `unmapped` (chưa hiểu nhãn).

Sau khi sửa `HarteMapper`, map lại dữ liệu đã lưu (không cần crawl lại):
```bash
docker exec chordsense_airflow python /opt/airflow/scripts/remap_labels.py --dry-run
docker exec chordsense_airflow python /opt/airflow/scripts/remap_labels.py
```

### Export cho Colab

```bash
docker exec chordsense_airflow python /opt/airflow/scripts/export_dataset.py --stats   # chỉ in phân bố lớp
docker exec chordsense_airflow python /opt/airflow/scripts/export_dataset.py           # ghi file
```

Kết quả ở `DE/data/export/`: `train_audio.npy`, `train_meta.parquet`, `val_*`, `test_*`, `classes.json`.
Upload thư mục này lên Drive/Colab. Lưu ý khi train:
- Audio là int16 **chưa chuẩn hóa** biên độ (stem Demucs có thể nhỏ) → normalize ở phía train (theo cả bài, không theo từng segment).
- Export **tự bỏ segment gần như im lặng** (RMS < −50 dBFS, đo thực tế 2,31 % = 1.151/49.726 segment: Hooktheory 911, JAAH 240): chúng mang nhãn hợp âm nhưng không có âm thanh (thường do Demucs tách stem bị trống). Đổi ngưỡng bằng `--min-rms-dbfs -45`, giữ lại tất cả bằng `--keep-silent`. Số segment bị bỏ ghi trong `classes.json` (`dropped_silent`).
- Export cũng **tự bỏ hàng có tổ hợp (quality, extension) không hợp lệ** theo `VALID_COMBOS` (vd dim+b9, sus4+b5; 233 / 49.840 = 0,47 %); giữ lại bằng `--keep-invalid-combos`. Số bị bỏ: `dropped_invalid_combos`; ma trận tổ hợp hợp lệ 6 × 15 để mask lúc inference: `valid_combo_mask` (đều trong `classes.json`). Hàng thiếu một head (`partial`) không bị loại.
- Meta có thêm cột `rms_dbfs` (độ lớn của segment gốc, đo **trước** chuẩn hóa) — dùng để chỉnh ngưỡng "không có hợp âm" ở inference.
- `partial`: `quality_id`/`ext_id` có thể null → mask head đó trong loss.
- Split đã cố định theo bài hát (70/20/10) — không shuffle lại ở mức segment.
- Augmentation (pitch-shift, noise, reverb…) chỉ áp cho `train`, làm ở phía train.
- Lệnh export (kể cả `--stats`) đọc lại toàn bộ audio để đo độ lớn (~5 GB) nên mất khoảng nửa phút đến vài phút.

---

## 7. Check Data

### PostgreSQL (DBeaver)

| Field | Value |
|-------|-------|
| Host / Port | `localhost` / `5432` |
| Database / User | `chordsense` / `chordsense` |
| Password | `chordsense_dev` |

```sql
SELECT source_type, status, count(*) FROM crawl_queue GROUP BY 1,2 ORDER BY 1,2;
SELECT error_message, count(*) FROM crawl_queue WHERE status='failed' GROUP BY 1 ORDER BY 2 DESC;
```

### Báo cáo tổng hợp data train (1 lệnh)

[`scripts/data_report.sql`](../scripts/data_report.sql) in ra: trạng thái hàng đợi theo nguồn, top lỗi, số segment theo nguồn/split, `label_status`, phân bố **quality / extension / root** (có tên lớp), lớp hiếm theo nguồn, nhãn `unmapped`, và tốc độ xử lý theo giờ. Chỉ đọc, không sửa gì, không dùng mạng.

```powershell
# chạy ở thư mục gốc repo (PowerShell)
Get-Content DE\scripts\data_report.sql -Raw | docker exec -i chordsense_postgres psql -U chordsense -d chordsense
```

> Lưu ý: `dim7` không có lớp riêng — nó được lưu là quality `dim` + extension `dom7`. Bảng extension gộp cả hai.

Kiểm tra DAG còn đang chạy không (trước khi tắt Docker):

```powershell
docker exec chordsense_airflow airflow dags list-runs -d dag_ingest_raw --state running
docker exec chordsense_airflow airflow dags list-runs -d dag_process_audio --state running   # "No data found" = rảnh
```

### Xem nhãn (label) trong `training_samples`

Mỗi dòng = 1 segment 2 s. Cột nhãn: `label_raw` (chuỗi Harte gốc, vd `Bb:7(b9)`), `root_id` / `quality_id` / `ext_id` (id lớp sau khi map), `label_status` (`ok` / `partial` / `no_chord` / `unmapped`).

```sql
-- nhãn gốc nào nhiều nhất
SELECT label_raw, count(*) AS n FROM training_samples GROUP BY 1 ORDER BY 2 DESC LIMIT 30;

-- xem từng segment của 1 nhãn (id lớp + nguồn + split)
SELECT song_id, source_type, split, label_raw, root_id, quality_id, ext_id, label_status
FROM training_samples WHERE label_raw LIKE 'C:7%' LIMIT 20;

-- số segment theo lớp (0=major 1=minor 2=dominant 3=dim 4=sus2 5=sus4)
SELECT quality_id, count(*) FROM training_samples WHERE label_status IN ('ok','partial') GROUP BY 1 ORDER BY 1;
SELECT ext_id, count(*) FROM training_samples WHERE label_status IN ('ok','partial') GROUP BY 1 ORDER BY 1;

-- nhãn chưa map được
SELECT label_raw, count(*) FROM training_samples WHERE label_status='unmapped' GROUP BY 1 ORDER BY 2 DESC;
```

Chạy trực tiếp từ PowerShell (không cần mở DBeaver):

```powershell
docker exec chordsense_postgres psql -U chordsense -d chordsense -c "SELECT label_raw, count(*) AS n FROM training_samples GROUP BY 1 ORDER BY 2 DESC LIMIT 30"
docker exec chordsense_postgres psql -U chordsense -d chordsense -c "SELECT quality_id, count(*) FROM training_samples WHERE label_status IN ('ok','partial') GROUP BY 1 ORDER BY 1"
```

Tên lớp theo id: `ext_id` 0 none, 1 b5, 2 #5, 3 maj7, 4 min7, 5 dom7, 6 add9, 7 b9, 8 9, 9 #9, 10 11, 11 #11, 12 13, 13 b13, 14 alt; `root_id` 0..11 = C, C#, D, D#, E, F, F#, G, G#, A, A#, B. (Bảng đầy đủ kèm tên đã có sẵn trong `data_report.sql`.)

### Nghe audio kèm nhãn (kiểm bằng tai)

`scripts/listen_samples.py` bốc ngẫu nhiên N segment theo bộ lọc, cắt ra file WAV và tạo trang `index.html`: mỗi thẻ có **trình phát audio**, nhãn Harte gốc, hợp âm đã giải mã (vd `D9sus`), nguồn/bài/thời điểm, và nút ✅ đúng / ❌ sai / ❓ không chắc + ô ghi chú. Kết quả lưu trong trình duyệt, bấm **Download CSV** để xuất.

```powershell
# ví dụ: 30 segment #11 của Hooktheory, kèm segment trước/sau để nghe chỗ đổi hợp âm
docker exec -w /opt/airflow chordsense_airflow env PYTHONPATH=/opt/airflow python scripts/listen_samples.py --source hooktheory --ext "#11" --n 30 --context --name sharp11_ht
# → mở DE\data\export\listen\sharp11_ht\index.html bằng trình duyệt
```

Bộ lọc (tùy chọn, kết hợp AND): `--source`, `--split`, `--status ok,partial`, `--root`, `--quality` (major/minor/dominant/dim/sus2/sus4), `--ext` (none, b5, #5, maj7, min7, dom7, add9, b9, nat9, #9, nat11, #11, nat13, b13, alt), `--label-like TXT`. Cùng `--seed` + cùng bộ lọc = cùng mẫu. Xem `python scripts/listen_samples.py --help` hoặc docstring đầu file.

- WAV được chuẩn hóa đỉnh về −1 dBFS **chỉ để nghe** (dữ liệu train không đổi); thêm `--raw-gain` nếu muốn nghe nguyên bản.
- Nhãn = hợp âm ở **giữa** segment 2 s; nốt mở rộng yếu (#11, b13…) hoặc đổi hợp âm trong cửa sổ có thể khó nghe → đây là kiểm mẫu, không phải ground truth.
- Dùng để: soi từng nguồn (nhất là Hooktheory do cộng đồng gán nhãn), và dựng **tập test hợp âm màu kiểm tay** (vài trăm mẫu).

### MongoDB (staging)

Compass: `mongodb://localhost:27017` → db `chordsense_raw` → `raw_audio_jobs`. Chỉ là vùng đệm (con trỏ `wav_path`); xóa được bất cứ lúc nào bằng script reset.

```bash
docker exec chordsense_mongo mongosh chordsense_raw --eval \
  "db.raw_audio_jobs.aggregate([{'\$group': {_id: '\$status', count: {'\$sum': 1}}}])"
```

---

## 8. Troubleshooting

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|-------------|------------------------|------------|
| `duration_mismatch` (JAAH) | YouTube chỉ có bản live/remaster khác thời lượng | Bình thường — bài bị loại để khỏi gán nhãn lệch. Tăng `ALIGN_TOLERANCE_SEC` nếu muốn dễ tính hơn |
| `No YouTube result within …s` | Không bản nào khớp thời lượng | Như trên; bài bị `failed` ngay, không tốn Demucs |
| HTTP 403 / "confirm you're not a bot" | YouTube chặn IP / thiếu cookies | Thêm `config/cookies.txt` (mục 2), giảm `INGEST_BATCH_SIZE` |
| `Demucs timeout` | Đang chạy CPU | Kiểm tra `cuda: True` (mục 3) |
| `CUDA out of memory` | VRAM 4 GB | Tự retry CPU; hoặc giảm `DEMUCS_SEGMENT` (vd 4) |
| `crawl_queue` kẹt `running` | Container bị dừng giữa chừng | `reset_for_recrawl.py --only-failed` |
| DAG Import Error | Lỗi import | `docker exec chordsense_airflow bash -c "cd /opt/airflow && PYTHONPATH=/opt/airflow python3 -c 'from dags.wrapper.ingest_raw_wrapper import pick_urls'"` |

Xem log task: Airflow UI → DAG → task đỏ → **Log**.

Xóa sạch mọi thứ: `docker compose down -v` (mất hết volume, **không** xóa `data/training/` vì là bind-mount) rồi `up -d` + seed lại.

---

## 9. Lịch chạy

| DAG | Schedule | Ghi chú |
|-----|----------|---------|
| `dag_ingest_raw` | `0 */2 * * *` (mỗi 2 giờ) | Tự trigger `dag_process_audio` khi xong |
| `dag_process_audio` | Không có lịch | Được trigger bởi `dag_ingest_raw` |
| `dag_analytics_rollup` | Mỗi ngày 23:30 | Phía hệ thống (mastery) — không thuộc phạm vi data train |

---

## 10. Kiến trúc tóm tắt

```
Local datasets (JAAH annotation, Kaggle WAV, ChoCo…)
        ↓ seed_local_datasets.py
crawl_queue [PostgreSQL]                  ← URL chờ xử lý
        ↓ dag_ingest_raw (mỗi 2h)
annotation → audio (YouTube / local) → WAV tạm /data/tmp/{id}/
        ↓
raw_audio_jobs [MongoDB]                  ← staging (con trỏ tới WAV tạm)
        ↓ dag_process_audio (auto-trigger)
AlignmentCheck → Demucs → Beat → Segment → Annotation → Label → SampleWriter
        ↓                                     ↓
/data/training/audio/*.npy        training_samples [PostgreSQL]   (xóa WAV tạm)
        ↓  scripts/export_dataset.py
DE/data/export/{train,val,test}_audio.npy + _meta.parquet   → Colab
```

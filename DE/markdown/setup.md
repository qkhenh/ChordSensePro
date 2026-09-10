# ChordSensePro DE — Setup Guide

> Hướng dẫn setup từ đầu đến khi pipeline chạy được.

---

## 1. Prerequisites

- Docker Desktop đang chạy
- Git, Python 3.11+
- DBeaver (để xem PostgreSQL)
- MongoDB Compass (để xem MongoDB) — optional

---

## 2. Chuẩn bị Dataset

Download thủ công, bỏ vào đúng thư mục:

```
DE/data/datasets/
  jaah/
    JAAH-v0.1/
      MTG-JAAH-xxx/
        annotations/    ← 113 JSON files
  kaggle/
    archive/
      piano_triads/     ← 432 WAV files
  choco/
    choco/
      choco/
        partitions/     ← JAMS files (clone từ GitHub)
```

### JAAH
1. Vào [zenodo.org/record/1290737](https://zenodo.org/record/1290737)
2. Download `MTG/JAAH-v0.1.zip` (~316MB) → giải nén vào `data/datasets/jaah/`

### Kaggle Piano Triads
1. Vào [kaggle.com/datasets/zakarii/piano-triads-audio](https://www.kaggle.com/datasets/zakarii/piano-triads-audio)
2. Download → giải nén vào `data/datasets/kaggle/archive/piano_triads/`

### ChoCo (optional)
```bash
cd DE/data/datasets/choco
git clone https://github.com/smashub/choco
```

---

## 3. Build & Start

```bash
cd DE
docker compose build airflow   # ~10 phút lần đầu (PyTorch ~2GB)
docker compose up -d
```

Kiểm tra tất cả containers đang healthy:
```bash
docker compose ps
```

Expected output:
```
chordsense_airflow      running (healthy)
chordsense_postgres     running (healthy)
chordsense_mongo        running (healthy)
chordsense_redis        running (healthy)
chordsense_airflow_meta running (healthy)
```

---

## 4. Seed Data vào crawl_queue

```bash
docker exec chordsense_airflow bash -c \
  "cd /opt/airflow && PYTHONPATH=/opt/airflow python3 scripts/seed_local_datasets.py"
```

Expected output:
```
[jaah]   Found 113 tracks
[kaggle] Found 432 files
Total: 545 entries
Done: 545 inserted, 0 already existed
```

---

## 5. Trigger Pipeline

1. Vào **http://localhost:8080** — login `admin / admin`
2. Bật toggle **dag_process_audio** (để sẵn sàng khi được trigger)
3. Bấm **▶ Trigger** DAG `dag_ingest_raw`

Pipeline tự chạy theo thứ tự:
```
dag_ingest_raw
  pick_urls → download_and_stage → send_summary → trigger_dag_process
                                                          ↓ (auto)
dag_process_audio
  pick_pending_jobs → process_and_save → send_summary
```

---

## 6. Check Data — PostgreSQL

Kết nối DBeaver:

| Field    | Value            |
|----------|------------------|
| Host     | `localhost`      |
| Port     | `5432`           |
| Database | `chordsense`     |
| Username | `chordsense`     |
| Password | `chordsense_dev` |

### Query check crawl_queue
```sql
-- Xem tổng quan trạng thái
SELECT source_type, status, count(*)
FROM crawl_queue
GROUP BY source_type, status
ORDER BY source_type, status;
```

Kết quả mong đợi sau seed (trước khi chạy DAG):
```
 source_type | status  | count
-------------|---------|-------
 jaah        | pending |   113
 kaggle      | pending |   432
```

Sau khi DAG chạy:
```
 source_type | status  | count
-------------|---------|-------
 jaah        | done    |   113
 kaggle      | done    |   432
```

### Query check song_analyses (kết quả cuối)
```sql
SELECT status, count(*)
FROM song_analyses
GROUP BY status;
```

---

## 7. Check Data — MongoDB

Dùng MongoDB Compass: `mongodb://localhost:27017`

Database: `chordsense_raw`
Collection: `raw_audio_jobs`

Hoặc dùng CLI:
```bash
docker exec chordsense_mongo mongosh chordsense_raw --eval \
  "db.raw_audio_jobs.aggregate([{'\$group': {_id: '\$status', count: {'\$sum': 1}}}])"
```

Expected sau khi DAG 1 chạy:
```js
[
  { _id: 'pending',    count: 10 },   // đang chờ DAG 2 xử lý
  { _id: 'done',       count: 0  },
  { _id: 'failed',     count: 0  }
]
```

Expected sau khi DAG 2 chạy:
```js
[
  { _id: 'done',    count: 10 }   // processed → saved to PostgreSQL
]
```

---

## 8. Troubleshooting

### DAG Import Errors (Broken DAG)
```bash
# Xem log lỗi
docker exec chordsense_airflow bash -c \
  "cd /opt/airflow && PYTHONPATH=/opt/airflow python3 -c 'from dags.wrapper.ingest_raw_wrapper import pick_urls'"
```

### Xem log task bất kỳ
Trong Airflow UI → click tên DAG → click vào task bị đỏ → **Log**

### Reset toàn bộ & làm lại
```bash
docker compose down -v          # xóa tất cả volumes (data mất hết)
docker compose up -d            # start lại (tables tự tạo qua init.sql)
# Seed lại data
docker exec chordsense_airflow bash -c \
  "cd /opt/airflow && PYTHONPATH=/opt/airflow python3 scripts/seed_local_datasets.py"
```

### Kiểm tra crawl_queue thủ công
```bash
docker exec chordsense_postgres psql -U chordsense -d chordsense \
  -c "SELECT source_type, status, count(*) FROM crawl_queue GROUP BY source_type, status;"
```

---

## 9. Lịch chạy tự động

| DAG                  | Schedule      | Ghi chú                              |
|----------------------|---------------|--------------------------------------|
| `dag_ingest_raw`     | Mỗi 6 giờ    | Tự trigger `dag_process_audio` sau   |
| `dag_process_audio`  | Manual only   | Được trigger bởi `dag_ingest_raw`    |
| `dag_analytics_rollup` | Mỗi ngày 23:30 | Tính mastery từ practice attempts |

---

## 10. Architecture tóm tắt

```
Local datasets (JAAH, Kaggle, ChoCo)
         ↓ seed_local_datasets.py
crawl_queue [PostgreSQL]  ← URLs chờ xử lý
         ↓ dag_ingest_raw (mỗi 6h)
Download audio → WAV
         ↓
raw_audio_jobs [MongoDB]  ← staging jobs
         ↓ dag_process_audio (auto-triggered)
Demucs → Features → Chord timeline
         ↓
song_analyses [PostgreSQL] ← kết quả cuối
```

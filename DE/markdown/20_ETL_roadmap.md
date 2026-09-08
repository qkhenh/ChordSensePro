# ChordSensePro DE — ETL Roadmap (DDD Architecture)

> Viết lại sau khi đọc lại kiến trúc DDD từ `01_ddd_architecture.md` và lấy
> `15_side_project_weather_pipeline.md` làm reference.

---

## 1. Vấn đề với code hiện tại

DAG đang làm **quá nhiều thứ** — vừa là orchestrator, vừa chứa business logic:

```python
# ❌ Hiện tại: dag_song_analysis.py biết quá nhiều
def download_audio(**context):
    url = _pull(ti, "source_url")
    handler = DownloadHandler(data_dir=DATA_DIR)
    result = handler.run(url)           # business logic nằm trong DAG
    ti.xcom_push(key="wav_path", ...)
```

**Đúng theo DDD:** DAG chỉ là adapter mỏng, gọi vào `src/data_ingest/`:

```python
# ✅ Đúng: DAG chỉ là bridge, không chứa logic
def run_pipeline(**context):
    conf = context["dag_run"].conf
    result = run_ingest_pipeline(         # gọi vào src/, không biết chi tiết
        analysis_id=conf["analysis_id"],
        source_url=conf["source_url"],
    )
    return result
```

---

## 2. Kiến trúc đúng — 3 module theo DDD

```
src/
├── shared/             ← Nền tảng chung (đã có, đúng)
│
├── data_loader/        ← Module 1: Tải audio về máy
│   ├── domain/
│   │   ├── models/     ← AudioFile (request model), DownloadResponse
│   │   ├── ports/      ← Downloader Protocol (interface)
│   │   └── services/
│   │       └── audio_dispatcher.py   ← Registry: source → downloader
│   │
│   └── application/
│       ├── entrypoints.py            ← run_data_loader()
│       └── downloaders/
│           ├── youtube_downloader.py ← yt-dlp
│           └── local_file_downloader.py
│
├── data_processing/    ← Module 2: Transform audio → features
│   ├── domain/
│   │   └── models/
│   │       ├── base_handler.py       ← BaseProcessingHandler (ABC, Chain of Responsibility)
│   │       └── processed_audio.py   ← ProcessedAudio (data bag qua pipeline)
│   │
│   └── application/
│       ├── entrypoints.py            ← run_data_processing()
│       ├── pipeline_factory.py       ← Wire handler chain
│       └── pipeline_handlers/
│           ├── separate_handler.py   ← Demucs stem separation
│           ├── beat_handler.py       ← librosa beat tracking
│           ├── segment_handler.py    ← 2s segments + overlap
│           └── feature_handler.py   ← chroma_cqt, chroma_cens, hpss, mel_high
│
├── data_ingest/        ← Module 3: Orchestrator (biết cả loader lẫn processing)
│   ├── domain/
│   │   └── models/
│   │       └── ingest_record.py      ← IngestRecord + IngestStatus
│   │
│   └── application/
│       ├── entrypoints.py            ← run_ingest_pipeline() — DAG gọi vào đây
│       └── ingest_service.py         ← IngestService: gọi loader → processing → save
│
└── audio_processing/   ← (Đổi tên thành data_processing theo pattern)
    (handlers hiện tại sẽ migrate vào đây)
```

> **Quy tắc import:**
> - `data_loader` không biết `data_processing`
> - `data_processing` không biết `data_loader`
> - `data_ingest` biết cả hai — đây là điểm duy nhất kết nối chúng
> - DAG chỉ gọi `data_ingest/entrypoints.py`

---

## 3. ETL — Mapping đúng

### E (Extract) = `data_loader/`

```
Nhiệm vụ: Tải audio từ YouTube hoặc local về /data/tmp/{analysis_id}/
Không validate, không transform. Chỉ tải.

Input:  AudioFile(source_url, analysis_id, source_type)
Output: DownloadResponse(local_path, status, file_size, duration_s)
```

**Cần tạo:**

| File | Nhiệm vụ |
|------|----------|
| `domain/models/audio_file.py` | `AudioFile` model — request model, kế thừa `BaseEntity` |
| `domain/models/download_response.py` | Output chuẩn sau khi tải |
| `domain/ports/downloader.py` | `Downloader` Protocol (interface) |
| `domain/services/audio_dispatcher.py` | Registry: `AudioSource` enum → Downloader |
| `application/downloaders/youtube_downloader.py` | Wrap `DownloadHandler` hiện tại |
| `application/downloaders/local_file_downloader.py` | Copy file local → working dir |
| `application/entrypoints.py` | `run_data_loader(audio_file, dest_path)` |

**MongoDB tracking** (theo pattern `03_data_loader.md`):
```
Trước tải → insert AudioFile(status=PENDING) vào MongoDB
Đang tải  → update status=DOWNLOADING
Tải xong  → update status=SUCCESS / FAILED
```
→ Dùng để dedup: nếu `status=SUCCESS` rồi thì skip.

---

### T (Transform) = `data_processing/`

```
Nhiệm vụ: WAV → stems → beats → segments → features
Không tải file, không lưu DB. Chỉ xử lý trong memory.

Input:  DownloadResponse (local_path tới WAV)
Output: ProcessedAudio (list[AudioSegment] với đủ 4 feature channels)
```

**Handler Chain:**
```
SeparateHandler → BeatHandler → SegmentHandler → FeatureHandler
```

**Cần tạo/refactor:**

| File | Nhiệm vụ |
|------|----------|
| `domain/models/base_handler.py` | `BaseProcessingHandler` — ABC với `handle()` + `set_next()` |
| `domain/models/processed_audio.py` | `ProcessedAudio` — data bag đi qua chain |
| `application/pipeline_factory.py` | Wire chain: `SeparateHandler → Beat → Segment → Feature` |
| `application/pipeline_handlers/*.py` | Move từ `audio_processing/application/handlers/` sang đây |
| `application/entrypoints.py` | `run_data_processing(response: DownloadResponse)` |

---

### L (Load) = `data_ingest/` + repositories

```
Nhiệm vụ 1: Lưu kết quả vào PostgreSQL (song_analyses, segments)
Nhiệm vụ 2: Export features ra /data/features/*.npy cho MERT training
```

**IngestService flow:**
```python
class IngestService:
    def run(self, analysis_id: str, source_url: str) -> IngestRecord:
        record = IngestRecord(analysis_id=analysis_id, status=LOADING)

        # E: tải audio
        audio_file = AudioFile(source_url=source_url, ...)
        download_resp = run_data_loader(audio_file, dest_path=tmp_dir)
        record.status = PROCESSING

        # T: xử lý audio
        processed = run_data_processing(download_resp)
        record.status = SAVING

        # L: lưu kết quả
        await song_analysis_repo.save(processed.to_domain())
        record.status = DONE

        return record
```

**Cần tạo:**

| File | Nhiệm vụ |
|------|----------|
| `domain/models/ingest_record.py` | `IngestRecord` + `IngestStatus` enum |
| `application/ingest_service.py` | `IngestService` — orchestrator |
| `application/entrypoints.py` | `run_ingest_pipeline()` — DAG gọi vào đây |

**L2 — Feature Store (cần trước mid-Oct):**

| File | Nhiệm vụ |
|------|----------|
| `scripts/export_features.py` | Query `song_analyses` → export `/data/features/features.npy` |
| Feature vector shape | `(N_segments, 152)` = chroma_cqt(12) + chroma_cens(12) + hpss(64) + mel_high(64) |

---

## 4. E — Crawler (Cần trước khi crawl)

Thêm một module nhỏ để quản lý URL queue:

```
src/shared/infrastructure/postgres/orm/crawl_queue_orm.py

crawl_queue table:
  id          UUID PK
  source_url  TEXT UNIQUE   ← dedup key
  status      VARCHAR(20)   ← pending / running / done / error
  priority    INT DEFAULT 0
  created_at  TIMESTAMPTZ
```

```
scripts/crawl_urls.py
  Input:  CSV file với youtube_url
  Output: Insert vào crawl_queue (skip nếu đã có)
          Trigger DAG dag_song_analysis qua Airflow REST API

dags/dag_crawl_batch.py
  Schedule: mỗi 30 phút
  Task 1: pick_urls  — SELECT N rows WHERE status='pending'
  Task 2: trigger    — POST /api/v1/dags/dag_song_analysis/dagRuns
```

---

## 5. DAG sau refactor — chỉ là thin adapter

```python
# dags/dag_song_analysis.py — SAU REFACTOR
# DAG chỉ có 1 task, không chứa business logic

from src.data_ingest.application.entrypoints import run_ingest_pipeline

def run_pipeline(**context):
    conf = context["dag_run"].conf or {}
    result = run_ingest_pipeline(
        analysis_id = conf["analysis_id"],
        source_url  = conf["source_url"],
        user_id     = conf["user_id"],
    )
    return result.status

t1 = PythonOperator(task_id="run_pipeline", python_callable=run_pipeline, dag=dag)
```

---

## 6. Trạng thái hiện tại → Target

| Thành phần | Hiện tại | Target |
|---|---|---|
| `data_loader/` | ❌ Không có | Tạo mới |
| `data_processing/` | ❌ Không có (handlers nằm sai chỗ) | Tạo mới + migrate handlers |
| `data_ingest/` | ❌ Không có (logic trong DAG) | Tạo mới |
| `audio_processing/` | ✅ Có handlers | Deprecated → migrate sang `data_processing/` |
| `Downloader` Protocol | ❌ Không có | Tạo mới |
| `AudioDispatcher` Registry | ❌ Không có | Tạo mới |
| `BaseProcessingHandler` | ❌ Không có | Tạo mới |
| MongoDB tracking | ❌ Client setup, không dùng | Implement |
| Feature export `.npy` | ❌ Không có | Tạo mới |
| DAG | ❌ Chứa business logic | Thin adapter |

---

## 7. Timeline refactor + crawl

```
Sep 8 - 10  ── Tạo data_loader/ (E)
                AudioFile, Downloader Protocol, AudioDispatcher
                YoutubeDownloader wrapper
                MongoDB tracking
                crawl_queue table + scripts/crawl_urls.py

Sep 10 - 13 ── Tạo data_processing/ (T)
                BaseProcessingHandler
                Migrate handlers từ audio_processing/
                ProcessedAudio bag
                pipeline_factory.py

Sep 13 - 15 ── Tạo data_ingest/ (orchestrator)
                IngestService
                Refactor DAG thành thin adapter
                Test end-to-end

Sep 15 ──────── Bắt đầu crawl data
                ~500 bài/tuần × 3 tuần = ~1,500 bài

Oct 1 ───────── Tạo export_handler + scripts/export_features.py (L2)

Mid Oct ──────── Train MERT
                 features.npy: ~1,500 × 120 segments = ~180,000 rows × 152 dims
```

---

## 8. Ưu tiên làm ngay

| # | Việc | Tại sao |
|---|------|---------|
| 1 | `crawl_queue` table + `scripts/crawl_urls.py` | Unblock crawl data ngay |
| 2 | `data_loader/` — `Downloader` Protocol + `AudioDispatcher` | Đúng kiến trúc cho E |
| 3 | `data_ingest/` — `IngestService` + entrypoint | DAG gọi đúng chỗ |
| 4 | `data_processing/` — migrate handlers | Đúng kiến trúc cho T |
| 5 | Refactor DAG thành thin adapter | Clean up |
| 6 | `scripts/export_features.py` | Cần trước mid-Oct |

# ChordSensePro DE — Issue Log

> Cập nhật: 2026-09-10
> Tổng: 6 issues

---

## Tổng quan

| # | Mức độ | Vấn đề | File liên quan |
|---|--------|--------|----------------|
| 1 | 🔴 Critical | `ProcessAudioService` dùng sai pipeline → `chord_timeline = []` mãi mãi | `process_audio_service.py` |
| 2 | 🔴 Critical | `RawAudioJob` thiếu field `annotation_path` | `raw_audio_job.py` |
| 3 | 🔴 Critical | Downloaders chỉ lấy audio, không lấy annotation file | `*_downloader.py` |
| 4 | 🟡 Medium | Không có Discovery DAG → phải thêm URL thủ công | DAG chưa tồn tại |
| 5 | 🟡 Medium | Không xóa file tạm sau xử lý → disk leak | `separate_handler.py` |
| 6 | 🟢 Low | Alembic phải chạy tay sau mỗi `docker compose up` | `docker-compose.yml` |

**Thứ tự fix:** #2 → #3 → #1 → #5 → #4 → #6

---

## Issue #1 — 🔴 `ProcessAudioService` dùng sai pipeline

**File:** [`process_audio_service.py` L66-90](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_ingest/application/process_audio_service.py#L66-L90)

**Vấn đề:**
```python
pipeline = build_pipeline()   # ← inference pipeline, không có AnnotationHandler

analysis.mark_done(
    timeline=[],   # hardcode rỗng — không lấy từ annotation file
    sheet={},
    plan=[],
)
```

`build_pipeline()` là inference pipeline (dùng cho bài nhạc mới của user, không có annotation).
Training data phải dùng `build_training_pipeline()` — đã có sẵn trong [`pipeline_factory.py` L31](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_processing/application/pipeline_factory.py#L31), có `AnnotationHandler` để parse JAMS/JSON.

**Hậu quả:** `song_analyses.chord_timeline = []` mãi mãi, dù Demucs chạy đúng và annotation file có sẵn.

**Fix:**
```python
from src.data_processing.application.pipeline_factory import build_training_pipeline

pipeline = build_training_pipeline()   # đổi sang đây

data = ProcessedAudio(audio_file=audio_file, wav_path=wav_path)
data.annotation_path = Path(job.annotation_path) if job.annotation_path else None

analysis.mark_done(
    key=processed.detected_key or "",
    bpm=processed.tempo_bpm or 0.0,
    timeline=processed.chord_timeline,   # lấy từ pipeline, không hardcode
    sheet=processed.chord_sheet or {},
    plan=[],
)
```

---

## Issue #2 — 🔴 `RawAudioJob` thiếu `annotation_path`

**File:** [`raw_audio_job.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_ingest/domain/models/raw_audio_job.py)

**Vấn đề:** `RawAudioJob` không có field `annotation_path`. DAG 1 download annotation file xong không biết lưu đường dẫn ở đâu. DAG 2 nhận job từ MongoDB nhưng không có thông tin annotation → `AnnotationHandler` skip.

**Fix:**

1. Thêm field vào dataclass:
```python
@dataclass
class RawAudioJob:
    annotation_path: str = ""    # đường dẫn file .jams / .json
```

2. Thêm vào MongoDB `save()` và `_to_domain()`:
```python
# save():
doc = { ..., "annotation_path": job.annotation_path }

# _to_domain():
annotation_path=doc.get("annotation_path", ""),
```

---

## Issue #3 — 🔴 Downloaders không lấy annotation file

**Files cần sửa:**
- [`choco_downloader.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_loader/application/downloaders/choco_downloader.py)
- [`jaah_downloader.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_loader/application/downloaders/jaah_downloader.py)
- [`mcgill_downloader.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_loader/application/downloaders/mcgill_downloader.py)
- [`download_response.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_loader/domain/models/download_response.py) — thiếu field `annotation_path`

**Data shape annotation của từng nguồn:**

| Source | Audio | Annotation | Format |
|--------|-------|-----------|--------|
| ChoCo | YouTube (yt-dlp) | GitHub raw JAMS file | JAMS (JSON) |
| JAAH | Zenodo MP3 | Zenodo JSON (cùng archive) | JSON |
| McGill | YouTube (yt-dlp) | GitHub JAMS | JAMS (JSON) |

**JAMS format (ChoCo / McGill):**
```json
{
  "file_metadata": {
    "title": "Let It Be",
    "identifiers": { "youtube_id": "dQw4w9WgXcQ" }
  },
  "annotations": [{
    "namespace": "chord",
    "data": [
      {"time": 0.0, "duration": 2.5, "value": "C"},
      {"time": 2.5, "duration": 2.5, "value": "Am"},
      {"time": 5.0, "duration": 2.5, "value": "F"},
      {"time": 7.5, "duration": 2.5, "value": "G"}
    ]
  }]
}
```

**JAAH JSON format:**
```json
{
  "title": "Autumn Leaves",
  "artist": "Miles Davis",
  "chords": [
    {"timestamp": 0.0, "duration": 4.0, "chord": "Cm7"},
    {"timestamp": 4.0, "duration": 4.0, "chord": "F7"},
    {"timestamp": 8.0, "duration": 4.0, "chord": "BbMaj7"}
  ]
}
```

**Fix:**
```python
# download_response.py
@dataclass
class DownloadResponse:
    annotation_path: Path | None = None   # thêm field này

# Mỗi downloader trả về annotation_path:
return DownloadResponse.ok(audio_file, wav_path, annotation_path=annotation_path)
```

---

## Issue #4 — 🟡 Không có Discovery DAG

**Vấn đề:** Không có DAG nào tự fetch danh sách track từ dataset. Phải thêm URL thủ công vào `crawl_queue`.

**Tất cả sources đều là STATIC data** (không phải live data):

| Source | Số track | Cập nhật | Index URL |
|--------|----------|----------|-----------|
| ChoCo | ~20,000 | Hiếm | `raw.githubusercontent.com/smashub/choco/main/choco/index.json` |
| JAAH | 113 | Không | `zenodo.org/api/records/1290737` |
| McGill | 1,300 | Không | GitHub JAMS files |

**Nên crawl theo giai đoạn** (máy 120GB disk):

| Giai đoạn | Dataset | Số bài | Disk tạm | Disk sau xử lý |
|-----------|---------|--------|----------|----------------|
| Prototype | JAAH | 113 | ~2GB | ~6MB |
| V1 | JAAH + 500 ChoCo | ~600 | ~10GB | ~30MB |
| Full | Tất cả | ~22,000 | — | ~1GB |

> Audio tạm (~105MB/bài: WAV + 6 stems + segments) bị **XÓA** sau xử lý. Chỉ giữ features trong Postgres (~50KB/bài).

**Fix:** Tạo `dag_discover_urls.py`:
1. Fetch index từ GitHub/Zenodo API
2. Extract audio URL + annotation URL
3. `INSERT INTO crawl_queue ... ON CONFLICT DO NOTHING` — tự dedup

---

## Issue #5 — 🟡 Không xóa file tạm sau xử lý

**File:** [`process_audio_service.py`](file:///d:/qkhenh/Project/ChordSensePro/DE/src/data_ingest/application/process_audio_service.py)

**Vấn đề:** Sau DAG 2 xử lý xong, các file này vẫn còn trên disk:
```
/data/tmp/{job_id}/
    song.wav           ~15MB   ← không xóa
    separated/
        piano.wav      ~15MB   ← không xóa
        guitar.wav     ~15MB
        other.wav      ~15MB
        ...
    segments/          ~50MB   ← không xóa
```
1,000 bài xử lý → ~100GB bị chiếm dù không cần nữa.

**Fix:** Thêm cleanup sau khi lưu Postgres thành công:
```python
import shutil
# Cuối process_audio_service.py, sau await repo.update_status(job.id, RawJobStatus.DONE):
shutil.rmtree(wav_path.parent, ignore_errors=True)
```

---

## Issue #6 — 🟢 Alembic phải chạy tay

**File:** [`docker-compose.yml`](file:///d:/qkhenh/Project/ChordSensePro/DE/docker-compose.yml)

**Vấn đề:** Sau `docker compose up`, phải tự chạy migration thủ công.

**Fix option A** — Thêm vào Airflow startup command:
```yaml
command: >
  bash -c "
    cd /opt/airflow &&
    PYTHONPATH=/opt/airflow alembic upgrade head &&
    airflow standalone
  "
```

**Fix option B** — Bỏ Alembic, dùng `Base.metadata.create_all()` tự động (đơn giản hơn cho dev).

---

## Luồng đầy đủ sau khi fix

```
ChoCo / JAAH / McGill (GitHub / Zenodo — static data)
    ↓ dag_discover_urls [fix #4]
crawl_queue: { audio_url, annotation_url, source_type, status: pending }
    ↓ dag_ingest_raw (DAG 1)
    ├── Download audio → WAV           [fix #3]
    └── Download annotation → .jams    [fix #3]
MongoDB raw_audio_jobs: { wav_path, annotation_path, crawl_queue_id }   [fix #2]
    ↓ dag_process_audio (DAG 2)
    Separate → Beat → Segment → Annotation → Augment → Feature   [fix #1]
    └── Cleanup: xóa WAV + stems + segments khỏi disk             [fix #5]
PostgreSQL song_analyses: {
    tempo_bpm:      120.5,
    chord_timeline: [{ start: 0.0, end: 2.5, chord: "C" }, ...],
    chord_sheet:    { verse: ["C", "Am", "F", "G"] },
    learning_plan:  []   ← sau khi có MERT model
}
    ↓
Train MERT v1 330M + LoRA (song_analyses làm ground truth)
    ↓
Deploy: predict chord cho bài nhạc mới của user (không có annotation)
```

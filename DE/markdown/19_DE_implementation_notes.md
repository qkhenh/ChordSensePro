# ChordSense Pro — DE Implementation Notes

> Ghi lại các quyết định thiết kế, bug fixes, và design decisions phát sinh trong quá trình code.
> Spec chính: `18_DE_AI_chordsense.md`

---

## Project Structure

```
DE/
├── src/
│   ├── shared/                    # Shared domain + infrastructure
│   ├── audio_processing/          # Pipeline handlers + domain
│   ├── song_analysis/             # Song analysis domain
│   ├── student_analytics/         # Analytics rollup
│   └── api/                       # FastAPI routes
├── tests/
├── dags/                          # Airflow DAGs
├── markdown/
├── pyproject.toml
├── pyrightconfig.json
└── docker-compose.yml
```

> `curriculum/` module đã bị loại bỏ. Project root là `DE/` (không phải `DE/chordsense/`).

---

## Package Manager

- `uv` — dùng `uv sync`, `uv run pytest`
- Rebuild venv nếu lỗi "trampoline failed": `Remove-Item -Recurse -Force .venv; uv sync`

---

## 1. `shared/` Layer ✅

### `domain/value_objects.py`

**Design decisions:**
- `RootNote`: sharp làm canonical form. Enharmonic disambiguation ở post-processing.
- `ChordQuality`: `aug` bị loại — quy về `major + #5`. 5 qualities: major/minor/dominant/dim/sus.
- `ChordExtension`: 15 classes. Naming theo Chordie.com.
- `dim` + extension cần xử lý đặc biệt trong `ChordLabel.display_name`:
  - `dim + min7` → `"m7b5"` (half-diminished)
  - `dim + dom7` → `"dim7"` (fully diminished)

**TODO:** `DatasetNotationMapper` — map alias raw dataset ("m7", "M7", "mi7"...) về canonical enum.
Vị trí: `src/audio_processing/application/mappers/harte_mapper.py`

### `domain/base_model.py`

- `BaseEntity`: id UUID, created_at UTC. `__eq__` và `__hash__` dựa trên `id`.
- `AuditedEntity`: thêm `updated_at` + `touch()`.

### `domain/processing_result.py`

- Railway-oriented: `Ok(value)` hoặc `Err(message)`.
- Dùng xuyên suốt mọi handler.

### `infrastructure/settings/config.py`

- `get_settings()` = singleton via `@lru_cache(maxsize=1)`.
- `from __future__ import annotations` phải là dòng đầu tiên.

### `infrastructure/postgres/client.py`

- `get_session()` = async context manager, auto commit/rollback.
- `expire_on_commit=False`.

### `infrastructure/postgres/base_repo.py`

- Unit of Work: `flush()` trong repo, `commit()` ở context manager.

### `infrastructure/mongo/client.py`

- Global singleton (không dùng `@lru_cache`) vì cần `close_mongo()`.

---

## 2. `audio_processing/` Layer ✅

### `domain/models/audio_segment.py`

```
AudioSegment (BaseEntity)
  ├── song_id: str
  ├── segment_idx: int
  ├── start_ms / end_ms: float  (milliseconds)
  ├── sample_rate: int = 44100  # DownloadHandler phải resample trước
  ├── audio_data: bytes
  ├── chroma_cqt: list[float] | None   # 12-dim (Constant-Q Transform)
  ├── chroma_cens: list[float] | None  # 12-dim (noise-robust)
  └── mel_high: list[float] | None     # 64-dim, fmin=2000Hz (detect extensions)
```

### `application/handlers/download_handler.py`

- Output: WAV 44100Hz mono — ffmpeg post-processing trong yt-dlp.
- Trả `ProcessingResult[Path]`, không bytes (file WAV ~50MB không load RAM).
- `download_temp()` context manager: tự xóa file temp sau khi block kết thúc.

```python
# Dùng trong DAG:
with handler.download_temp(url) as result:
    audio_path = result.unwrap()
    # xử lý tiếp (separate, beat track, segment...)
# file WAV đã bị xóa tự động ở đây
```

**Luồng 1 cần thêm (TODO):**
- Check `song_analyses` cache trước khi download.
- Async via Airflow DAG — user nhận `job_id`, poll kết quả.
- Raw WAV → MongoDB `song_audio_cache` TTL 7 ngày, file temp xóa ngay.

### `application/handlers/separate_handler.py` ✅

- Model: **`htdemucs_6s`** (6-stem) — updated from `htdemucs` (4-stem)
- Fallback chain: `piano` → `guitar` → `other` (handles songs without explicit piano stem)
- Runs via `subprocess` — Demucs loads ~800MB model weights, subprocess isolates memory
- Model weights auto-downloaded on first run to `~/.cache/torch/hub/`
- Install: `demucs>=4.0.0` added to `[project.optional-dependencies] audio` in `pyproject.toml`

### `application/handlers/beat_handler.py` ✅

- **madmom replaced with `librosa.beat.beat_track()`** — madmom unmaintained since 2022, build fails on Python 3.12+
- `librosa.beat.beat_track()` uses dynamic programming beat tracker — sufficient for thesis
- `estimate_bpm(beat_times)` — calculates avg BPM from inter-beat intervals
- madmom removed from `pyproject.toml` audio deps entirely

### `application/handlers/segment_handler.py` ✅

- Slices audio into 2s segments with 0.5s overlap (`SEGMENT_DURATION=2.0`, `OVERLAP=0.5`)
- `_snap_to_beat()` — snaps segment start to nearest beat within 0.1s tolerance
- `_to_wav_bytes()` — converts numpy array to WAV bytes via `io.BytesIO` (no temp file on disk)
- Returns `list[AudioSegment]` with `audio_data` filled, features still `None`

### `application/handlers/feature_handler.py` ✅

- Resamples to `TARGET_SR=22050` (librosa default) for feature extraction
- Extracts 3 feature vectors per spec section 3.3:
  - `chroma_cqt`  (12-dim): `librosa.feature.chroma_cqt` — root note detection
  - `chroma_cens` (12-dim): `librosa.feature.chroma_cens` — noise-robust version
  - `mel_high`    (64-dim): `librosa.feature.melspectrogram(fmin=2000, fmax=8000)` — extension detection
- `run_batch(segments)` — convenience method for processing a list
- Returns new `AudioSegment` instance (immutable pattern) with features populated

---

## 3. IDE / Tooling

- `pyrightconfig.json`: `extraPaths: ["."]` → resolve `from src.shared...` không cần `# pyrefly: ignore`.
- `src/__init__.py`: thêm để IDE nhận diện package.

---

## 4. Database Schema

### PostgreSQL

```sql
song_analyses (id, student_id, source_url, song_title, detected_key,
               tempo_bpm, chord_timeline JSONB, chord_sheet JSONB,
               learning_plan JSONB, status, created_at)

chord_attempts (id, student_id, session_id, target_chord, detected_chord,
                root_conf, quality_conf, extension_conf, is_correct, timestamp)

user_chord_mastery (user_id, chord, date,
                    accuracy_today, rolling_accuracy_3d, is_mastered,
                    PRIMARY KEY (user_id, chord, date))
                    -- Fix: student_id → user_id (bug trong spec cũ đã sửa)
```

### MongoDB

| Collection | TTL | Nội dung |
|---|---|---|
| `raw_audio` | 30 ngày | Binary WAV từ user |
| `song_audio_cache` | 7 ngày | Demucs output |
| `mel_spectrograms` | 90 ngày | Matrix per segment |
| `training_samples` | — | Processed samples |

---

## 5. Deployment

### 5.1 DE — Local Setup (thesis demo)

**Prerequisites:**
- Docker Desktop
- CUDA 12.x + cuDNN (for RTX 3050 GPU acceleration)
- uv (package manager)

**Start infrastructure:**
```bash
# PostgreSQL + MongoDB + Redis + Airflow
docker-compose up -d

# Install DE dependencies
uv sync --extra audio --extra db --extra api
```

**Run pipeline manually (no Airflow):**
```bash
uv run python main.py analyze "https://youtube.com/..."
```

**Expose to GVHD for demo (no server needed):**
```bash
# Install ngrok → https://ngrok.com
ngrok http 8000
# Gives: https://abc123.ngrok.io → accessible from anywhere
```

**GPU acceleration:**
```bash
# Verify CUDA available
uv run python -c "import torch; print(torch.cuda.is_available())"

# Demucs auto-uses GPU if CUDA available
# MERT inference: model.to('cuda') in inference handler
```

**Expected performance on RTX 3050:**
- Demucs htdemucs_6s: ~30-45s per song
- MERT inference: ~0.3-0.5s per 2s segment
- Full pipeline (4-min song): ~2 minutes total

---

### 5.2 BE — Deployment Recommendations (for team)

> DE không chịu trách nhiệm phần này. Ghi ở đây để tham khảo khi tích hợp.

**For thesis demo:**
```
FastAPI (uvicorn) → localhost:8000
ngrok tunnel → public URL for GVHD
Docker Compose: postgres + mongo + redis chạy local
```

**For small production (post-thesis, optional):**

| Service | Recommendation | Cost |
|---------|---------------|------|
| App server | Render.com (auto-deploy from GitHub) | $7/month |
| PostgreSQL | Render Postgres or Supabase free tier | $0-7/month |
| MongoDB | MongoDB Atlas M0 free | $0 |
| Redis | Upstash free tier | $0 |
| GPU inference | Keep on local machine, expose via ngrok | $0 |
| Total | | ~$7-14/month |

**Architecture for scale (if ever needed):**
```
User request → FastAPI → Airflow DAG (async)
                              ↓
                    DownloadHandler (yt-dlp)
                              ↓
                    SeparateHandler (Demucs, GPU)
                              ↓
                    FeatureHandler (librosa)
                              ↓
                    MERT inference (ONNX, GPU or CPU)
                              ↓
                    Save → PostgreSQL + MongoDB
                              ↓
              User polls GET /song/analyze/{job_id} → result
```

**On-device recommendation (future):**
- Export MERT + heads → ONNX → INT8 quantized (~150-200MB)
- Practice Mode (real-time 2s) can run in browser via ONNX Runtime Web
- Song Analysis still server-side (Demucs too heavy for browser)


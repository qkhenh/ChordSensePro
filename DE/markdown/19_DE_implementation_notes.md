# ChordSense Pro — DE Implementation Notes

> Ghi lại các quyết định thiết kế, bug fixes, và design decisions phát sinh trong quá trình code.
> Spec chính: `18_DE_AI_chordsense.md`

---

## Project Structure

```
DE/
├── src/
│   ├── shared/                    # Domain value objects, DB clients, dataset_split
│   ├── data_loader/               # E: downloaders (YouTube/yt-dlp, local, Kaggle…) + AudioDispatcher
│   ├── data_processing/           # T: handler chain, annotation parsers, HarteMapper
│   ├── data_ingest/              # Orchestration services (IngestRawService, ProcessAudioService), crawl_queue, Mongo staging
│   ├── training_data/             # L: TrainingSample + TrainingSampleRepository (training_samples table)
│   ├── song_analysis/             # (system side — parked) song analysis domain
│   └── user_analytics/            # (system side — parked) analytics rollup
├── tests/
├── dags/                          # Airflow DAGs + wrapper/ (thin adapters)
├── scripts/                       # seed_local_datasets, reset_for_recrawl, export_dataset
├── init/init.sql                  # Postgres schema (fresh volume only)
├── markdown/
├── pyproject.toml
├── pyrightconfig.json
└── docker-compose.yml
```

> `curriculum/` module đã bị loại bỏ. Project root là `DE/` (không phải `DE/chordsense/`).
> `audio_processing/` đã được migrate sang `data_processing/` — các section 2 bên dưới dùng tên module cũ, đọc kèm **section 6**.

---

## Package Manager

- `uv` — dùng `uv sync`, `uv run pytest`
- Rebuild venv nếu lỗi "trampoline failed": `Remove-Item -Recurse -Force .venv; uv sync`

---

## 1. `shared/` Layer ✅

### `domain/value_objects.py`

**Design decisions:**
- `RootNote`: sharp làm canonical form. Enharmonic disambiguation ở post-processing.
- `ChordQuality`: `aug` bị loại — quy về `major + #5`. 6 qualities: major/minor/dominant/dim/sus2/sus4.
- `ChordExtension`: 15 classes. Naming theo Chordie.com.
- `dim` + extension cần xử lý đặc biệt trong `ChordLabel.display_name`:
  - `dim + min7` → `"m7b5"` (half-diminished)
  - `dim + dom7` → `"dim7"` (fully diminished)

**✅ DONE (2026-10-19):** `HarteMapper` — map nhãn Harte/JAMS thô → `(root_id, quality_id, ext_id)` + `LabelStatus`.
Vị trí: `src/data_processing/application/mappers/harte_mapper.py` (xem section 6.2).

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
- Harmonic mix: `piano` + `guitar` + `other` được cộng thành `harmonic.wav` (peak ≤ 0.99; bỏ drums/bass/vocals). Thay cho fallback cũ vốn không bao giờ chạy vì `htdemucs_6s` luôn ghi `piano.wav`.
- Runs via `subprocess` — Demucs loads ~800MB model weights, subprocess isolates memory
- Model weights auto-downloaded on first run to `~/.cache/torch/hub/`
- Install: `demucs>=4.0.0` is listed in `requirements.txt` (the old `pyproject.toml` was removed 2026-10-23)

### `application/handlers/beat_handler.py` ✅

- **madmom replaced with `librosa.beat.beat_track()`** — madmom unmaintained since 2022, build fails on Python 3.12+
- `librosa.beat.beat_track()` uses dynamic programming beat tracker — sufficient for thesis
- `estimate_bpm(beat_times)` — calculates avg BPM from inter-beat intervals
- madmom is not a dependency (removed from the project entirely)

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

> **Cập nhật 2026-10-19:** trong training ETL MongoDB chỉ có **một** collection `raw_audio_jobs` (staging: `wav_path`, `annotation_path`,
> `crawl_queue_id`, `status`). Các collection bên dưới là thiết kế gốc cho phía hệ thống, chưa được cài. Dữ liệu train nằm ở Postgres
> `training_samples` + file `.npy` (section 6).

| Collection | TTL | Nội dung |
|---|---|---|
| `raw_audio` | 30 ngày | Binary WAV từ user |
| `song_audio_cache` | 7 ngày | Demucs output |
| `mel_spectrograms` | 90 ngày | Matrix per segment |
| `training_samples` | — | *(không dùng — đã chuyển sang Postgres)* |

---

## 5. Deployment

### 5.1 DE — Local Setup (thesis demo)

**Prerequisites:**
- Docker Desktop
- CUDA 12.x + cuDNN (for RTX 3050 GPU acceleration)
- Python 3.11 (local tests only; the runtime is the Airflow Docker image)

**Start infrastructure:**
```bash
# PostgreSQL + MongoDB + Redis + Airflow
docker-compose up -d

# Local dev/test dependencies (pytest, ruff) — runtime deps are baked into the Docker image
pip install -r requirements-dev.txt
```

**Run pipeline manually (no Airflow):**
```bash
python main.py analyze "https://youtube.com/..."
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


---

## 6. Training ETL — như đã cài (2026-10-19)

> Mục tiêu duy nhất: **crawl + xử lý data để train AI**. Chi tiết thay đổi: [`change_log.md`](change_log.md). Hướng dẫn chạy: [`setup.md`](setup.md).

### 6.1 Luồng dữ liệu

```
crawl_queue ──► IngestRawService ──► Mongo raw_audio_jobs ──► ProcessAudioService ──► training_samples + .npy
 (Postgres)     annotation → duration       (staging)           training pipeline        (Postgres)   (volume)
                / section → audio
                (YouTube / local)
```

DAG chạy mỗi 2 giờ, `INGEST_BATCH_SIZE=100` URL/lượt, lấy `priority` cao trước (hooktheory 6–7 → guitarset 5 → swd 4 → choco 3 → billboard 1 → còn lại 0).
Mỗi nguồn chỉ khác ở 4 điểm: cách lấy audio, parser nhãn, kiểm tra thời lượng/đoạn, có chạy Demucs không — xem bảng trong [`data_flow.md`](data_flow.md).

Chain: `AlignmentCheck → Separate → Beat → Segment → Annotation → Label → SampleWriter`
(`build_training_pipeline` trong `pipeline_factory.py`; chain inference `build_pipeline` không đổi).

| Handler | Việc | Ghi chú |
|---------|------|---------|
| `AlignmentCheckHandler` | JAAH/ChoCo: so duration WAV với annotation · Hooktheory: so độ dài WAV với độ dài đoạn đã yêu cầu (`SECTION_CHECKED`) | lệch > `ALIGN_TOLERANCE_SEC` (3 s) → fail `duration_mismatch` |
| `SeparateHandler` | Demucs `htdemucs_6s` → trộn stem piano+guitar+other | GPU nếu có; nguồn sạch (`kaggle`/`local`/`pianoteq`) **bỏ qua Demucs** |
| `BeatHandler` | librosa beat tracking | nguồn sạch không có beat → không lỗi |
| `SegmentHandler` | cửa sổ 2 s, bước 1.5 s (overlap 0.5 s), start snap theo beat | giữ nguyên |
| `AnnotationHandler` | nhãn Harte theo midpoint của segment | Kaggle: nhãn từ tên file; không có chord → `N`; Hooktheory: segment ngoài `labelled_range` (±0.25 s) giữ `None` → bị bỏ, không thành `N` |
| `LabelHandler` | nhãn → `root_id/quality_id/ext_id/label_status` | dùng `HarteMapper` |
| `SampleWriterHandler` | 24 kHz int16 `(N,48000)` → `.npy`; điền `sample_rows` | không truy cập DB |

### 6.2 `HarteMapper`

`map_harte(label) -> MappedChord(status, root, quality, extension)`; id lấy theo thứ tự enum trong `shared/domain/value_objects.py`
(khớp đánh số trong spec). Quy ước: `dim7` = (dim, dom7), `hdim7` = (dim, min7), `aug` = (major, #5).

| `LabelStatus` | Ý nghĩa | Ví dụ |
|---------------|---------|-------|
| `ok` | đủ 3 head | `C:maj7`, `Bb:7(b9)`, `G:sus4` |
| `partial` | có root, thiếu quality/extension trong taxonomy | `C:6`, `C:5` (power chord) |
| `no_chord` | không có lớp "N" | `N`, `X` |
| `unmapped` | chưa hiểu nhãn — **vẫn lưu** để map lại sau, khỏi crawl lại | nhãn lạ |

### 6.3 Bảng `training_samples`

`(song_id, segment_idx)` UNIQUE. Cột: nguồn (`source_type`, `source_url`), vị trí (`start_ms`, `end_ms`), audio (`audio_path` tương đối so với
`TRAINING_DIR`, `audio_idx` = chỉ số dòng trong `.npy`, `sample_rate`=24000), nhãn (`label_raw`, `label_status`, `root_id`, `quality_id`, `ext_id`),
`split` (train/val/test), `align_status`. DDL ở `init/init.sql` và `TrainingSampleRepository.ensure_table()`.
Ghi theo bài hát: `replace_song()` = DELETE + INSERT trong một transaction → chạy lại không sinh bản sao.

### 6.4 Quyết định & điểm cần nhớ

- **Lưu bản gốc (không augment)**; split theo `hash(source_url)` (70/20/10) quyết định lúc ingest.
  Kaggle: mỗi file WAV là một "bài" riêng nên các octave/variant của cùng một hợp âm có thể rơi vào split khác nhau (rò rỉ nhẹ, chấp nhận được cho nguồn bổ trợ).
- Biên độ audio **không chuẩn hóa** (stem Demucs có thể nhỏ) → normalize ở phía train.
- JAAH chỉ kiểm tra duration (option B). Căn chỉnh beat (option C) chưa làm.
- Stem harmonic nay là bản trộn piano+guitar+other (đã sửa); 27 bài JAAH cũ đã được xếp lại hàng đợi để xử lý lại với stem mới (2026-10-19).
- **Hooktheory** (`source_type=hooktheory`): `seed_local_datasets.py --source hooktheory --hooktheory-scope rare|all` đọc `Hooktheory.json.gz` (Sheet Sage), đổi hợp âm xếp chồng sang nhãn Harte interval-list
  (`annotation_parsers/hooktheory.py`), ghi JSON annotation (thời gian theo đoạn, `section` = vùng nhãn ± 3 s). `HooktheoryDownloader` chỉ tải đúng đoạn đó bằng yt-dlp `download_ranges`;
  `source_url = youtube?v=<id>#ht=<id>` (phần `#ht=` bị bỏ khi hash split và khi tải) để nhiều clip trong cùng video không trùng khóa queue nhưng vẫn chung split.
- `reset_for_recrawl.py` có thể reset **một nguồn** (`--source jaah --truncate-samples --priority 8`): chỉ xóa `training_samples` + file `.npy` của nguồn đó và xếp lại các dòng `crawl_queue` tương ứng.
- `song_analyses` không còn được ghi (dữ liệu cũ giữ nguyên).

### 6.5 Việc còn mở (chuyển từ `issue_log.md` đã xóa 2026-10-23)

- **Stem piano của `htdemucs_6s`**: model luôn xuất đủ 6 stem nên `piano.wav` luôn tồn tại; với jazz không có piano (sax, trumpet…) stem này có thể chỉ là phần rò rỉ. Nghe thử 5–10 bài JAAH (`scripts/listen_samples.py` chỉ nghe segment đã xử lý; muốn so stem cần nghe file `separated/` trước khi bị xóa); nếu kém thì chuyển sang mix `other+piano+guitar` (đang dùng) hoặc toàn bộ mix.
- **Không có DAG discover URL**: nguồn nạp qua `seed_local_datasets.py` (đủ dùng, vì mọi dataset đều tĩnh).
- **`crawl_queue` kẹt `running`** khi container bị dừng giữa chừng: chưa có cơ chế tự phục hồi → `UPDATE crawl_queue SET status='pending' WHERE status='running'` (chỉ chạy khi cả hai DAG rảnh).
- **39 nhãn `unmapped`** đều chứa bậc `b2` (vd `F:(b2,b5,b7)`), chưa có quy tắc.
- ~~Hàng có tổ hợp (quality, ext) không hợp lệ~~ → **đã xử lý (2026-10-23):** `export_dataset.py` tự loại (233 / 49.840 hàng, 0,47 %; vd dim+b9 61, sus4+b5 22, sus2+nat13 19). DB vẫn giữ nguyên các hàng này (chỉ export bỏ). `--keep-invalid-combos` để giữ lại.

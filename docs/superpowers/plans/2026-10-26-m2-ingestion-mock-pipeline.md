# M2 — Audio & Link Ingestion + Mock Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Người dùng đã đăng nhập tải file nhạc lên hoặc dán URL công khai (hoặc bấm bài demo), thấy stepper chạy theo thời gian thực, rồi tự mở chart mock đã lưu vào thư viện; mọi nhánh lỗi (413/415/422/429/timeout/AI_FAILURE) có test BE + hiển thị FE.

**Architecture:** Bounded context mới `DE/src/ingestion/{domain,application,infrastructure}` đứng sau các port (`MediaResolverPort`, `JobRunnerPort`, `AnalysisEngine`, `JobEvents`); HTTP mỏng ở `DE/src/api/routers/`. Gateway chỉ làm việc rẻ đồng bộ (kiểm tra kích thước/magic/ffprobe, SSRF, trừ quota, tạo job) rồi trả 202; chuẩn hoá WAV, tính `content_hash`, phân tích và lưu chart chạy trong `JobExecutor` (in-process ở dev, Celery ở 2.6). FE thêm `web/src/analyze/` (Source Selector, `useJob` WS→polling, JobTracker) trên Shell của 1.5.

**Tech Stack:** FastAPI + SQLAlchemy Core/asyncpg + Alembic (đã có), `ffprobe`/`ffmpeg` qua subprocess, `httpx` (SafeFetcher), `dnspython` không cần (dùng `asyncio.get_running_loop().getaddrinfo`), Celery + Redis (2.6), React 19 + TanStack Query + MSW + Playwright (đã có).

**Spec:** `.agents/context/implementation-plan.md` §2 (contract, mock), §3 "MILESTONE 2" (bảng 2.1–2.8, TDD, DoD); `markdown/use_case_specifications.md` UC1 (AF1–AF2, EF1–EF4); `latex_last_project/main.tex` NFR-06, NFR-12, NFR-16, FR-20.

## Quyết định cần bạn xác nhận trước khi chạy (mặc định = cách plan đang viết)

| # | Vấn đề | Mặc định của plan | Vì sao |
| :-- | :-- | :-- | :-- |
| D1 | `content_hash` = sha256 của WAV chuẩn hoá chỉ biết **sau** khi decode (1–2 s) ⇒ không thể có trong POST ≤ 300 ms (NFR-06), nhưng UC1 bước 4 đòi "trùng nội dung thì không trừ quota" | Hai khoá: `source_fingerprint` = sha256 byte upload (tính miễn phí khi stream) tra bảng mới `source_fingerprints → content_hash` ngay lúc POST; URL/encode khác thì phát hiện trùng **sau** chuẩn hoá trong job và **hoàn** quota. Contract không đổi | Giữ nguyên định nghĩa contract, vẫn đáp ứng 300 ms và quy tắc quota |
| D2 | Nút "huỷ" job (plan gốc) | **Hoãn**: contract chưa có state `cancelled`; FE có "Dừng theo dõi" + "Thử lại" | Thêm state = nâng contract lên 1.1 |
| D3 | Chart API (3.1) | Kéo sớm **chỉ** `GET /charts/{id}` (owner-only) + trang `/charts/:id` tạm để DoD "tự mở chart" thực hiện được; M3.1 mở rộng (list, ETag, Range audio) | DoD của M2 cần mở được chart |
| D4 | Timeout 90 s (EF1) | Đặt trong `JobExecutor` (dùng chung mọi runner), 2.6 chỉ thêm giới hạn cứng của Celery | Chạy đúng cả khi chưa có Celery |
| D5 | Lưu audio | Giữ WAV chuẩn hoá của chart `file`/`demo` ở `DATA_DIR/audio/<content_hash>.wav`; file upload gốc và audio URL **xoá ngay** sau xử lý | Q2 đã chốt: không giữ audio URL; M3 cần phát file |
| D6 | Giới hạn "100 MB" | `100 * 1024 * 1024` byte (100 MiB); 10 phút = 600 000 ms | NFR-12 không nói MB hay MiB — chọn MiB, ghi vào memory.md |
| D7 | Auth WebSocket | Subprotocol: `new WebSocket(url, ['bearer', accessToken])` (token không vào URL/log) | Plan gốc cho phép query hoặc subprotocol |

## Global Constraints

- Mọi lệnh có tiền tố `rtk` (`rtk uv run pytest -v`, `rtk npm test -- --run`); chạy từ `DE/` hoặc `web/`.
- TDD: test viết trước, **chạy thấy fail đúng lý do**, rồi mới viết code; commit nguyên tử trên nhánh `tienkee`, không push.
- Mỗi commit kết thúc bằng dòng `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Audio chuẩn: WAV **44.1 kHz mono** 16-bit (`-ar 44100 -ac 1`); thời gian là số nguyên **ms**.
- Giới hạn: `MAX_UPLOAD_BYTES = 104_857_600`, `MAX_DURATION_MS = 600_000`; định dạng: mp3, wav, flac, m4a, ogg. Đúng giới hạn thì nhận, +1 byte / +1 ms thì 413.
- Mã lỗi/HTTP lấy từ `ERROR_HTTP_STATUS` (`DE/src/analysis_contract/models.py`): 415, 413, 413, 422, 422, 429, 504, 502. Body lỗi luôn `{code, message}`.
- `POST` nhận job trả **202 + `job_id` p95 ≤ 300 ms** trên mock (NFR-06); đổi trạng thái job đẩy tới client ≤ 1 s; polling dự phòng 2 s.
- Job: `queued → processing → done | failed`, `queued → done|failed` cho cache-hit/từ chối; `done`/`failed` là trạng thái cuối, `progress` không giảm.
- Quota trừ lúc nhận job, hoàn **đúng một lần** khi failed/timeout (`jobs.quota_charged`), cache-hit không trừ.
- Truy cập job/chart của người khác trả **404** (không 403) để không lộ sự tồn tại.
- Ruff (`rtk uv run ruff check . && rtk uv run ruff format --check .`) và `rtk npm run lint && rtk npm run build` xanh sau mỗi task; contract đổi thì `rtk uv run python -m src.analysis_contract.export` rồi `rtk npm run gen:api` (test drift sẽ đỏ nếu quên).
- Không commit `.wav/.mp3`; audio demo sinh bằng script và gitignored.

## Review Focus (không task nào được phép bỏ qua; mỗi dòng có test ở task sở hữu)

1. **File rỗng / đuôi đúng nhưng nội dung không phải audio / ffprobe không trả duration** → 415, không bao giờ 500 và không tạo job (T4, T7).
2. **Tên file độc hại** (`../../x.mp3`, tên rất dài, unicode, `fail-ai` giả để kích hoạt mock) → file luôn lưu dưới tên sinh ra; title được làm sạch ≤ 300 ký tự (T4, T7).
3. **Job treo `processing`**: exception bất ngờ (không phải `AnalysisError`), server khởi động lại giữa chừng, `enqueue` thất bại → job thành `failed(ai_failure)` + hoàn quota đúng một lần, không bao giờ kẹt (T6, T8).
4. **Hai request đồng thời ở biên quota, và hoàn quota qua nửa đêm UTC** → không vượt 5, hoàn về đúng ngày đã trừ (`jobs.charged_on`) (T8).
5. **Xem job/chart/WS của người khác, token hết hạn giữa chừng, F5 giữa job** → 404 / đóng WS 4401 / tiếp tục theo dõi đúng job (T7, T9, T15).

Ngoài ra bộ đối kháng SSRF của T2 bao gồm cả `http://user:pass@host`, `localhost.` (dấu chấm cuối), `::ffff:127.0.0.1`, hex/octal/thập phân, redirect và DNS-rebinding.

---

## File Structure

```
DE/src/ingestion/
  settings.py                       # IngestSettings (env INGEST_*)
  domain/{job.py, errors.py, ports.py, upload_rules.py}
  application/{ssrf.py, fetcher.py, normalizer.py, executor.py, submission.py,
               charts.py, demo.py, cleanup.py}
  infrastructure/{memory.py, postgres.py, ffmpeg.py, media.py, runner.py,
                  events.py, celery_app.py, mock_engine.py}
DE/src/api/routers/{analyze.py, charts.py, demo.py, jobs_ws.py}
DE/migrations/versions/0002_ingestion.py
DE/scripts/gen_demo_audio.py        # stdlib `wave`, ghi DATA_DIR/demo/*.wav (gitignored)
DE/tests/{unit/ingestion/, contract/, integration/test_ingestion_postgres.py}
web/src/analyze/{SourceSelector,Dropzone,LinkForm,JobTracker,QuotaBanner,DemoTracks}.tsx
web/src/analyze/{useJob.ts, validate.ts}  web/src/api/analyze.ts
web/src/pages/{WorkspacePage,ChartPage}.tsx   web/e2e/analyze.spec.ts
```

---

## Phần A — Backend

### Task 1: Nền tảng ingestion (settings, lỗi, state machine, port, repo in-memory, migration 0002)

**Files:**
- Create: `DE/src/ingestion/{__init__,settings}.py`, `domain/{__init__,job,errors,ports,upload_rules}.py`, `infrastructure/{__init__,memory}.py`, `DE/migrations/versions/0002_ingestion.py`
- Modify: `DE/src/persistence/tables.py` (thêm `source_fingerprints`, cột `jobs.input_path`, `jobs.title`, `jobs.fingerprint`, `jobs.charged_on`), `DE/src/api/errors.py` (map `IngestionError`)
- Test: `DE/tests/unit/ingestion/{test_job_state.py,test_errors_api.py}`, `DE/tests/integration/test_migrations.py` (mở rộng)

**Interfaces:**
- Produces:
  - `IngestSettings(BaseSettings)`: `data_dir: Path`, `max_upload_bytes=104_857_600`, `max_duration_ms=600_000`, `job_timeout_s=90`, `mock_stage_delay_ms=0`, `engine: Literal["mock"]="mock"`, `media_resolver: Literal["fake"]="fake"`, `job_runner: Literal["inprocess","celery"]="inprocess"`, `temp_ttl_s=3600`.
  - `class IngestionError(Exception)`: `code: ErrorCode`, `message: str`, `retryable: bool=False`.
  - `@dataclass(frozen=True) class Job` — trường: `id, owner_id: UUID; source_kind: SourceKind; source_url: str|None; title: str; input_path: str|None; fingerprint: str|None; content_hash: str|None; state: JobState; stage: JobStage|None; progress: float; error: JobError|None; result_chart_id: UUID|None; quota_charged: bool; charged_on: date|None; created_at, updated_at: datetime`.
  - `def advance(job: Job, *, state: JobState, stage: JobStage|None=None, progress: float|None=None, error: JobError|None=None, chart_id: UUID|None=None, now: datetime) -> Job` — ném `InvalidTransition(ValueError)`.
  - `def to_progress(job: Job) -> JobProgress` (hợp đồng đã có).
  - Protocols trong `ports.py`: `JobRepository{add, get, save, list_unfinished, try_mark_refunded(job_id)->bool}`, `ChartRepository{add(chart), get(chart_id, owner_id)->Chart|None}`, `AnalysisCache{get(content_hash)->AnalysisResult|None, put(result), fingerprint_to_hash(fp)->str|None, remember_fingerprint(fp, content_hash)}`, `MediaResolverPort{resolve(url, dest_dir)->ResolvedMedia}`, `JobRunnerPort{enqueue(job_id)}`, `JobEvents{publish(JobProgress), subscribe(job_id)->AsyncIterator[JobProgress]}`; `ResolvedMedia(path: Path, title: str, artist: str|None, playback: PlaybackSource)`.
  - `InMemoryJobRepository/ChartRepository/AnalysisCache` cho unit test.

- [ ] **Step 1: Viết test fail** — `test_job_state.py`: bảng chuyển hợp lệ duy nhất (`queued→processing`, `processing→processing` chỉ khi `progress` không giảm và stage không lùi, `→done` cần `chart_id` và progress 1.0, `→failed` cần `error`, `queued→done` (cache-hit) và `queued→failed` cho phép); `done→processing`, `failed→done`, `progress` giảm, stage lùi ⇒ `InvalidTransition`; `to_progress(job)` luôn qua được validator `JobProgress`. `test_errors_api.py`: route thăm dò ném `IngestionError(ErrorCode.FILE_TOO_LARGE, "x")` ⇒ 413 `{"code":"file_too_large","message":"x"}` — kiểm cả 8 mã theo `ERROR_HTTP_STATUS`. `test_migrations.py`: `0002` lên/xuống/lên sạch, `compare_metadata` rỗng, `source_fingerprints.fingerprint` là sha256 hex 64 ký tự, FK tới `analyses.content_hash` không cho giá trị mồ côi.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/ingestion tests/integration/test_migrations.py -q` — Expected: FAIL (module chưa có).
- [ ] **Step 3: Cài đặt** theo Interfaces; migration viết tay (như `0001`), `downgrade` xoá đúng những gì `upgrade` thêm; `install_error_handlers` thêm handler `IngestionError` dùng `error_response`.
- [ ] **Step 4: Chạy lại** cùng lệnh + `rtk uv run ruff check . && rtk uv run ruff format --check .` — Expected: PASS, sạch.
- [ ] **Step 5: Commit** `feat(ingestion): add job state machine, ports and migration 0002`

### Task 2: SSRF guard thuần (2.2 — phần chặn)

**Files:**
- Create: `DE/src/ingestion/application/ssrf.py`
- Test: `DE/tests/unit/ingestion/test_ssrf.py`

**Interfaces:**
- Produces: `Resolver = Callable[[str, int], Awaitable[list[str]]]` (host, port → danh sách IP chuỗi); `@dataclass(frozen=True) class SafeTarget: url: str; host: str; port: int; pinned_ip: str`; `async def check_url(url: str, resolve: Resolver) -> SafeTarget` ném `IngestionError(ErrorCode.SSRF_BLOCKED, <lý do cụ thể>)`; `def is_public_ip(ip: str) -> bool`.

- [ ] **Step 1: Viết test fail** (bảng `parametrize`, kỳ vọng mọi dòng bị chặn với lý do khác rỗng): `file:///etc/passwd`, `ftp://h/x`, `gopher://h`, `javascript:alert(1)`, `http://127.0.0.1/`, `http://[::1]/`, `http://10.0.0.5/`, `http://172.16.0.1/`, `http://192.168.1.1/`, `http://169.254.169.254/latest/meta-data`, `http://[fd00::1]/`, `http://[fe80::1]/`, `http://0.0.0.0/`, `http://2130706433/`, `http://0x7f000001/`, `http://0177.0.0.1/`, `http://127.1/`, `http://[::ffff:127.0.0.1]/`, `http://localhost/`, `http://localhost./`, `http://user:pass@example.com/`, `http://example.com:22/` (cổng ngoài {80,443,8080,8443}? — **chỉ cho 80/443**), host phân giải về private, host phân giải **hỗn hợp** (một IP public + một private ⇒ chặn). Hợp lệ: `https://example.com/a.mp3` với resolver trả `93.184.216.34` ⇒ `SafeTarget(pinned_ip="93.184.216.34", port=443)`; IP-literal public `http://93.184.216.34/x.mp3` không gọi resolver.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/ingestion/test_ssrf.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt `check_url`**: parse bằng `urllib.parse`, chuẩn hoá IP-literal qua `ipaddress` + `socket.inet_aton` cho dạng thập phân/hex/octal, dùng `ip.is_global` (không dùng danh sách tay) và giải `IPv4-mapped IPv6` về IPv4 trước khi kiểm; trả IP đầu tiên đã kiểm để **pin**.
- [ ] **Step 4: Chạy lại** — Expected: PASS 100 % (tiêu chí nghiệm thu NFR-16/FR-20).
- [ ] **Step 5: Commit** `feat(ingestion): add SSRF guard with adversarial URL suite`

### Task 3: SafeFetcher, FakeMediaResolver, playback_for_url (2.2 — phần lấy)

**Files:**
- Create: `application/fetcher.py`, `infrastructure/media.py`
- Test: `DE/tests/unit/ingestion/{test_fetcher.py,test_media.py}`

**Interfaces:**
- Consumes: `check_url`, `SafeTarget`, `Resolver` (Task 2); `MediaResolverPort`, `ResolvedMedia` (Task 1).
- Produces: `class SafeFetcher(resolve: Resolver, transport: httpx.AsyncBaseTransport|None, max_bytes: int, timeout_s: float=30, max_redirects: int=3)` với `async def fetch(self, url: str, dest: Path) -> Path`; `def playback_for_url(url: str) -> PlaybackSource` (youtube/youtu.be ⇒ `youtube` + video id; soundcloud.com ⇒ `soundcloud` + url; còn lại ⇒ `direct` + `audio_url=url`); `class FakeMediaResolver(demo_audio: Path)` — `resolve` copy WAV demo vào `dest_dir`, `title` lấy từ tên path của URL; URL chứa `fail-unresolvable` ⇒ `IngestionError(URL_UNRESOLVABLE, ...)`.

- [ ] **Step 1: Viết test fail** (dùng `httpx.MockTransport`, resolver giả): kết nối tới **IP đã pin** (kiểm bằng header `Host` + địa chỉ trong request giả), mỗi redirect (301/302/307/308) được kiểm lại bằng `check_url` — redirect `→ http://169.254.169.254/` bị `SSRF_BLOCKED`, redirect tới URL public hợp lệ đi tiếp, quá 3 redirect ⇒ `URL_UNRESOLVABLE`; **DNS rebinding**: resolver trả public lần 1, private lần 2 ⇒ vẫn kết nối IP lần 1 (không resolve lại) và redirect kế tiếp bị chặn; stream vượt `max_bytes` ⇒ dừng + xoá file dở + `FILE_TOO_LARGE`; vượt `timeout_s` ⇒ `URL_UNRESOLVABLE`; HTTP 403/404/5xx ⇒ `URL_UNRESOLVABLE`; `Content-Type: text/html` ⇒ `UNSUPPORTED_FORMAT`. `playback_for_url`: bảng 5 URL (youtube watch, youtu.be, soundcloud, `.mp3` trực tiếp, URL lạ) ⇒ đúng `PlaybackSource`, và kết quả luôn qua validator `PlaybackSource`.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/ingestion/test_fetcher.py tests/unit/ingestion/test_media.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt**: pin bằng cách dựng request tới `https://<pinned_ip>:port` kèm `Host` và `sni_hostname` (extension của httpcore) — chọn một cách và ghi vào memory.md; chỉ chấp nhận `Content-Type` `audio/*` hoặc `application/octet-stream`; đếm byte khi stream. **Không** triển khai yt-dlp (hoãn 5.3).
- [ ] **Step 4: Chạy lại + ruff** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ingestion): add pinned SafeFetcher, fake media resolver and playback mapping`

### Task 4: Normalizer ffprobe/ffmpeg và luật upload (2.1 — phần xử lý audio)

**Files:**
- Create: `domain/upload_rules.py`, `application/normalizer.py`, `infrastructure/ffmpeg.py`
- Test: `DE/tests/unit/ingestion/{test_upload_rules.py,test_normalizer.py}`, `DE/tests/support/audio.py` (sinh WAV/MP3 nhỏ bằng stdlib `wave` + gọi `ffmpeg` cho mp3; test dùng ffmpeg thật, `pytest.skip` nếu thiếu)

**Interfaces:**
- Produces:
  - `def sniff_format(head: bytes) -> str|None` (đọc 16 byte đầu: `ID3`/sync MPEG ⇒ `mp3`, `RIFF…WAVE` ⇒ `wav`, `fLaC` ⇒ `flac`, `….ftyp` ⇒ `m4a`, `OggS` ⇒ `ogg`).
  - `def check_limits(size_bytes: int, duration_ms: int|None, *, settings) -> None` ném `IngestionError` (`FILE_TOO_LARGE`, `DURATION_TOO_LONG`, `UNSUPPORTED_FORMAT` khi `duration_ms` là `None`/≤ 0).
  - `def safe_title(filename: str) -> str` (bỏ thư mục và đuôi, ký tự điều khiển, cắt 300, rỗng ⇒ `"Untitled"`).
  - `@dataclass(frozen=True) MediaInfo(format_name: str, duration_ms: int|None, has_audio: bool)`; `@dataclass(frozen=True) NormalizedAudio(path: Path, content_hash: str, duration_ms: int)`.
  - `class Normalizer(Protocol): async def probe(self, path: Path) -> MediaInfo; async def normalize(self, src: Path, dest_dir: Path, stem: str) -> NormalizedAudio` — `FfmpegNormalizer` chạy `asyncio.create_subprocess_exec` (không shell), ffmpeg `-vn -ar 44100 -ac 1 -c:a pcm_s16le`, hash sha256 toàn bộ byte WAV đầu ra. `stem` đi vào tên file `<stem>.wav` (đã làm sạch `[a-z0-9_-]`).

- [ ] **Step 1: Viết test fail**: `check_limits` đúng `104_857_600` byte OK / `+1` ⇒ `FILE_TOO_LARGE`; đúng `600_000` ms OK / `600_001` ⇒ `DURATION_TOO_LONG`; `duration_ms=None` ⇒ `UNSUPPORTED_FORMAT`; `sniff_format` đủ 5 định dạng + `b"hello"` ⇒ `None` + byte rỗng ⇒ `None`; `safe_title("../../etc/passwd.mp3")=="passwd"`, tên 1000 ký tự ⇒ ≤ 300, `"\x00\x1b[31m.wav"` không còn ký tự điều khiển. `FfmpegNormalizer` (ffmpeg thật): WAV 22.05 kHz stereo 1 s ⇒ đầu ra 44100 Hz, 1 kênh, 16-bit, `duration_ms` ≈ 1000 ±20, hai lần chuẩn hoá cùng input cho **cùng `content_hash`**; tệp văn bản đặt tên `.mp3` ⇒ `probe` lỗi ⇒ `UNSUPPORTED_FORMAT`; tệp 0 byte ⇒ `UNSUPPORTED_FORMAT`; ffprobe không có ⇒ lỗi cấu hình rõ ràng (không 500 mơ hồ).
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/ingestion/test_upload_rules.py tests/unit/ingestion/test_normalizer.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces; `ffprobe -v error -show_entries format=format_name,duration -of json`; chuyển `duration` giây → ms bằng `round`.
- [ ] **Step 4: Chạy lại + ruff** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ingestion): add upload rules and ffmpeg normalizer`

### Task 5: MockAnalysisEngine + bộ contract test dùng lại ở M5 (2.3)

**Files:**
- Create: `infrastructure/mock_engine.py`
- Test: `DE/tests/contract/test_analysis_engine_contract.py` (+ `DE/tests/contract/__init__.py`)

**Interfaces:**
- Consumes: `AnalysisEngine`, `AnalysisError` (`analysis_contract/ports.py`), fixtures `contracts/fixtures/results/*.result.json`.
- Produces: `class MockAnalysisEngine(fixtures_dir: Path, stage_delay_ms: int=0)`; `async def analyze(request, on_progress) -> AnalysisResult`. Chọn kịch bản theo `Path(request.audio_path).stem`: bắt đầu bằng `demo-<tên fixture>` ⇒ đúng fixture đó; chứa `fail-ai` ⇒ `AnalysisError(AI_FAILURE, retryable=True)`; chứa `fail-timeout` ⇒ `await asyncio.sleep(3600)`; còn lại ⇒ fixture thứ `int(content_hash,16) % 7`. Kết quả = fixture với `job_id` và `content_hash` thay bằng của request (`model_copy`). Phát progress `separating 0.45 → beat_tracking 0.7 → recognizing 0.9` (qua `on_progress`, state `processing`).

- [ ] **Step 1: Viết test fail** — một hàm `engine_factories` (hiện chỉ `[mock]`, M5 thêm adapter thật) và các test áp dụng cho mọi adapter: kết quả qua `AnalysisResult.model_validate`; `result.job_id == request.job_id` và `content_hash` khớp; mọi `JobProgress` hợp lệ, `progress` không giảm, `stage` theo thứ tự `JobStage`, không phát `done`/`failed` (việc của executor); `fail-ai` ⇒ `AnalysisError` có `code == ErrorCode.AI_FAILURE`; huỷ task (`asyncio.CancelledError`) lan ra ngoài không bị nuốt; cùng input ⇒ cùng output (xác định); `stage_delay_ms=20` ⇒ tổng thời gian ≥ 3×20 ms (fake clock không cần).
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/contract -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces; nạp fixture một lần (cache trong `__init__`), không đọc đĩa mỗi lần phân tích.
- [ ] **Step 4: Chạy lại + ruff** — Expected: PASS. Thêm `tests/contract` vào `testpaths` nếu chưa tự nhận.
- [ ] **Step 5: Commit** `feat(ingestion): add mock analysis engine and engine contract tests`

### Task 6: JobExecutor, dựng chart, repo PostgreSQL (2.3)

**Files:**
- Create: `application/executor.py`, `application/charts.py`, `infrastructure/runner.py`, `infrastructure/postgres.py`
- Test: `DE/tests/unit/ingestion/{test_executor.py,test_chart_build.py,test_runner.py}`, `DE/tests/integration/test_ingestion_postgres.py`

**Interfaces:**
- Consumes: Task 1 (Job, repos, ports), Task 3 (`MediaResolverPort`), Task 4 (`Normalizer`), Task 5 (engine), `QuotaService.refund`, `ChargeReceipt` (`identity/application/quota.py`).
- Produces:
  - `class JobExecutor(jobs, charts, cache, engine, normalizer, resolver, quota, events, settings, clock)` — `async def run(self, job_id: UUID) -> None` (không bao giờ ném ra ngoài; mọi nhánh kết thúc ở `done` hoặc `failed`); `async def recover_orphans(self) -> int` (job `queued/processing` cũ hơn `job_timeout_s` ⇒ `failed(ai_failure)` + hoàn quota; trả số job đã xử lý).
  - Thứ tự stage: URL: `fetching → normalizing → (cache?) → separating → beat_tracking → recognizing`; FILE/DEMO bỏ `fetching`.
  - `def build_chart(job: Job, result: AnalysisResult, *, playback: PlaybackSource, now: datetime) -> Chart` (title = `job.title`, `ai_events` = `result.chord_events`, `source.kind/url/content_hash`).
  - `class InProcessJobRunner(executor)` — `enqueue` tạo `asyncio.Task`, giữ tham chiếu mạnh trong `set`, bỏ khi xong; `async def drain()` cho test/shutdown.
  - `PgJobRepository`, `PgChartRepository`, `PgAnalysisCache` (SQLAlchemy Core như `identity/infrastructure/postgres.py`; `save` dùng `UPDATE … WHERE state NOT IN ('done','failed')` để job cuối cùng không bị ghi đè; `try_mark_refunded` là một `UPDATE … SET quota_charged=false WHERE id=:id AND quota_charged RETURNING`).

- [ ] **Step 1: Viết test fail** (engine/normalizer/resolver giả, repo in-memory, `FakeClock`): đường vui FILE ⇒ chuỗi `JobProgress` đúng thứ tự stage, kết thúc `done` có `result_chart_id`, chart được lưu với `ai_events` của kết quả và `source.playback.kind=="file"`, `audio_url` trỏ `/api/v1/charts/{id}/audio`; URL ⇒ có stage `fetching` và `playback_for_url`; `AnalysisError(AI_FAILURE)` ⇒ `failed` với đúng `JobError`, **không** có chart, hoàn quota đúng một lần (gọi `run` hai lần vẫn một lần); engine ngủ quá `job_timeout_s` (dùng giá trị 0.05) ⇒ `failed(analysis_timeout)`; `RuntimeError("secret")` ⇒ `failed(ai_failure)` với message chung, **không lộ** "secret", được log; `IngestionError` từ resolver/normalizer ⇒ giữ nguyên `code`; file tạm (input, thư mục fetch) luôn bị xoá ở mọi nhánh; `recover_orphans` xử lý job `processing` cũ và bỏ qua job mới; `InProcessJobRunner` giữ task đến khi xong và `drain()` chờ được. `test_ingestion_postgres.py` (integration): CRUD job/chart/cache trên `chordsense_test`, `save` không ghi đè job đã `done`, `try_mark_refunded` trả `True` đúng một lần kể cả khi gọi song song 10 lần.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/ingestion tests/integration/test_ingestion_postgres.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces: bọc toàn bộ `run` bằng `try/except Exception`, timeout bằng `asyncio.timeout(settings.job_timeout_s)` quanh phần engine (không bọc normalize); mỗi lần đổi trạng thái: `advance` → `jobs.save` → `events.publish`.
- [ ] **Step 4: Chạy lại + ruff** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ingestion): add job executor, chart building and PostgreSQL repositories`

### Task 7: Submission service + endpoints `analyze/file|url`, `analyze/status`, `GET /charts/{id}` (2.1, 2.2, 2.3 — phần HTTP)

**Files:**
- Create: `application/submission.py`, `api/routers/{analyze,charts}.py`
- Modify: `api/container.py` (thêm `submission`, `jobs`, `charts`), `api/wiring.py`, `api/app.py` (router, lifespan gọi `recover_orphans`), `api/schemas.py` (`JobAccepted{job_id}`), `analysis_contract/export.py` (thêm `JobAccepted`, `Chart` đã có), `tests/support/{identity,api}.py` (container in-memory có ingestion)
- Test: `DE/tests/unit/api/{test_analyze_file.py,test_analyze_url.py,test_job_status.py,test_charts_api.py,test_analyze_latency.py}`, `DE/tests/integration/test_app_postgres.py` (thêm luồng upload thật)

**Interfaces:**
- Consumes: Tasks 1–6; `QuotaService.consume` ⇒ `ChargeReceipt`.
- Produces:
  - `class SubmissionService(jobs, runner, quota, normalizer, settings, clock, work_dir)` — `async def submit_file(user: User, filename: str, stream: AsyncIterator[bytes]) -> Job`; `async def submit_url(user: User, url: str) -> Job`. Thứ tự: đọc stream ⇒ ghi file tạm dưới tên sinh ra, đếm byte, dừng + xoá khi vượt `max_upload_bytes`, tính `fingerprint` ⇒ `sniff_format` ⇒ `probe` ⇒ `check_limits` ⇒ **(Task 8 chèn cache ở đây)** ⇒ `quota.consume` ⇒ `jobs.add` (lưu `charged_on`) ⇒ `runner.enqueue`; `enqueue` ném ⇒ hoàn quota + xoá file + `failed`. URL: `check_url` ⇒ consume ⇒ add ⇒ enqueue.
  - `POST /api/v1/analyze/file` (multipart, field `file`) và `POST /api/v1/analyze/url` (`{"url": "..."}`) ⇒ `202 JobAccepted`; `GET /api/v1/analyze/status/{id}` ⇒ `JobProgress` (404 nếu không phải chủ); `GET /api/v1/charts/{id}` ⇒ `Chart` (404 nếu không phải chủ). Chặn body sớm: ASGI wrapper trên route `analyze/file` từ chối khi `Content-Length > max_upload_bytes + 64 KiB` và đếm byte nhận được; giới hạn **chính xác** tính trên byte của part `file`.

- [ ] **Step 1: Viết test fail** (container in-memory, `ffmpeg` thật cho file nhỏ): file WAV hợp lệ ⇒ 202 + `job_id`; sau `runner.drain()` `status` = `done` và `GET /charts/{result_chart_id}` trả chart với `ai_events` hợp lệ; chưa đăng nhập ⇒ 401; **biên**: part đúng `max_upload_bytes` (dùng `max_upload_bytes` nhỏ qua settings, ví dụ 2 048) ⇒ 202, `+1` ⇒ 413 và **không** còn file tạm; `Content-Length` khai láo nhỏ nhưng stream lớn ⇒ vẫn 413; tệp `.mp3` chứa chữ ⇒ 415; 0 byte ⇒ 415; WAV 10:00.000 ⇒ 202, 10:00.001 ⇒ 413 (`DURATION_TOO_LONG`) — tạo bằng WAV header giả + ffprobe thật hoặc `Normalizer` giả trả duration; tên `../../x.mp3` / 1 000 ký tự / `fail-ai.mp3` giả không thoát thư mục work (chỉ `fail-ai` *thật* qua tên được làm sạch mới kích hoạt mock — ghi rõ trong test); URL: `http://127.0.0.1/` ⇒ 422 `ssrf_blocked` trước khi tạo job (không trừ quota), URL public ⇒ 202; trừ quota: 5 file OK, lần thứ 6 ⇒ 429 có `limit`, `resets_at`, `Retry-After`; lỗi 413/415/422 **không** trừ quota; job của user khác ⇒ 404 cho cả `status` lẫn `charts`; `test_analyze_latency.py`: 50 POST tuần tự (WAV ~100 KB, mock engine chậm) ⇒ p95 ≤ 300 ms. Integration: đăng ký → upload → poll → chart thật trên `chordsense_test`.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/api -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces; `build_container` thêm ingestion (Pg repos, `FfmpegNormalizer`, `MockAnalysisEngine`, `FakeMediaResolver`, `InProcessJobRunner`); thêm `JobAccepted` vào `_EXPORTED`, chạy `rtk uv run python -m src.analysis_contract.export`.
- [ ] **Step 4: Chạy lại** `rtk uv run pytest -q` toàn bộ + ruff — Expected: PASS, drift openapi xanh.
- [ ] **Step 5: Commit** `feat(api): add analyze file/url, job status and chart read endpoints`

### Task 8: Hoàn quota đúng một lần, cache theo nội dung, race (2.4)

**Files:**
- Modify: `application/submission.py` (tra cache trước `consume`), `application/executor.py` (cache-hit sau chuẩn hoá ⇒ hoàn), `infrastructure/postgres.py` (`remember_fingerprint`)
- Test: `DE/tests/unit/ingestion/test_quota_flow.py`, `DE/tests/integration/test_quota_race.py`

**Interfaces:**
- Consumes: `AnalysisCache.fingerprint_to_hash/remember_fingerprint/get/put` (Task 1), `Job.charged_on`.
- Produces: không thêm API công khai; hành vi: cache-hit ở POST ⇒ `Job` tạo thẳng ở `done` kèm chart (bản sao cho chủ mới, `forked_from=None`), `quota_charged=False`; cache-hit sau chuẩn hoá ⇒ `executor` `try_mark_refunded` + `quota.refund(ChargeReceipt(user_id, charged_on, True))`.

- [ ] **Step 1: Viết test fail**: upload cùng file lần 2 (cùng byte) ⇒ 202 với job đã `done`, `used` không tăng kể cả khi user đã dùng 5/5 (không 429); cùng nhạc nhưng encode khác (WAV vs FLAC cùng waveform) ⇒ trừ lúc POST, sau chuẩn hoá hoàn lại, `used` về như cũ; job `failed` ⇒ `used` giảm 1, gọi hoàn lần hai không giảm tiếp; trừ lúc 23:59:59 UTC, job fail lúc 00:00:30 ⇒ hoàn vào **ngày đã trừ** (`FakeClock`); user Pro không bao giờ bị trừ/hoàn; **race** (integration, PG thật): 10 POST đồng thời của user Free còn 3 lượt ⇒ đúng 3 × 202 và 7 × 429, `used == 5`.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/ingestion/test_quota_flow.py tests/integration/test_quota_race.py -q` — Expected: FAIL (cache-hit chưa có; hoàn theo ngày chưa đúng).
- [ ] **Step 3: Cài đặt** theo Interfaces.
- [ ] **Step 4: Chạy lại + toàn bộ** `rtk uv run pytest -q` + ruff — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ingestion): refund-once quota and content-hash cache`

### Task 9: Realtime — WebSocket + polling (2.5)

**Files:**
- Create: `infrastructure/events.py` (`InMemoryJobEvents`), `api/routers/jobs_ws.py`
- Modify: `api/app.py`, `api/container.py` (`events`)
- Test: `DE/tests/unit/api/test_jobs_ws.py`, `DE/tests/unit/ingestion/test_events.py`

**Interfaces:**
- Produces: `WS /api/v1/ws/jobs/{job_id}`; auth bằng `Sec-WebSocket-Protocol: bearer, <access_token>` (server `accept(subprotocol="bearer")`); gửi ngay **snapshot** `JobProgress` hiện tại rồi các thay đổi, JSON giống `GET /analyze/status`; đóng sau trạng thái cuối. Mã đóng: `4401` (token thiếu/sai/hết hạn), `4404` (không tồn tại hoặc không phải chủ). `InMemoryJobEvents`: mỗi subscriber có hàng đợi riêng, không rò rỉ khi client rớt.

- [ ] **Step 1: Viết test fail** (`starlette.testclient.TestClient`): kết nối khi job đang `processing` ⇒ nhận snapshot đầu tiên rồi các stage tiếp theo, nhận `done` rồi server đóng; kết nối **sau** khi job đã `done` ⇒ nhận đúng 1 snapshot rồi đóng (F5 muộn); token sai ⇒ đóng 4401; job của user khác ⇒ 4404; thời gian từ lúc `events.publish` đến lúc client nhận < 1 s (đo); client ngắt giữa chừng ⇒ subscriber bị huỷ (không còn trong registry); bắt tay không có subprotocol `bearer` ⇒ 4401.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/api/test_jobs_ws.py tests/unit/ingestion/test_events.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces; xác thực dùng lại `container.tokens.verify(token, "access")` + `users.get_by_id`.
- [ ] **Step 4: Chạy lại + ruff** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(api): stream job progress over WebSocket with polling-compatible snapshots`

### Task 10: Demo catalog (2.7)

**Files:**
- Create: `application/demo.py`, `api/routers/demo.py`, `DE/scripts/gen_demo_audio.py`
- Modify: `.gitignore` (`DE/data/demo/`), `api/container.py`, `submission.py` (`submit_demo`)
- Test: `DE/tests/unit/ingestion/test_demo.py`, `DE/tests/unit/api/test_demo_api.py`

**Interfaces:**
- Produces: `GET /api/v1/demo-tracks` ⇒ `[{id, title, artist, duration_ms, tempo_bpm, key}]` (7 bài từ fixtures: `ii_v_i_c`, `blues_f`, `autumn_leaves_gm`, `pop_slash_c`, `waltz_g_3_4`, `compound_6_8_e`, `rubato_downbeats_c`); `POST /api/v1/analyze/demo/{id}` ⇒ `202 JobAccepted`, **không trừ quota**, không ràng buộc cache; id lạ ⇒ 404. Script `gen_demo_audio.py` (stdlib `wave`, `math`) tổng hợp click + hợp âm theo đúng timeline của từng fixture ⇒ `DATA_DIR/demo/demo-<id>.wav` (tên `demo-…` để mock chọn đúng fixture).

- [ ] **Step 1: Viết test fail**: catalog có đúng 7 mục, mỗi `duration_ms` khớp fixture; `POST …/demo/ii_v_i_c` bởi user Free đã hết 5/5 ⇒ vẫn 202 và `used` không đổi; chart tạo ra có `source.kind == "demo"` và `ai_events` đúng fixture; file WAV sinh ra là 44.1 kHz mono và độ dài lệch fixture ≤ 10 ms; id `../x` ⇒ 404; thiếu WAV (chưa chạy script) ⇒ lỗi rõ ràng 503 `{code:"demo_unavailable"}`, không 500.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/ingestion/test_demo.py tests/unit/api/test_demo_api.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces; chạy `rtk uv run python scripts/gen_demo_audio.py` một lần ở dev; `run_demo` job dùng `source_kind=demo`, `quota_charged=False`.
- [ ] **Step 4: Chạy lại + ruff** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ingestion): add demo track catalog and generator script`

### Task 11: Celery + Redis runner (2.6)

**Files:**
- Create: `infrastructure/celery_app.py`, `infrastructure/events.py` (thêm `RedisJobEvents`), `DE/tests/integration/test_celery_runner.py`
- Modify: `DE/pyproject.toml` (extra `jobs = ["celery>=5.4", "redis>=5.2"]`), `api/wiring.py` (chọn runner/events theo `INGEST_JOB_RUNNER`), `DE/.env.example`
- Test: `DE/tests/unit/ingestion/test_celery_runner.py`

**Interfaces:**
- Produces: `class CeleryJobRunner(app: Celery)` với `enqueue(job_id)` ⇒ `run_job.delay(str(job_id))`; task `run_job` tự dựng container (cùng `build_container`) và gọi `JobExecutor.run`; `soft_time_limit = job_timeout_s + 5`, `time_limit = job_timeout_s + 15`; `RedisJobEvents(redis_url)` pub/sub kênh `jobs:<id>` để API (process khác) nhận progress của worker. Windows: worker chạy `--pool=solo` (ghi vào README/memory.md).

- [ ] **Step 1: Viết test fail**: (unit, `task_always_eager=True`, Redis giả) `enqueue` chạy `executor.run` đúng một lần với `job_id` dạng chuỗi; `SoftTimeLimitExceeded` ⇒ job `failed(analysis_timeout)` + hoàn quota; chọn runner theo `INGEST_JOB_RUNNER`; (integration, cần `TEST_REDIS_URL`, skip nếu thiếu — mở rộng `tests/conftest.py`) `RedisJobEvents` publish ở một client, subscriber ở client khác nhận đủ, theo thứ tự.
- [ ] **Step 2: Chạy** `rtk uv sync --extra api --extra db --extra jobs` rồi `rtk uv run pytest tests/unit/ingestion/test_celery_runner.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces; mặc định vẫn `inprocess` (không bắt buộc Redis/Docker).
- [ ] **Step 4: Chạy lại** (kèm `TEST_REDIS_URL=redis://localhost:6379/15` nếu Redis native đang chạy) + toàn bộ + ruff — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ingestion): add Celery job runner and Redis job events`

### Task 12: Dọn dẹp file tạm (2.8)

**Files:**
- Create: `application/cleanup.py`
- Modify: `api/app.py` (lifespan: quét lúc khởi động + tác vụ nền mỗi `temp_ttl_s/4`)
- Test: `DE/tests/unit/ingestion/test_cleanup.py`

**Interfaces:**
- Produces: `class TempSweeper(root: Path, ttl_s: int, jobs: JobRepository, clock)` — `async def sweep() -> int` xoá thư mục con của `DATA_DIR/tmp/` cũ hơn `ttl_s` **và** không thuộc job `queued/processing`; không bao giờ đụng `DATA_DIR/audio/` hay `DATA_DIR/demo/`.

- [ ] **Step 1: Viết test fail**: thư mục cũ của job đã xong bị xoá; của job đang chạy được giữ dù cũ; thư mục mới được giữ; `audio/` và `demo/` còn nguyên; đường dẫn ngoài `root` (symlink/`..`) không bị đụng; trả đúng số thư mục đã xoá.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/ingestion/test_cleanup.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces (so `mtime` với `clock()`).
- [ ] **Step 4: Chạy lại + toàn bộ + ruff** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ingestion): sweep stale temporary files`

---

## Phần B — Frontend (`web/`)

### Task 13: Tầng API phân tích (multipart, lỗi 429, types sinh lại)

**Files:**
- Create: `web/src/api/analyze.ts`
- Modify: `web/src/api/http.ts` (`body` nhận `FormData`; `ApiError` thêm `limit`, `resetsAt`), `web/src/api/generated/schema.d.ts` (sinh lại), `web/src/test/server.ts` (handler mặc định cho analyze)
- Test: `web/src/api/analyze.test.ts`, bổ sung `http.test.ts`

**Interfaces:**
- Produces: `uploadFile(client, file: File): Promise<JobAccepted>`; `analyzeUrl(client, url: string): Promise<JobAccepted>`; `analyzeDemo(client, id: string)`; `getJob(client, id): Promise<JobProgress>`; `getChart(client, id): Promise<Chart>`; `listDemoTracks(client)`; `ApiError.limit: number|null`, `ApiError.resetsAt: Date|null` (đọc từ body 429); `jobSocketUrl(id)` + `openJobSocket(client, id): WebSocket` (subprotocol `['bearer', token]`).

- [ ] **Step 1: Viết test fail**: `uploadFile` gửi `multipart/form-data` đúng field `file` và **không** tự đặt `Content-Type` (để trình duyệt thêm boundary); 401 hết hạn vẫn refresh một lần rồi gửi lại cùng `FormData`; body 429 `{code:"quota_exceeded", limit:5, resets_at}` ⇒ `error.limit===5`, `error.resetsAt instanceof Date`; `getJob/getChart` parse đúng kiểu sinh từ OpenAPI (compile-time + fixture `contracts/fixtures/jobs/*.json`).
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/api` — Expected: FAIL.
- [ ] **Step 3: Cài đặt**; chạy `rtk uv run python -m src.analysis_contract.export` (ở `DE/`) rồi `rtk npm run gen:api`.
- [ ] **Step 4: Chạy lại** + `rtk npm run lint && rtk npm run build` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(web): add analyze API client with multipart upload and quota error details`

### Task 14: Source Selector — tab Upload / Record (disabled) / Link (2.1, 2.2 FE)

**Files:**
- Create: `web/src/analyze/{SourceSelector,Dropzone,LinkForm}.tsx`, `web/src/analyze/validate.ts`, `web/src/pages/WorkspacePage.tsx`
- Modify: `web/src/i18n/messages.ts` (vi/en), `web/src/routes.tsx` (`/workspace` dùng `WorkspacePage`)
- Test: `web/src/analyze/{validate.test.ts,SourceSelector.test.tsx}`

**Interfaces:**
- Produces: `validateFile(file: {name: string; size: number}): ErrorKey|null` (quá 104 857 600 byte ⇒ `fileTooLarge`; đuôi ngoài mp3/wav/flac/m4a/ogg ⇒ `unsupportedFormat`); `validateUrl(raw: string): ErrorKey|null` (không phải http/https, có `user:pass@`, host là IP private/loopback/`localhost` ⇒ lý do cụ thể); `SourceSelector({ onSubmit(source: {kind:'file',file:File}|{kind:'url',url:string}) })`.

- [ ] **Step 1: Viết test fail**: bảng `validateFile`/`validateUrl` (cả ranh giới đúng/+1 byte; `file://`, `http://127.0.0.1`, `http://user:pw@a.com`, `https://youtu.be/x` hợp lệ); tab **Record** hiển thị nhưng `disabled` + có `aria-describedby` giải thích; chọn file hợp lệ ⇒ gọi `onSubmit`; file 101 MB ⇒ hiển thị lỗi, **không** gọi; link không hợp lệ ⇒ lỗi theo lý do và gợi ý "tải file lên thay thế" (AF1) có nút chuyển sang tab Upload; kéo-thả file hoạt động (`fireEvent.drop`); điều hướng bàn phím giữa các tab (`role=tablist`, mũi tên).
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/analyze` — Expected: FAIL.
- [ ] **Step 3: Cài đặt**; chuỗi thêm vào cả hai từ điển `vi`/`en` (TypeScript ép đủ khoá).
- [ ] **Step 4: Chạy lại + lint + build** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(web): add source selector with upload dropzone and link form`

### Task 15: `useJob` (WS → polling 2 s) + JobTracker (2.3, 2.5 FE)

**Files:**
- Create: `web/src/analyze/{useJob.ts,JobTracker.tsx}`
- Modify: `web/src/pages/WorkspacePage.tsx`, `web/src/i18n/messages.ts` (nhãn stage + thông điệp theo `ErrorCode`)
- Test: `web/src/analyze/{useJob.test.tsx,JobTracker.test.tsx}`, `web/src/test/fakeSocket.ts`

**Interfaces:**
- Consumes: `openJobSocket`, `getJob` (Task 13).
- Produces: `useJob(jobId: string|null): { job: JobProgress|null; transport: 'ws'|'polling'|'idle'; error: ApiError|null }`; lưu `jobId` đang theo dõi ở `sessionStorage['chordsense.activeJob']` và khôi phục sau F5 (xác thực lại bằng `getJob`; 404 ⇒ xoá); `JobTracker({ job, onRetry, onDismiss })` hiển thị 5 bước (fetching chỉ khi nguồn URL), thanh `progress`, thông điệp lỗi theo `ErrorCode`, nút **Thử lại** (chỉ khi `failed`) và **Dừng theo dõi**.

- [ ] **Step 1: Viết test fail** (WebSocket giả + MSW): nhận snapshot rồi cập nhật ⇒ UI đổi stage; socket `onclose` bất thường ⇒ chuyển sang polling mỗi 2 000 ms (fake timers) và tiếp tục đến `done`; kết nối WS lỗi ngay ⇒ polling từ đầu; không gọi `getJob` song song khi WS còn sống; dừng hẳn (đóng socket/clear timer) khi `done|failed` và khi unmount; mount với `activeJob` trong storage ⇒ tiếp tục theo dõi đúng job (F5 giữa chừng); `activeJob` trỏ job 404 ⇒ bị xoá, UI về trạng thái chọn nguồn; mỗi `ErrorCode` (8 mã) ⇒ đúng thông điệp (vi + en); `failed` ⇒ nút "Thử lại" gọi `onRetry`; `done` ⇒ gọi `onDone(chartId)` đúng một lần (không lặp khi re-render).
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/analyze` — Expected: FAIL.
- [ ] **Step 3: Cài đặt**; polling dùng `setTimeout` đệ quy (không `setInterval`) để không chồng request.
- [ ] **Step 4: Chạy lại + lint + build** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(web): track analysis jobs over WebSocket with polling fallback`

### Task 16: Banner 429, làm mới quota, trang chart tạm và tự mở chart (2.4 FE, DoD)

**Files:**
- Create: `web/src/analyze/QuotaBanner.tsx`, `web/src/pages/ChartPage.tsx`
- Modify: `web/src/routes.tsx` (`/charts/:id`), `web/src/shell/QuotaBadge.tsx` (export `quotaKey`), `web/src/pages/WorkspacePage.tsx`
- Test: `web/src/analyze/QuotaBanner.test.tsx`, `web/src/pages/ChartPage.test.tsx`, `web/src/analyze/flow.test.tsx`

**Interfaces:**
- Produces: `QuotaBanner({ error: ApiError })` (giờ reset theo múi giờ máy, CTA "Nâng cấp Pro" là liên kết tạm tới `#upgrade`, EF2); `ChartPage` (tải `getChart`, hiển thị tiêu đề, key, tempo, danh sách `symbol` + mốc thời gian — thay bằng Smart Grid ở M3); sau mỗi lần nộp job và khi job kết thúc: `queryClient.invalidateQueries({ queryKey: quotaKey(user.id) })`.

- [ ] **Step 1: Viết test fail**: `flow.test.tsx` (MSW, cả luồng) chọn file ⇒ 202 ⇒ stepper chạy ⇒ `done` ⇒ router tới `/charts/<id>` và thấy tiêu đề + ít nhất một hợp âm; 429 ⇒ banner hiện `limit` và giờ reset, không có job; badge quota được làm mới sau khi nộp (gọi lại `/users/me/quota`); 413/415/422 hiện thông điệp đúng và **không** đổi quota; `ChartPage` với chart không tồn tại (404) ⇒ thông báo + liên kết về Thư viện; nhấn "Thử lại" sau `failed` nộp lại **cùng file/URL** (giữ trong state, không bắt chọn lại).
- [ ] **Step 2: Chạy** `rtk npm test -- --run` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces.
- [ ] **Step 4: Chạy lại + lint + build** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(web): show quota banner, refresh quota badge and open the finished chart`

### Task 17: Nút "Thử bài demo" (2.7 FE)

**Files:**
- Create: `web/src/analyze/DemoTracks.tsx`
- Modify: `web/src/pages/WorkspacePage.tsx`, `web/src/i18n/messages.ts`
- Test: `web/src/analyze/DemoTracks.test.tsx`

**Interfaces:**
- Consumes: `listDemoTracks`, `analyzeDemo` (Task 13), `useJob` (Task 15).
- Produces: `DemoTracks({ onStart(jobId: string) })` — danh sách 7 bài với tên/nhịp/giọng; bấm một bài ⇒ `analyzeDemo`.

- [ ] **Step 1: Viết test fail**: hiển thị 7 bài từ API (skeleton khi đang tải, lỗi ⇒ toast `toastFor`); bấm bài ⇒ gọi `POST …/analyze/demo/<id>` và `onStart(jobId)`; user hết lượt (quota 0) vẫn bấm được bài demo (không bị khoá bởi trạng thái quota).
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/analyze/DemoTracks.test.tsx` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces.
- [ ] **Step 4: Chạy lại + lint + build** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(web): add demo track picker`

### Task 18: E2E thật + nghiệm thu DoD

**Files:**
- Create: `web/e2e/analyze.spec.ts`
- Modify: `.agents/context/context.md`, `.agents/memory/memory.md`, `.agents/context/implementation-plan.md` (đánh dấu M2 ✅)

- [ ] **Step 1: Viết spec fail** (`E2E_BACKEND=1`, skip nếu thiếu; user ngẫu nhiên mỗi lần; WAV nhỏ sinh bằng `Buffer` trong test): upload WAV ⇒ thấy stepper ⇒ tới `/charts/<id>`; file `.txt` đổi tên `.mp3` ⇒ thông báo 415 tiếng Việt; URL `http://127.0.0.1/x.mp3` ⇒ thông báo bị chặn (SSRF); URL `http://93.184.216.34/song.mp3` (IP-literal, resolver giả, không ra mạng) ⇒ stepper có bước "Tải" ⇒ chart; 5 lần upload khác nhau rồi lần 6 ⇒ banner 429; bài demo ⇒ chart; `fail-ai.wav` ⇒ lỗi `ai_failure` + nút Thử lại; F5 giữa job ⇒ vẫn tiếp tục theo dõi.
- [ ] **Step 2: Chạy** `E2E_BACKEND=1 rtk npm run test:e2e` — Expected: FAIL tới khi mọi task trước xong; sau đó PASS.
- [ ] **Step 3: Nghiệm thu DoD**: (a) `rtk uv run pytest -v` + `ruff check/format --check` xanh; (b) `rtk npm run lint && rtk npm test -- --run && rtk npm run build` xanh; (c) SSRF suite 100 % (Task 2/3); (d) test p95 ≤ 300 ms (Task 7); (e) mọi mã 413/415/422/429/timeout/`ai_failure` có test BE (T4, T7, T8, T6) **và** FE (T15, T16); (f) migration 0002 lên/xuống sạch.
- [ ] **Step 4: Cập nhật** `context.md` (M2 ✅, blockers: Docker/Redis), `memory.md` (ADR: D1–D7, cách pin IP, cách worker Windows, giới hạn MiB), đánh dấu M2 trong `implementation-plan.md`.
- [ ] **Step 5: Commit** `test(e2e): cover ingestion flows end to end; docs: close milestone 2`

---

## Thứ tự & phụ thuộc

`T1 → {T2 → T3} ∥ T4 ∥ T5 → T6 → T7 → T8 → T9 → {T10 ∥ T11 ∥ T12}`; FE: `T13` cần openapi của T7 (có thể bắt đầu bằng MSW sau T7), `T14` độc lập sau T13, `T15` cần T9 (WS) nhưng test bằng socket giả nên có thể song song, `T16` sau T14+T15, `T17` sau T10+T15, `T18` cuối cùng.

Mốc có thể demo sớm: **sau T7** gọi được upload → status → chart bằng `curl`; **sau T16** dùng được trên trình duyệt.

## Self-review (đã chạy)

- **Phủ spec 2.1–2.8:** 2.1 ⇒ T4, T7, T14; 2.2 ⇒ T2, T3, T7, T14; 2.3 ⇒ T5, T6, T7, T15; 2.4 ⇒ T7 (trừ/429), T8 (hoàn/cache/race), T16; 2.5 ⇒ T9, T15; 2.6 ⇒ T11; 2.7 ⇒ T10, T17; 2.8 ⇒ T12. Danh sách "TDD cần viết trước" của plan gốc: SSRF (T2/T3), biên kích thước (T4/T7), quota (T7/T8), state machine (T1), contract test engine (T5), FE `useJob` (T15) — đủ.
- **Khác plan gốc có chủ đích:** D1–D7 ở đầu tài liệu; "huỷ job" hoãn (D2).
- **Nhất quán tên:** `Job`, `advance`, `to_progress`, `JobRepository.try_mark_refunded`, `ChargeReceipt(user_id, charged_on, True)`, `SafeTarget.pinned_ip`, `check_url`, `build_chart`, `JobExecutor.run/recover_orphans`, `quotaKey` được dùng thống nhất giữa các task.
- **Rủi ro biết trước:** `ffprobe/ffmpeg` có sẵn trên máy dev (ffprobe 8.0.1 đã thấy) nhưng CI/máy khác phải cài; Celery không chạy prefork trên Windows (dùng `--pool=solo`); cách pin IP bằng httpx/httpcore cần thử nghiệm sớm ở T3 (nếu không đạt ⇒ fallback transport tự viết bằng `asyncio.open_connection`).

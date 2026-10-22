# ChordSense Pro — Web App & Core Backend: Implementation Plan

> Trạng thái: **DRAFT chờ duyệt** (2026-10-22). Chưa có dòng code nào được sinh.
> Nguồn: `CLAUDE.md`, `AGENTS.md`, `markdown/use_case_specifications.md` (UC1–UC20), `markdown/ChordSense.MD`, `markdown/Lý Thuyết Hợp Âm Jazz.md`, `latex_last_project/main.tex` (FR-01..41, NFR-01..17, §API Design, §Bounded Contexts), code hiện có trong `DE/` và `landing_page/`.
> Mọi lệnh shell có tiền tố `rtk`.

---

## 0. Hiện trạng đã khảo sát (facts, không phải giả định)

| Hạng mục | Hiện trạng |
| :-- | :-- |
| `DE/` | Chỉ có pipeline DE/AI (Airflow DAGs, Mongo staging, Chain-of-Responsibility handlers). **Chưa có FastAPI app nào** (`main.py` là hello-world). Có `pyproject.toml` với extras `api/db/audio/ai`, pytest `asyncio_mode=auto`. Cấu trúc DDD `src/<context>/{domain,application,infrastructure}`. |
| DB hiện có | `DE/init/init.sql`: `crawl_queue`, `song_analyses` (id VARCHAR(36), `chord_timeline` JSONB, `detected_key`, `tempo_bpm`), `chord_attempts`, `user_chord_mastery`. **Chưa có `users`, `charts`, `playlists`, `chord_dictionary`.** |
| Value objects DE | `shared/domain/value_objects.py`: `RootNote` (12 nốt, **dấu thăng là chuẩn**), `ChordQuality` (major/minor/dominant/diminished/suspended), `ChordExtension` (15 lớp, quy ước Chordie: `Am7`, `Caug`, `Bm7b5`), `ConfidenceScore` (3 head), `ChordLabel.display_name`. **Không có `bass`** (slash chord). |
| Audio chuẩn | Pipeline DE dùng **WAV 44.1 kHz mono**; feature extraction hạ xuống 22.05 kHz. |
| `landing_page/` | 1 file `App.jsx` 1632 dòng (JSX, không TS), Tailwind v4 + Motion + lucide, theme tối `#04040f`, tím `#7C3AED` / cyan `#06B6D4`, font Inter + Outfit. Không có test runner, router, state lib. Lint = `oxlint`. |
| Toolchain máy này | ✅ Node 22.23, npm 12, Python 3.13.5, ffprobe 8.0, git 2.54 (branch `tienkee`, clean). ❌ ~~`uv`~~ (đã cài 2026-10-22), ❌ Docker (user sẽ cài), ❌ `redis-server`. PostgreSQL 18 native có sẵn (service `postgresql-x64-18`, `psql` chưa trên PATH). |

### Mâu thuẫn giữa các tài liệu — cách xử lý đề xuất

| # | Mâu thuẫn | Quyết định đề xuất |
| :-- | :-- | :-- |
| D1 | `ChordSense.MD` ghi **Prisma**; `CLAUDE.md` ghi **SQLAlchemy/Asyncpg** | Theo `CLAUDE.md`: SQLAlchemy 2 async + **Alembic**. |
| D2 | `ChordSense.MD`: bảng `song_charts` 1 cột `chord_events`; `main.tex`: tài liệu JSONB **phân lớp** (AI baseline / master / personal) | Theo `main.tex` (FR-09/36: baseline AI không bao giờ bị ghi đè). Bảng `charts` giữ `ai_layer` JSONB bất biến + `personal_layer` JSONB. |
| D3 | `init.sql` (DE sở hữu `song_analyses`) vs Alembic (Gateway) | DE giữ `song_analyses`; Gateway dùng Alembic cho bảng của mình (`users`, `charts`, `jobs`, `playlists`, …), schema `app`. Hai bên chỉ giao tiếp qua **AnalysisEngine port** (§2), không JOIN chéo. |
| D4 | Roman numeral: yêu cầu của bạn liệt kê trong "output AI"; `ChordSense.MD`/FR-04 quy định tính **ở client** | **AI KHÔNG trả Roman numeral.** AI trả `key` + chord events; Roman/scale là *derived view* do Harmonic Engine (TS, client) tính. Có golden fixtures dùng chung để chống lệch. |
| D5 | Celery (`CLAUDE.md`, yêu cầu của bạn) vs "Redis job queue" (main.tex) | Celery + Redis, **đặt sau `JobRunnerPort`**; có `InProcessJobRunner` cho test/dev vì máy chưa có Redis/Docker. |
| D6 | Jazz notation (`-7`, `Δ`, `ø`) vs Chordie (`Am7`, `Bm7b5`) | Hợp đồng lưu **cấu trúc** (root/quality/extension/bass) + `symbol` chuẩn Chordie. FE có `ChordFormatter` với style `pop` / `jazz` / `basic`. |

---

## 1. Agent Workspace & Memory System

### 1.1 Cấu trúc thư mục (đã khởi tạo trong repo)

```
.agents/
├── memory/
│   └── memory.md          # Quyết định kiến trúc, lỗi đã fix, quy tắc (append-only, có ngày)
├── context/
│   ├── context.md         # Trạng thái hiện tại: endpoint xong / đang làm / blocker
│   └── implementation-plan.md   # file này
└── skills/                # (tạo khi cần) SKILL.md riêng cho repo: contract-first, tdd-fastapi, harmony-golden
```

Lưu ý: thư mục `ChordSensePro/.agents/rules/antigravity-rtk-rules.md` ở **ngoài** git repo (cấp cha) là của Antigravity; `.agents/` mới nằm **trong** repo để được version-control và cả hai agent cùng đọc.

**Quy tắc vận hành**
1. Đầu phiên: đọc `context.md` → `memory.md` (chỉ mục ngắn). Cuối task: cập nhật `context.md`; lỗi/quyết định mới → append `memory.md`.
2. `memory.md` chỉ chứa thứ **không suy ra được từ code/git log**. Mỗi mục: `YYYY-MM-DD · [ADR|FIX|RULE] · tiêu đề · Why · How to apply`.
3. Antigravity review `git diff` + `context.md`; Claude Code không tự sửa tài liệu LaTeX/diagram (đúng phân vai `AGENTS.md`).

### 1.2 MCP servers & skills đề xuất

Nguyên tắc (cùng tinh thần RTK): **ít server, mỗi server phải có việc rõ**, vì mỗi MCP tốn context. Cấu hình ở phạm vi project (`.mcp.json` ở repo root, hoặc `rtk claude mcp add --scope project …`). **Trên Windows**, server chạy qua `npx` cần bọc `cmd /c`. Chưa tạo `.mcp.json` — chờ bạn duyệt.

| Ưu tiên | Mục đích | Repo | Ghi chú xác minh |
| :-- | :-- | :-- | :-- |
| Nên có | Kiểm thử UI/E2E, chụp màn hình Workspace | [microsoft/playwright-mcp](https://github.com/microsoft/playwright-mcp) | Đã xác minh lệnh `claude mcp add playwright npx @playwright/mcp@latest` |
| Nên có (M1+) | Xem schema/chạy query PostgreSQL | [crystaldba/postgres-mcp](https://github.com/crystaldba/postgres-mcp) (Postgres MCP Pro) hoặc [bytebase/dbhub](https://github.com/bytebase/dbhub) | **Không dùng** `@modelcontextprotocol/server-postgres` — đã bị archive, có báo cáo lỗ hổng SQL injection. Dùng user DB **read-only**. Cờ CLI cần kiểm lại README khi cài. |
| Nên có | Docs thư viện cập nhật (FastAPI, SQLAlchemy 2, Vite 8, Tailwind v4) | [upstash/context7](https://github.com/upstash/context7) | Chưa xác minh trong phiên này |
| Tuỳ chọn | Đo hiệu năng Smart Grid (NFR-01) bằng Chrome trace | [ChromeDevTools/chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp) | Chưa xác minh; dùng ở M3 |
| Tuỳ chọn | Fetch trang/doc | [modelcontextprotocol/servers → fetch](https://github.com/modelcontextprotocol/servers) | Repo chính thức, nhưng README nói chỉ là ví dụ tham khảo |
| Không cần | Memory MCP | `servers → memory` | Dùng `.agents/memory/*.md` (git-tracked, Antigravity đọc được) thay vì knowledge-graph ẩn |

Skills tham khảo (đọc/chọn lọc, không cài hàng loạt): [anthropics/skills](https://github.com/anthropics/skills) (chuẩn định dạng SKILL.md), [obra/superpowers](https://github.com/obra/superpowers) (quy trình TDD/debug) — cả hai chưa xác minh trong phiên này.

Cấu hình mẫu `.mcp.json` (đề xuất, chưa ghi file):

```json
{
  "mcpServers": {
    "playwright": { "command": "cmd", "args": ["/c", "npx", "@playwright/mcp@latest"] },
    "postgres":   { "command": "uvx", "args": ["postgres-mcp", "--access-mode=restricted"],
                    "env": { "DATABASE_URI": "postgresql://chordsense_ro:***@localhost:5432/chordsense" } },
    "context7":   { "command": "cmd", "args": ["/c", "npx", "-y", "@upstash/context7-mcp"] }
  }
}
```
Không commit mật khẩu thật; `DATABASE_URI` lấy từ biến môi trường.

---

## 2. API Contract & Data Schema (Mock Strategy)

### 2.1 Nguyên tắc
- **Nguồn sự thật duy nhất = Pydantic v2** (`DE/src/analysis_contract/`). OpenAPI sinh ra → `contracts/openapi.json` → `openapi-typescript` sinh TS types cho `web/`.
- Fixtures JSON dùng chung ở `contracts/fixtures/` — được **validate bởi Pydantic (pytest)** và **type-check bởi tsc (vitest)**.
- Đơn vị thời gian: **số nguyên mili-giây tuyệt đối** (không lưu Grid/Bar — rủi ro đã nêu trong `ChordSense.MD §4`).
- `schema_version: "1.0"`; đổi breaking → tăng major + migration (rule D-Contract trong memory.md).

### 2.2 Ranh giới Gateway ↔ AI (port)

```python
# DE/src/analysis_contract/ports.py  (đặc tả, chưa viết code)
class AnalysisEngine(Protocol):
    async def analyze(self, req: AnalysisRequest,
                      on_progress: Callable[[JobProgress], Awaitable[None]]) -> AnalysisResult: ...
```
Adapters: `MockAnalysisEngine` (M2) → `PipelineAnalysisEngine` / `HttpAnalysisEngine` (M5, do DE/AI bàn giao).
**Contract test suite** `DE/tests/contract/test_analysis_engine_contract.py` parametrize theo adapter: đây là tiêu chí nghiệm thu cho team DE/AI.

### 2.3 Models (Pydantic — phác thảo)

```python
class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

# ── Vào ───────────────────────────────────────────────
class SourceKind(StrEnum): FILE="file"; URL="url"; RECORDING="recording"; DEMO="demo"

class AnalysisRequest(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    job_id: UUID
    content_hash: str                 # sha256 của waveform đã chuẩn hoá
    source_kind: SourceKind
    audio_path: str                   # WAV 44.1 kHz mono do Normalizer (Gateway) tạo
    source_url: HttpUrl | None = None # chỉ để audit, AI không tự fetch
    options: AnalysisOptions = AnalysisOptions()   # key_hint, model_version

# ── Tiến độ ───────────────────────────────────────────
class JobState(StrEnum): QUEUED="queued"; PROCESSING="processing"; DONE="done"; FAILED="failed"
class JobStage(StrEnum): FETCHING="fetching"; NORMALIZING="normalizing"; SEPARATING="separating"; \
                         BEAT_TRACKING="beat_tracking"; RECOGNIZING="recognizing"
class JobProgress(ContractModel):
    job_id: UUID; state: JobState; stage: JobStage | None; progress: float  # 0..1
    error: JobError | None = None; result_chart_id: UUID | None = None
class ErrorCode(StrEnum):  # ánh xạ HTTP
    UNSUPPORTED_FORMAT; FILE_TOO_LARGE; DURATION_TOO_LONG; URL_UNRESOLVABLE   # 415/413/413/422
    SSRF_BLOCKED; QUOTA_EXCEEDED; ANALYSIS_TIMEOUT; AI_FAILURE               # 422/429/504/502

# ── Ra ────────────────────────────────────────────────
class ChordLabelDTO(ContractModel):          # khớp DE ChordLabel
    root: RootNote; quality: ChordQuality; extension: ChordExtension
    bass: RootNote | None = None             # ⚠ DE chưa có — xem Q3
class Confidence(ContractModel): root: float; quality: float; extension: float  # 0..1

class ChordEvent(ContractModel):
    event_id: str                            # ổn định trong chart ("e0001")
    start_ms: int; end_ms: int               # end_ms > start_ms, không chồng lấn
    chord: ChordLabelDTO
    symbol: str                              # canonical Chordie: "Am7", "G7#9", "C/E"
    confidence: Confidence | None
class TimeSignature(ContractModel): numerator: int; denominator: int
class BeatGrid(ContractModel):
    tempo_bpm: float; tempo_confidence: float
    time_signature: TimeSignature; first_beat_offset_ms: int
    downbeats_ms: list[int] | None = None    # nếu có thì ưu tiên hơn nội suy từ tempo
class KeyEstimate(ContractModel): tonic: RootNote; mode: Literal["major","minor"]; confidence: float
class AnalysisMetadata(ContractModel):
    duration_ms: int; key: KeyEstimate; beat_grid: BeatGrid
    model_version: str; pipeline_version: str
class AnalysisResult(ContractModel):
    schema_version: Literal["1.0"]; job_id: UUID; content_hash: str
    metadata: AnalysisMetadata; chord_events: list[ChordEvent]
```

**Chart (Gateway lưu, FE nhận)** — lớp hoá theo D2:
```python
class Chart(ContractModel):
    id: UUID; owner_id: UUID; title: str; artist: str | None
    source: ChartSource            # kind, url?, content_hash, playback
    metadata: AnalysisMetadata
    ai_events: list[ChordEvent]            # bất biến
    overrides: list[EventOverride]         # lớp cá nhân: {event_id, op: set|delete|insert, event?}
    is_customized: bool; forked_from: UUID | None; version: int
```
```python
class PlaybackSource(ContractModel):   # cách FE phát audio của chart
    kind: Literal["file", "youtube", "soundcloud", "direct"]
    audio_url: str | None = None       # kind=file|direct: URL do Gateway cấp (Range)
    embed_id: str | None = None        # kind=youtube|soundcloud: video id / track url
```
`file` = upload/demo (audio lưu riêng tư theo owner); `youtube|soundcloud` = chỉ lưu id nhúng, **server không giữ audio**.

`PATCH /charts/{id}/events` nhận danh sách thao tác chạm (không gửi lại cả mảng), server gắn `is_user_edited`, dùng `version` làm optimistic lock (409 khi lệch). *Effective events* = `apply(ai_events, overrides)` — hàm thuần, có bản TS và test chung fixture.

**Derived view (chỉ ở FE, KHÔNG nằm trong AI contract)**
```ts
interface HarmonicAnnotation { roman: string; degree: number; fn: 'tonic'|'subdominant'|'dominant'|'other';
  scales: { name: 'Ionian'|'Dorian'|'Phrygian'|'Lydian'|'Mixolydian'|'Aeolian'|'Locrian'|'Altered'|…; confidence: 'primary'|'alt' }[];
  diatonic: boolean; custom?: boolean }
```

### 2.4 Endpoints (đúng `main.tex §API Design` + `ChordSense.MD §5`)

Phạm vi iteration này (✅) và hoãn (⏸):
```
✅ POST /api/v1/auth/register|login|refresh   GET /api/v1/users/me   GET /api/v1/users/me/quota
✅ POST /api/v1/analyze/file | /analyze/url   → 202 {job_id}
✅ GET  /api/v1/analyze/status/{id}           WS /api/v1/ws/jobs/{id}
✅ GET  /api/v1/charts   GET /charts/{id}   PATCH /charts/{id}/events   POST /charts/{id}/fork
✅ GET  /api/v1/charts/{id}/audio             (HTTP Range — bắt buộc cho seek)
✅ CRUD /api/v1/playlists  + POST /playlists/{id}/add  (+ reorder, UC4)
✅ GET  /api/v1/dictionary/search?root=&quality=&instrument=
✅ GET  /api/v1/demo-tracks                   (UC1 V1 — catalog demo)
✅ GET  /api/v1/charts/{id}/export?format=chordpro|json   (Pro)
⏸ /analyze/recording, /recognize, /classes*, /telemetry*, /feedback*, /admin/*, /charts (MIDI save)
```

### 2.5 Cơ chế Mock (3 lớp, FE độc lập ngay từ ngày 1)

1. **Backend mock thật** — `ANALYSIS_ENGINE=mock`: `MockAnalysisEngine` đọc fixture theo `content_hash`/tên demo, **mô phỏng vòng đời job** (queued→processing[5 stage]→done) với `MOCK_STAGE_DELAY_MS` chỉnh được và hook lỗi (`?mock_fail=AI_FAILURE`, `timeout`) để FE test các nhánh EF. Fixture nhận diện bằng tên file/URL chứa `fail-*`.
2. **MSW handlers cho FE** — sinh từ cùng fixtures: FE chạy `npm run dev:mock` không cần backend (và là nền cho Vitest + Playwright).
3. **Fixtures vàng** (`contracts/fixtures/`):
   - `charts/` ≥ 5 bài: ii–V–I Cmaj, 12-bar blues F7, Autumn Leaves (G minor, có slash `C/E`, `Bm7b5`), pop I–V–vi–IV (kiểm tra Basic/Precise: `G9sus4`→`G`), 1 bài 3/4 và 1 bài 6/8, 1 bài tempo đổi/offset ≠ 0.
   - `jobs/` — chuỗi progress mẫu + từng `ErrorCode`.
   - `harmony_golden.json` — `(key, chord) → (roman, scale)`; dùng ở M4 cho TS, và cho validator Python.
   - **Audio demo**: script `scripts/gen_demo_audio.py` tổng hợp tiếng click+hợp âm khớp đúng timeline (≤ vài trăm KB, **gitignored**, theo luật không commit `.wav/.mp3`). Nhờ đó M3 kiểm chứng sai số sync thật.

---

## 3. Kế hoạch triển khai

### Quy ước chung
- **Layout mới**: BE tại `DE/src/{api,identity,ingestion,chart_library,dictionary,analysis_contract}` (đúng bounded contexts `main.tex`); FE tại **`web/`** (Vite 8 + React 19 + **TypeScript**), tách khỏi `landing_page/` (xem Q1); engine thuần TS tại `web/src/engine/{audio,grid,harmony}` không phụ thuộc React.
- **FE stack**: React Router, TanStack Query (server state), Zustand (workspace), Tailwind v4 (tái dùng token màu landing), Vitest + Testing Library + MSW, Playwright (E2E), oxlint + `tsc --noEmit`.
- **TDD**: test đỏ → xanh → refactor; commit nguyên tử `type(scope): …`. Test DB-thật gắn `@pytest.mark.integration`, tự skip nếu thiếu `TEST_DATABASE_URL`.
- **Đã trừ khỏi phạm vi**: Education (UC10–13), Admin (UC5–8, 14–17), MIDI (UC9), mic-recording, feedback (UC19), Real-Time Practice (FR-11) → backlog "Won't (iteration này)". Roles enum vẫn khai báo đủ 8 vai trò để không phải migrate sau.

### MILESTONE 1 — Foundation, DB, Auth, Workspace Shell (P0)

| ID | Pri | Task | Backend | Frontend |
| :-- | :-- | :-- | :-- | :-- |
| 1.0 | **P0 blocker** | Unblock môi trường | `uv` ✅ đã cài; tạo DB `chordsense_app` + role trên PG18 native; Docker Desktop (bạn cài) cho Redis; tránh trùng cổng 5432 với compose DE (Q4). `rtk uv sync --all-extras`; chạy `rtk uv run pytest` và `ruff check` trên code hiện có để lấy baseline (chưa có `[tool.ruff]`, code cũ có thể vi phạm → khoanh ruff vào package mới hoặc thêm per-file-ignores). | `web/` scaffold: `npm create vite` (TS), cài deps, `rtk npm run lint/build/test` xanh. |
| 1.1 | P0 | Contract package | `analysis_contract/` models §2.3 + `contracts/fixtures` + script sinh `openapi.json` | `openapi-typescript` → `web/src/api/generated`; script `npm run gen:api` |
| 1.2 | P0 | DB schema (Alembic) | `users`, `charts`, `jobs`, `playlists`, `playlist_items`, `chord_dictionary`, `usage_counters`; JSONB cho layers; index `(owner_id, updated_at)`, `content_hash` unique-per-scope | — |
| 1.3 | P0 | Auth + RBAC cơ bản | register/login/refresh/me; bcrypt cost ≥ 12 (prod; thấp hơn khi test qua setting); JWT TTL cấu hình; `require_role/tier` dependency; policy quota đọc từ `PolicyProvider` (không hard-code 5) | Trang Login/Register, `AuthProvider`, route guard, token refresh interceptor |
| 1.4 | P0 | App factory + lỗi chuẩn | `create_app()`, `/healthz`, handler lỗi dạng `{code,message,detail}`, structlog request-id, CORS cho dev | API client typed (fetch wrapper), error→toast mapper |
| 1.5 | P0 | Workspace Shell | — | Layout: sidebar (Library/Workspace/Dictionary), header có **Quota/Tier badge** (FR-18), theme từ landing, trang trống có skeleton; i18n-ready (vi/en) |

**TDD cần viết trước**
- BE unit: hash/verify password (cost); JWT hết hạn/giả mạo/sai alg → 401 (100% case, NFR-09); register trùng email → 409; ma trận role×endpoint (NFR-10) bản đầu; `GET /users/me/quota` Free = 5, reset đúng 00:00 UTC (hoặc múi giờ cấu hình).
- BE integration (skip khi thiếu DB): `alembic upgrade head` → `downgrade base` sạch; repo CRUD `users`.
- Contract: mọi fixture parse được bằng Pydantic; `openapi.json` không đổi so với bản commit (drift check).
- FE: route guard chuyển hướng khi chưa login; form validation; interceptor refresh 1 lần rồi logout; badge hiển thị `remaining/limit`.

**DoD**: `rtk uv run pytest -v` + `ruff check/format --check` xanh cho package mới; `rtk npm run lint && build && test` xanh; đăng ký→đăng nhập→thấy Shell với quota badge chạy được end-to-end (Playwright smoke); migration lên/xuống sạch; `context.md` + `memory.md` cập nhật.

### MILESTONE 2 — Audio & Link Ingestion + Mock Pipeline (P0)

| ID | Pri | Task | Backend | Frontend |
| :-- | :-- | :-- | :-- | :-- |
| 2.1 | P0 | Upload file | `POST /analyze/file`: multipart, kiểm MIME+magic bytes+`ffprobe` (≤100 MB, ≤10 phút → 413; định dạng lạ → 415), Normalizer → WAV 44.1k mono, tính `content_hash` | Source Selector (tab Upload; tab Record/Link hiển thị trước nhưng Record disabled), dropzone, kiểm tra kích thước phía client |
| 2.2 | P0 | **SSRF guard + URL** | `POST /analyze/url`: chỉ http/https; resolve DNS rồi **pin IP**; chặn private/loopback/link-local/metadata (`169.254.169.254`), IPv6 tương đương, kiểm lại ở **mỗi redirect**; giới hạn size/time. `MediaResolverPort` → `FakeMediaResolver` (trả demo audio). yt-dlp thật hoãn đến 5.3 | Tab Link: validate URL sơ bộ, thông báo lý do cụ thể (EF3), gợi ý "upload thay thế" (AF1) |
| 2.3 | P0 | Job lifecycle | `JobRunnerPort` + `InProcessJobRunner`; bảng `jobs`; `MockAnalysisEngine`; trả **202 + job_id ≤ 300 ms** (NFR-06); ghi chart khi done | Job tracker UI: stepper theo `stage`, progress, huỷ/thử lại |
| 2.4 | P0 | Quota | Trừ quota lúc nhận job, **hoàn khi failed/timeout** (UC1 EF1), lần thứ 6 → 429 kèm `reset_at`; cache theo `content_hash` → trả chart cũ **không trừ quota** | Banner 429 + giờ reset + CTA nâng cấp (EF2) |
| 2.5 | P1 | Realtime | `WS /ws/jobs/{id}` (auth qua token query/subprotocol), đẩy đổi trạng thái ≤ 1 s; polling `GET /analyze/status/{id}` | Hook `useJob`: WS trước, rớt thì **polling 2 s** (AF2); hiển thị đúng khi F5 giữa chừng |
| 2.6 | P1 | Celery adapter | `CeleryJobRunner` + Redis, `task_always_eager` cho test, timeout 90 s → `ANALYSIS_TIMEOUT` (EF1) | — |
| 2.7 | P1 | Demo catalog | `GET /demo-tracks` + `POST /analyze/demo/{id}` (không trừ quota) | Nút "Thử bài demo" ngay trên Shell |
| 2.8 | P2 | Dọn dẹp | TTL xoá file tạm/WAV sau xử lý | — |

**TDD cần viết trước**
- SSRF suite đối kháng (NFR-16 tinh thần): `file://`, `ftp://`, `http://127.0.0.1`, `http://[::1]`, `http://10.0.0.5`, `http://169.254.169.254`, số thập phân/hex/octal (`http://2130706433`), DNS trỏ về private, **redirect 302 → private**, DNS-rebinding (lần resolve 2 khác lần 1) → **100% bị chặn**; URL hợp lệ công khai → qua.
- Biên kích thước: đúng 100 MB OK / +1 byte → 413; đúng 10:00 OK / 10:01 → 413 (NFR-12); file hỏng/giả đuôi `.mp3` → 415.
- Quota: 5 lần OK, lần 6 → 429; job lỗi → quota hoàn lại; cache hit không trừ; 2 request đồng thời không vượt quota (race, dùng transaction/`SELECT … FOR UPDATE`).
- Job state machine: chuyển trạng thái hợp lệ duy nhất; không `done` → `processing`.
- Contract test cho `MockAnalysisEngine` (sẽ tái dùng ở M5).
- FE: `useJob` rớt WS → chuyển polling; trạng thái failed hiển thị đúng thông điệp theo `ErrorCode`; refresh trang giữa job vẫn tiếp tục theo dõi.

**DoD**: upload mp3 nhỏ hoặc dán URL công khai → thấy stepper chạy → tự mở chart mock; mọi nhánh lỗi (413/415/422/429/timeout/AI_FAILURE) có test BE + hiển thị FE; SSRF suite 100%; POST p95 ≤ 300 ms trên mock (đo bằng test); CI-lệnh local xanh.

### MILESTONE 3 — Interactive Chord Viewer & Audio Sync Player (P1)

| ID | Pri | Task | Backend | Frontend |
| :-- | :-- | :-- | :-- | :-- |
| 3.1 | P1 | Chart API | `GET /charts`, `GET /charts/{id}` (+ETag), `GET /charts/{id}/audio` hỗ trợ **Range** (chỉ cho `kind=file`; chart YouTube/SoundCloud trả 404 vì không giữ audio) | Library list, chart loader |
| 3.2 | P1 | **Spike Smart Grid (NFR-01)** — làm trước | — | Prototype đo offset highlight vs `currentTime`, p95 ≤ 100 ms ở Chromium+Firefox; quyết định render (DOM+CSS-transform hay canvas) dựa trên số đo |
| 3.3 | P1 | Grid engine | — | `grid/`: từ `BeatGrid` + events → bars/beats (nội suy từ tempo, `first_beat_offset_ms`, đổi nhịp 3/4, 6/8; ưu tiên `downbeats_ms` nếu có); ánh xạ ms↔bar; hàm thuần |
| 3.4 | P1 | Transport đa nguồn (`PlaybackSource`) | — | Interface `Transport {play, pause, seek(ms), setRate, currentTimeMs(), on('state'|'error')}` với 3 adapter: **`HtmlAudioTransport`** (`<audio>` + `MediaElementAudioSourceNode`, `preservesPitch` giữ cao độ khi 0.5×/0.75×; `AudioBufferSource.playbackRate` đổi cao độ nên không dùng); **`YouTubeTransport`** (YouTube IFrame Player API, `setPlaybackRate` 0.5/0.75 giữ cao độ); **`SoundCloudTransport`** (Widget API). Video phát cạnh bảng hợp âm như Chordify. Space & ←/→; unlock AudioContext bằng modal (UC2 2a). Xử lý: video bị chặn/xoá/giới hạn nhúng → thông báo + gợi ý upload file (AF1); quảng cáo/buffering làm lệch → đồng bộ lại theo `onStateChange`. |
| 3.5 | P1 | Playhead | — | rAF loop **ngoài React state** cập nhật class ô nhịp hiện tại; giảm tần suất khi máy yếu (7a) |
| 3.6 | P1 | Click-to-seek & A/B loop | — | Click ô → seek ms chính xác; kéo chọn nhiều ô → loop; wrap-around ≤ 100 ms; Clear Loop |
| 3.7 | P1 | Metronome | — | Look-ahead scheduler trên `AudioContext` clock; accent beat 1; theo tốc độ phát; BPM 30–280 + tap tempo (manual mode miễn phí) |
| 3.8 | P1 | Basic/Precise toggle | — | `ChordFormatter` `basic` (`G9sus4`→`G`) tức thì, không gọi API |
| 3.9 | P2 | Tier gating | 403 cho tính năng Pro | Free: loop đơn 1×; Pro: speed, metronome-sync, count-in, lưu loop + upsell prompt |

**TDD cần viết trước**
- Grid: 4/4@120 BPM offset 0 → bar k bắt đầu `k·2000 ms`; offset 350 ms; 3/4; 6/8 (2 xung chính); tempo không nguyên; sự kiện nằm giữa 2 bar; `downbeats_ms` ghi đè nội suy; bài rỗng.
- `msToBar/barToMs` là cặp nghịch đảo (property test).
- Loop controller: playhead chạm B → seek A; B ≤ A bị từ chối; speed đổi không lệch vị trí loop.
- Metronome scheduler dùng **`OfflineAudioContext`** → so onset với grid, sai số ≤ 10 ms (NFR-04).
- Formatter Basic/Precise: bảng input→output (kể cả slash `C/E`, `Bm7b5`, `Caug`).
- BE: `Range` request trả 206 đúng byte; chart của user khác → 404/403.
- Playwright: bấm ô nhịp 9 → `currentTime` ≈ thời điểm bar 9 (±100 ms).
- Transport contract test chạy chung cho 3 adapter (play/seek/rate/error), với IFrame/Widget API được mock; `YouTubeTransport`: video lỗi 101/150 → emit `error` + gợi ý upload.
- Spike 3.2 đo NFR-01 **riêng cho từng loại nguồn**; với YouTube/SoundCloud nếu p95 > 100 ms thì ghi nhận giới hạn và dùng bù trễ (calibration offset) thay vì giả định.

**DoD**: mở demo chart, phát nhạc, ô nhịp sáng đúng, click-seek, loop bar 9–16, 0.75× không méo cao độ, metronome trùng beat; số đo NFR-01 được ghi vào `memory.md`; không re-render React theo từng frame (xác nhận bằng profiler/Chrome trace).

### MILESTONE 4 — Music Theory / Jazz Harmony (P1)

| ID | Pri | Task | Backend | Frontend |
| :-- | :-- | :-- | :-- | :-- |
| 4.1 | P1 | Harmonic Engine lõi | Parser/validator ký hiệu hợp âm (Python) để `PATCH` từ chối symbol không hợp lệ | `harmony/`: pitch-class, `RootNote`↔semitone, **speller theo key** (Bb vs A#; DE lưu sharp-canonical), Roman numeral theo tonic+mode (hoa=trưởng, thường=thứ, `°`, `ø`, `♭VII`, `♯iv°`) |
| 4.2 | P1 | Chord-scale mapping | — | Bảng: Imaj7→Ionian, ii7→Dorian, iii7→Phrygian, IVmaj7→Lydian, V7→Mixolydian, vi7→Aeolian, viiø→Locrian; V7 có alteration→Altered/Lydian♭7…; `primary` + `alt` |
| 4.3 | P1 | Hiển thị | — | Roman + scale dưới tên hợp âm; style `pop`/`jazz` (`-7`, `Δ`, `ø`); tooltip cấu thành nốt |
| 4.4 | P1 | Edit Marker (PATCH) | `PATCH /charts/{id}/events` (ops, `version` lock, `is_user_edited`, 403 cho Free ở cloud → lưu local), `fork` | Modal 4 bước tối giản: gợi ý tương đương (Exact/Partial) → in-place/fork → picker → custom; **recompute ≤ 50 ms @400 events (NFR-03)**; Free: IndexedDB; "Reset to AI" |
| 4.5 | P2 | Chức năng nâng cao | — | Secondary dominants (`V7/ii`), tritone sub, borrowed chords (`♭VI`, `iv`), key minor (harmonic/melodic), modal interchange |
| 4.6 | P2 | Dictionary | `chord_dictionary` seed + `GET /dictionary/search` | Panel Fretboard/Piano + audition bằng Web Audio synth (soundfont) |
| 4.7 | P2 | Export | `GET /charts/{id}/export` ChordPro/JSON (Pro) | Nút export + gate |

**TDD cần viết trước**: `harmony_golden.json` — tối thiểu: C major {Cmaj7→Imaj7/Ionian, Dm7→ii7/Dorian, Em7→iii7/Phrygian, Fmaj7→IVmaj7/Lydian, G7→V7/Mixolydian, Am7→vi7/Aeolian, Bm7b5→viiø/Locrian}; A minor; Eb major (speller: `Bb` không phải `A#`); chromatic (`Db7`→♭II7 / tritone-sub); slash (`C/E`); quality `sus`, `dim7` vs `m7b5`; mọi 12 root × 5 quality không throw. Exact vs Partial matches (`Cmaj9` ↔ `Em7/C` Exact; `G/C` Partial thiếu 3rd). Edit: PATCH idempotent, version conflict → 409, symbol sai → 422, Reset to AI khôi phục đúng. Ma trận Free/Pro. Property test: transpose key + chord cùng số semitone ⇒ Roman không đổi.

**DoD**: golden fixtures 100% xanh; lệch với `markdown/Lý Thuyết Hợp Âm Jazz.md` = 0 (vai trò ii/V/I đúng); sửa hợp âm cập nhật Roman/Scale ≤ 50 ms (đo 4× CPU throttle); sửa của Free không gọi API ghi cloud (kiểm bằng MSW "no request").

### MILESTONE 5 — Tích hợp AI thật (P2)

| ID | Pri | Task | Nội dung |
| :-- | :-- | :-- | :-- |
| 5.1 | P2 | Thoả thuận contract với DE/AI | Review §2.3 cùng DE/AI; chốt bass/slash (Q3), `downbeats_ms`, confidence; đóng băng `schema_version 1.0` |
| 5.2 | P2 | Adapter thật | `PipelineAnalysisEngine` (đọc `song_analyses` do DAG ghi, hoặc HTTP tới AI service) cài `AnalysisEngine`; chuyển đổi `chord_timeline` cũ → `ChordEvent[]` (có test mapper với dữ liệu thật từ DE) |
| 5.3 | P2 | yt-dlp thật | `YtDlpMediaResolver` chạy sau SSRF guard, sandbox/timeout, AF1 fallback; chính sách xoá audio sau xử lý (Q2) |
| 5.4 | P2 | Chạy contract suite | `ANALYSIS_ENGINE=pipeline` → suite §2.2 xanh; so sánh latency với NFR-05 (15–45 s) |
| 5.5 | P2 | Cắt mock | Mock giữ lại cho CI/dev; flag cấu hình; E2E Playwright trên cả hai engine |

**TDD**: contract suite chạy cho cả hai adapter; mapper test với ≥ 3 timeline thật; test lỗi AI (`AI_FAILURE`, timeout → hoàn quota).
**DoD**: một bài thật đi qua upload → AI thật → chart hiển thị đúng; không đổi code FE; mọi test cũ vẫn xanh (không silent breakage — `AGENTS.md §6.1`).

---

## 4. Rủi ro & phương án

| Rủi ro | Giảm thiểu |
| :-- | :-- |
| Không có Docker/Postgres/Redis | Port + fake in-memory cho unit; integration tự skip; xử lý ở task 1.0 |
| `DE` ruff baseline có thể đỏ | Đo ở 1.0; khoanh phạm vi, không "format cả repo" làm nhiễu diff của Antigravity |
| Sync <100 ms trên trình duyệt | Spike 3.2 trước khi dựng UI; đo bằng số, không cảm tính |
| Lệch Roman numeral giữa Py/TS | Một engine (TS) + golden JSON chung; Python chỉ parse/validate |
| Bản quyền audio URL | Đã chốt Q2: không lưu/phát lại audio URL; phát bằng player nhúng chính thức, server xoá audio tạm sau phân tích |
| Sync với player nhúng kém chính xác hơn `<audio>`; metronome (Web Audio clock) lệch khi nguồn là YouTube | Đo riêng ở spike 3.2; metronome chế độ YouTube dùng hiệu chỉnh theo `getCurrentTime()` và ghi giới hạn vào `memory.md` |
| Contract thay đổi muộn từ team AI | `schema_version`, `extra="forbid"`, drift check OpenAPI trong test |

---

## 5. Quyết định đã chốt (2026-10-22)

1. **Q1 — FE**: `web/` mới (TypeScript); `landing_page/` giữ làm marketing site. ✅
2. **Q2 — Phát lại audio**: **player theo nguồn** (`PlaybackSource`), giống trải nghiệm Chordify: link YouTube/SoundCloud → phát bằng **player nhúng chính thức**, hợp âm chạy đồng bộ theo `getCurrentTime()`; upload/demo → `<audio>` HTML5. Server vẫn tải audio tạm để phân tích rồi xoá, **không phát lại/redistribute** audio URL. Chi tiết ở task 3.4 và §2.3 (`ChartSource.playback`). ✅
   *Cơ sở*: tài liệu hỗ trợ của Chordify nói khi nguồn là YouTube thì video phát song song cạnh bảng hợp âm, con trỏ chỉ hợp âm hiện tại, bấm hợp âm để nhảy tới đoạn đó; video không khả dụng thì không lấy được hợp âm/âm thanh. Chordify không công bố chi tiết cài đặt player nên việc dùng IFrame API là **suy luận của tôi**, không phải xác nhận.
3. **Q3 — `bass` optional**: *bass* là nốt trầm nhất của hợp âm đảo/slash, ví dụ `C/E` = hợp âm C trưởng nhưng nốt trầm là E. Mô hình AI của DE hiện chỉ cho `root + quality + extension` (không có bass), nên trong contract trường `bass` **có mặt nhưng được phép `null`**: AI trả `null` ⇒ hiển thị `C`; khi người dùng tự sửa thành `C/E` thì `bass = E`. Câu hỏi còn lại cho team AI (không chặn tiến độ): sau này mô hình có nhận diện slash chord không. ✅ (mặc định)
4. **Q4 — Hạ tầng**: bạn cài Docker Desktop; **PostgreSQL 18 native đã có** (service `postgresql-x64-18`). Gateway dùng DB `chordsense_app` trên PG18 native; Docker dùng cho Redis (và stack Airflow của DE khi cần). ⚠ `DE/docker-compose.yml` đang map container `postgres:16` ra cổng **5432** — trùng với PG18 native; không bật cùng lúc, hoặc đổi cổng map sang 5433. SQL của Gateway phải tương thích cả 16 (compose/prod) lẫn 18 (dev). ✅
5. **Q5 — Phạm vi**: Education/Admin/MIDI/Feedback/mic-recording ngoài iteration này. ✅
6. **Q6 — Git**: commit nguyên tử trên branch `tienkee`, không push khi chưa được yêu cầu. ✅

## 6. Công cụ đã cài (2026-10-22)

| Công cụ | Trạng thái |
| :-- | :-- |
| `uv` 0.13.5 | ✅ cài bằng `pip install uv` (chạy được `rtk uv …`) |
| Plugin **superpowers** (TDD, systematic-debugging, writing-plans…) | ✅ `superpowers@claude-plugins-official`, phạm vi project (`.claude/settings.json`). Có hook SessionStart → có hiệu lực từ **phiên mới**. |
| MCP `playwright`, `context7`, `postgres` | ✅ khai báo trong `.mcp.json`, đã kiểm tra package tải/chạy được. ⏸ Claude Code yêu cầu **bạn duyệt 1 lần** khi mở `claude` (cơ chế bảo mật của MCP cấp project, tôi không tự vượt). |
| `postgres` MCP | Đã đổi sang **`@bytebase/dbhub`** (postgres-mcp của crystaldba không cài được trên Windows, xem memory.md). Dùng biến `CHORDSENSE_RO_DATABASE_URI` (role `chordsense_ro`, chỉ `SELECT`). ✅ đã tạo DB/role trên PG18. |
| Trình duyệt Playwright | Cài lúc dựng `web/` (task 1.0) bằng `npx playwright install chromium`. |
| Docker Desktop | Bạn tự cài. |

# M3 — Interactive Chord Viewer & Audio Sync Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Người dùng mở một chart (bài demo, file đã tải lên, hoặc link YouTube/SoundCloud) trong Practice Workspace: nghe nhạc, thấy ô nhịp sáng đúng theo thời gian, bấm ô để tua, kéo chọn bar 9–16 để lặp, phát 0.75× không méo cao độ, bật metronome trùng beat, đổi Basic/Precise — với độ lệch highlight ≤ 100 ms (NFR-01).

**Architecture:** Backend thêm danh sách chart, ETag và endpoint audio hỗ trợ HTTP Range xác thực bằng *ticket* ký ngắn hạn (vì `<audio>` không gửi được header). Frontend tách thành các mô-đun thuần (`grid/`, `format/`, `loop/`, `metronome/`) không phụ thuộc React, một lớp `Transport` đa nguồn (`<audio>`, YouTube IFrame, SoundCloud Widget) và UI mỏng; playhead chạy bằng `requestAnimationFrame` ghi thẳng vào DOM, **không** qua React state. Một spike đo NFR-01 trên Chromium + Firefox quyết định cách render trước khi dựng UI đầy đủ.

**Tech Stack:** FastAPI/SQLAlchemy Core (đã có), React 19 + TanStack Query + Vitest + Testing Library + MSW + Playwright (Chromium **và Firefox**), Web Audio API (`AudioContext`, `OfflineAudioContext`), YouTube IFrame Player API, SoundCloud Widget API. Không thêm thư viện runtime mới.

**Spec:** `.agents/context/implementation-plan.md` §3 "MILESTONE 3" (bảng 3.1–3.9, TDD, DoD); `markdown/use_case_specifications.md` UC2 (2a, 4b, 5a, 7a); `latex_last_project/main.tex` FR-03, FR-05, FR-06, FR-07, FR-14 và NFR-01, NFR-04; contract `DE/src/analysis_contract/models.py` (`BeatGrid`, `ChordEvent`, `EventOverride`, `PlaybackSource`).

## Quyết định cần bạn xác nhận trước khi chạy (mặc định = cách plan đang viết)

| # | Vấn đề | Mặc định của plan | Vì sao |
| :-- | :-- | :-- | :-- |
| D1 | `<audio src>` không gửi được `Authorization`, nhưng endpoint audio phải riêng tư theo chủ chart và hỗ trợ Range | **Ticket ký**: `POST /charts/{id}/audio-ticket` → URL `…/audio?ticket=…` (JWT loại `media`, gắn user + chart, TTL 30 phút); endpoint audio nhận Bearer *hoặc* ticket. Ticket hết hạn giữa chừng ⇒ FE xin ticket mới và phát tiếp tại vị trí cũ | Giữ streaming + Range thật; cách "tải cả file thành Blob" tốn RAM với bài 10 phút và mất tải dần |
| D2 | Plan gốc dùng `MediaElementAudioSourceNode` | **Bỏ**: `<audio>` phát trực tiếp (`preservesPitch = true`), `AudioContext` chỉ dùng cho metronome | Đỡ một tầng độ trễ và yêu cầu CORS; giữ time-stretch có sẵn của trình duyệt. Chênh lệch trễ đầu ra giữa `<audio>` và `AudioContext` xử lý bằng *calibration offset* (đo ở T9) |
| D3 | Task 3.9 "403 cho tính năng Pro" và "lưu loop" | M3 chỉ **gate ở FE** theo `user.role ∈ {paid, teacher, student}` (đúng `PRO_ROLES` của BE). Lưu loop xuống cloud **hoãn sang M4** cùng lớp chỉnh sửa cá nhân (cần bảng + endpoint mới) | M3 chưa có endpoint nào cần chặn 403; thêm bảng chỉ để chặn là phạm vi thừa |
| D4 | "Basic" của FR-14 chưa định nghĩa đủ | Basic = **root + (`m` nếu minor, `dim` nếu diminished)**, bỏ mọi extension và sus (`G9sus4→G`, `G7→G`, `Cmaj9→C`, `Bm7b5→Bdim`, `Caug→C`), **giữ slash bass** (`C/E`). Precise = `event.symbol` nguyên bản | Khớp ví dụ của FR-14; slash là nghịch đảo của triad nên giữ |
| D5 | Phím ←/→ (UC2 biến thể 3) | ←/→ tua **một bar**, `Shift+←/→` tua 5 s; Space phát/dừng (bỏ qua khi đang gõ trong ô nhập) | UC2 không nói đơn vị tua; bar là đơn vị tự nhiên của lưới |
| D6 | NFR-01 trên YouTube/SoundCloud | Đo **thật** chỉ cho `<audio>` (Chromium + Firefox). Embed không đo được offline: kiểm thử bằng mock + hiệu chỉnh `calibration offset` theo cấu hình, và ghi vào `memory.md` là **chưa đo thật**; cần bạn phát thử tay một lần với link thật. SoundCloud Widget không có `setPlaybackRate` ⇒ tắt điều khiển tốc độ cho nguồn này | Không có mạng/embed thật trong CI; nói thẳng giới hạn thay vì giả định |
| D7 | Playwright mới chỉ có Chromium | Thêm project **Firefox** (`npx playwright install firefox`, ~90 MB) chỉ cho các spec đo NFR (`*.perf.spec.ts`) | NFR-01 yêu cầu cả hai trình duyệt |
| D8 | Trang `/charts/:id` hiện là bảng hợp âm tạm | Trở thành **Practice Workspace**; bảng cũ bị thay (test cũ của `ChartPage` được viết lại) | Đúng ý "M3 thay trang tạm" của plan M2 |

## Global Constraints

- Mọi lệnh có tiền tố `rtk`; `DE/` dùng `rtk uv run …`, `web/` dùng `rtk npm …`. TDD: test viết trước, **chạy thấy fail đúng lý do**, rồi mới code; commit nguyên tử trên `tienkee`, không push; commit kết thúc bằng `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Thời gian là **số nguyên ms**; không lưu Grid/Bar vào DB (tính ở client từ `BeatGrid` + events).
- NFR-01: độ lệch giữa ô đang sáng và bar suy ra từ giờ phát ≤ **100 ms ở p95** (Chromium và Firefox, 60 fps). NFR-04: wrap-around của loop ≤ **100 ms**; click metronome lệch khỏi lưới beat ≤ **10 ms** (đo bằng `OfflineAudioContext`).
- **Không re-render React theo từng frame**: highlight, thời gian hiển thị, thanh tiến trình cập nhật bằng ref/DOM; kiểm bằng test đếm render.
- Bar: `beats_per_bar` từ `BeatGrid` (6/8 ⇒ 2), `bar k` bắt đầu tại `first_beat_offset_ms + k·beats_per_bar·60000/tempo_bpm`; nếu có `downbeats_ms` thì **ưu tiên** (bar = khoảng giữa hai downbeat liên tiếp).
- Tốc độ phát hợp lệ: 0.5, 0.75, 1 (FR-07), luôn giữ cao độ (`preservesPitch`); loop và metronome tính theo ms của bài, **không** theo giờ thực.
- Tier: Free = lưới, loop đơn ở 1×, metronome thủ công; Pro (`paid|teacher|student`) = thêm tốc độ, metronome đồng bộ bài, count-in. Prompt nâng cấp dùng lại phong cách `QuotaBanner`.
- Chuỗi mới vào từ điển i18n (`vi`/`en`) — TypeScript ép đủ khoá. Phím tắt và vùng bấm có thể truy cập bằng bàn phím; không bắt phím khi focus ở `input/textarea/select`.
- Lint/test sạch sau mỗi task: `rtk uv run pytest -q`, `rtk uv run ruff check . && rtk uv run ruff format --check .`, `rtk npm run lint` (0 cảnh báo), `rtk npm test -- --run`, `rtk npm run build`. Contract đổi ⇒ `rtk uv run python -m src.analysis_contract.export` rồi `rtk npm run gen:api`.
- Không commit `.wav/.mp3`; audio cho test sinh trong bộ nhớ.

## Review Focus (mỗi dòng có test ở task sở hữu)

1. **Tab ẩn / máy chậm:** rAF dừng khi tab ẩn ⇒ khi quay lại ô sáng phải nhảy đúng bar ngay; scheduler metronome **không** phát dồn các beat đã trôi qua (bỏ beat trễ, không bắn một loạt) (T6, T10, T11).
2. **Lưới không đều:** `downbeats_ms` (rubato), `first_beat_offset_ms ≠ 0`, 3/4 và 6/8, sự kiện bắc qua nhiều bar, hợp âm đầu bắt đầu sau bar 1, vùng *trước* beat đầu tiên (không bar nào sáng; bấm vào đó tua về 0) (T3, T10).
3. **Lỗi âm thanh giữa chừng:** ticket hết hạn (401/403 ở request Range) ⇒ xin ticket mới và phát tiếp đúng vị trí; chart nguồn embed không có audio (404) ; mạng rớt; video YouTube bị chặn/xoá (101/150) ⇒ thông báo + gợi ý tải file (AF1) (T2, T7, T8).
4. **Loop biên:** kéo ngược (B trước A), A==B, loop vượt cuối bài, đổi tốc độ khi đang lặp, tua ra ngoài loop, bấm Clear Loop khi đang ở giữa vòng (T5, T12).
5. **Range & quyền:** `Range` sai/nhiều đoạn/suffix/vượt cỡ file (206/200/416), `HEAD`, chart của người khác (404), ticket của chart A dùng cho chart B, access token hoặc refresh token bị dùng làm ticket và ngược lại, chart `youtube/soundcloud/direct` không có audio (T2).

---

## File Structure

```
DE/src/ingestion/domain/ports.py              # ChartRepository.list + ChartSummary
DE/src/ingestion/infrastructure/{memory,postgres}.py
DE/src/identity/application/media_tickets.py  # MediaTicketService
DE/src/api/routers/charts.py                  # list, ETag, audio-ticket, audio (Range)
DE/src/api/{schemas,deps,container,wiring}.py
web/src/grid/{grid.ts,effective.ts}           # pure
web/src/format/chord.ts                       # pure (Basic/Precise)
web/src/loop/loop.ts                          # pure
web/src/metronome/{beats.ts,tap.ts,click.ts,scheduler.ts}
web/src/transport/{types.ts,html.ts,youtube.ts,soundcloud.ts,select.ts,useTransport.ts}
web/src/practice/{SmartGrid.tsx,usePlayhead.ts,TransportBar.tsx,LoopControls.tsx,
                  MetronomePanel.tsx,UnlockModal.tsx,entitlements.ts,PracticePage.tsx}
web/src/pages/LibraryPage.tsx
web/src/spike/SyncSpike.tsx                   # dev-only route /__spike/sync
web/e2e/{sync.perf.spec.ts,practice.spec.ts}  contracts/fixtures/charts/*.effective.json
```

---

## Phần A — Backend

### Task 1: Danh sách chart và ETag (3.1)

**Files:**
- Modify: `DE/src/ingestion/domain/ports.py`, `infrastructure/{memory,postgres}.py`, `DE/src/api/routers/charts.py`, `DE/src/api/schemas.py`, `DE/src/analysis_contract/export.py`
- Test: `DE/tests/unit/api/test_charts_list.py`, `DE/tests/integration/test_ingestion_postgres.py` (mở rộng)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class ChartSummary: id: UUID; title: str; artist: str | None; source_kind: SourceKind; playback_kind: str; duration_ms: int; tempo_bpm: float; key: str; is_customized: bool; version: int; updated_at: datetime`.
  - `ChartRepository.list(owner_id: UUID, *, limit: int, after: tuple[datetime, UUID] | None) -> list[ChartSummary]` — mới nhất trước, hoà thì theo `id` giảm dần; chỉ chart của `owner_id`.
  - `GET /api/v1/charts?limit=20&cursor=…` ⇒ `ChartPage {items: list[ChartSummaryOut], next_cursor: str | None}`; `limit` 1–100; `cursor` là chuỗi mờ (base64 của `updated_at|id`).
  - `GET /api/v1/charts/{id}` thêm `ETag: W/"<version>-<updated_at ms>"`, `Cache-Control: private, no-cache`; `If-None-Match` khớp (một giá trị, danh sách, hoặc `*`) ⇒ `304` không body.

- [ ] **Step 1: Viết test fail** — thứ tự mới→cũ; phân trang 25 chart/`limit=10` đi hết 3 trang, không trùng, không sót, kể cả 5 chart cùng `updated_at`; `limit=0|101` ⇒ 422 `validation_error`; cursor rác ⇒ 422; chart người khác không bao giờ xuất hiện; danh sách rỗng ⇒ `items=[]`, `next_cursor=null`; mỗi item có `duration_ms`, `tempo_bpm`, `key` (`"C major"`) đúng với chart; `GET /charts/{id}` có ETag, gửi lại bằng `If-None-Match` ⇒ 304, đổi `version` ⇒ ETag khác ⇒ 200; `If-None-Match: *` ⇒ 304; chưa đăng nhập ⇒ 401. Integration (PG): cùng bộ phân trang trên `chordsense_test` và dùng được index `ix_charts_owner_updated` (không lỗi với 3 000 hàng).
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/api/test_charts_list.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces; Pg lấy `metadata->>'duration_ms'`, `metadata->'beat_grid'->>'tempo_bpm'`, `metadata->'key'` bằng biểu thức SQLAlchemy (không đọc cả `ai_events`); thêm `ChartPage`, `ChartSummaryOut` vào `_EXPORTED`, chạy export.
- [ ] **Step 4: Chạy lại** toàn bộ `rtk uv run pytest -q` + ruff — Expected: PASS, drift OpenAPI xanh.
- [ ] **Step 5: Commit** `feat(api): list charts with keyset pagination and ETag`

### Task 2: Ticket và endpoint audio có Range (3.1)

**Files:**
- Create: `DE/src/identity/application/media_tickets.py`
- Modify: `DE/src/api/routers/charts.py`, `DE/src/api/deps.py`, `DE/src/api/container.py`, `DE/src/api/wiring.py`, `DE/src/api/schemas.py`, `DE/tests/support/{identity,ingestion}.py`
- Test: `DE/tests/unit/identity/test_media_tickets.py`, `DE/tests/unit/api/test_chart_audio.py`

**Interfaces:**
- Produces:
  - `class MediaTicketService(secret: str, *, ttl: timedelta, clock)`: `issue(user_id: UUID, chart_id: UUID) -> tuple[str, datetime]`; `verify(token: str, chart_id: UUID) -> UUID` (user id) ném `InvalidToken | TokenExpired`. JWT HS256, claims `sub`, `chart`, `type="media"`, `exp`, `jti`; `TokenService.verify(…, "access")` **không** chấp nhận ticket và ngược lại.
  - `POST /api/v1/charts/{id}/audio-ticket` ⇒ `{url: "/api/v1/charts/{id}/audio?ticket=…", expires_at}`.
  - `GET|HEAD /api/v1/charts/{id}/audio`: xác thực bằng `Authorization: Bearer` (access) **hoặc** `?ticket=`; file ở `DATA_DIR/audio/<source.content_hash>.wav`; `Accept-Ranges: bytes`, `Content-Type: audio/wav`, `Cache-Control: private, max-age=3600`.
    - Không `Range` ⇒ 200 đủ file. `bytes=a-b`, `bytes=a-`, `bytes=-n` ⇒ 206 + `Content-Range`/`Content-Length` đúng. Nhiều đoạn ⇒ 200 đủ file (được phép theo RFC 9110). Cú pháp sai ⇒ bỏ qua `Range` (200). `start ≥ size` hoặc suffix `-0` ⇒ 416 + `Content-Range: bytes */<size>`.
    - Chart không tồn tại / của người khác / `playback.kind ≠ file` / file đã mất ⇒ 404 (`not_found`).

- [ ] **Step 1: Viết test fail** — `MediaTicketService`: vòng issue→verify; hết hạn (FakeClock +31 phút) ⇒ `TokenExpired`; ticket của chart A đưa cho chart B ⇒ `InvalidToken`; chữ ký sai/`alg=none` ⇒ `InvalidToken`; access token và refresh token đưa vào `verify` ⇒ `InvalidToken`, ticket đưa vào `TokenService.verify(…,"access")` ⇒ `InvalidToken`. Endpoint (WAV 10 000 byte sinh trong test): 200 byte-exact; `0-99` ⇒ 206, 100 byte đầu; `9900-` ⇒ 100 byte cuối; `-50` ⇒ 50 byte cuối; `0-99999` (quá cỡ) ⇒ 206 cắt về hết file; `10000-` và `-0` ⇒ 416; `bytes=abc` ⇒ 200; `bytes=0-9,20-29` ⇒ 200; `HEAD` ⇒ header đúng, body rỗng; dùng ticket đúng ⇒ 200/206, ticket hết hạn ⇒ 401 `token_expired`, ticket chart khác ⇒ 401; ticket **không** mở được `GET /charts/{id}` hay endpoint khác; Bearer của chủ ⇒ 200; user khác ⇒ 404 cho cả ticket-xin lẫn audio; chart `youtube` (tạo bằng repo) ⇒ 404 ở `audio-ticket` và `audio`; xoá file trên đĩa ⇒ 404; chart demo (kind `demo`) ⇒ 200.
- [ ] **Step 2: Chạy** `rtk uv run pytest tests/unit/identity/test_media_tickets.py tests/unit/api/test_chart_audio.py -q` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces; đọc file theo khối 64 KiB trong thread (`asyncio.to_thread`), không nạp cả file; TTL cấu hình qua `AuthSettings.media_ticket_ttl_minutes=30`; tên file chỉ lấy từ `content_hash` đã qua CHECK sha256 (không nhận đường dẫn từ client).
- [ ] **Step 4: Chạy lại** toàn bộ + ruff — Expected: PASS.
- [ ] **Step 5: Commit** `feat(api): serve chart audio with HTTP range through short-lived tickets`

---

## Phần B — Frontend: mô-đun thuần

### Task 3: Grid engine và effective events (3.3)

**Files:**
- Create: `web/src/grid/{grid.ts,effective.ts}`; `contracts/fixtures/charts/autumn_leaves_gm.effective.json` (sinh bởi `contracts/gen_fixtures.py`)
- Test: `web/src/grid/{grid.test.ts,effective.test.ts}`, `DE/tests/unit/analysis_contract/test_effective_fixture.py`

**Interfaces:**
- Produces:
  - `interface Bar { index: number; startMs: number; endMs: number; beatsMs: number[]; chords: BarChord[] }`; `interface BarChord { event: ChordEvent; fromMs: number; toMs: number; startsHere: boolean }`.
  - `interface Grid { bars: readonly Bar[]; durationMs: number; firstBarMs: number; barAtMs(ms: number): number /* -1 trước bar đầu */; barStartMs(index: number): number; barEndMs(index: number): number }`.
  - `buildGrid(meta: AnalysisMetadata, events: readonly ChordEvent[]): Grid` — hàm thuần, bar cuối cắt ở `duration_ms`; `barAtMs` dùng tìm nhị phân.
  - `applyOverrides(aiEvents, overrides): ChordEvent[]` — cùng ngữ nghĩa `DE/src/analysis_contract/overrides.py` (`delete` bỏ, `set/insert` thay và gắn `is_user_edited: true`, sắp theo `start_ms`, chồng lấn ⇒ ném `OverlapError`).

- [ ] **Step 1: Viết test fail (grid)** — 4/4@120 BPM offset 0 ⇒ bar `k` bắt đầu `k·2000`, mỗi bar 4 beat cách 500 ms; offset 350 ⇒ bar 0 bắt đầu 350, `barAtMs(349)=-1`, `barAtMs(350)=0`; 3/4@90 (`beats_per_bar 3`, bar 2000 ms); 6/8@60 với `beats_per_bar 2` ⇒ bar 2000 ms, 2 beat; tempo 96 (bar 2500 ms) và 133.33 (làm tròn ms nhất quán, bar sau không lệch tích luỹ: `bar k = round(offset + k·barMs)`); `downbeats_ms` `[0, 1900, 4100]` ghi đè nội suy (bar 0 dài 1900, bar 1 dài 2200, bar cuối kéo tới `duration_ms`); sự kiện 5 000–9 000 ms ở 4/4@120 nằm trong bar 2 và 3 và 4 với `startsHere` chỉ ở bar đầu; hợp âm đầu bắt đầu muộn (bar 0 không có chord); bài không có event ⇒ vẫn có bar theo `duration_ms`; bar cuối bị cắt `endMs = duration_ms`; `barAtMs(ms)` với `ms ≥ duration_ms` ⇒ **bar cuối** (playhead dừng ở cuối bài, không biến mất); 7 fixture trong `contracts/fixtures/results/*` đều dựng được lưới mà mọi event được phủ trọn bởi tổng các `BarChord`. Property (PRNG có seed, 500 mẫu): với tempo∈[40,240], offset∈[0,2000], `beats_per_bar`∈[1,12], mọi `ms`∈[firstBarMs, duration): `barStartMs(barAtMs(ms)) ≤ ms < barEndMs(barAtMs(ms))` và các bar liền kề không hở/chồng.
- [ ] **Step 2: Viết test fail (effective)** — `delete`, `set`, `insert`, override lên event không tồn tại, nhiều override, kết quả chồng lấn ⇒ `OverlapError`, baseline không bị sửa; **fixture chung**: `autumn_leaves_gm.edited.chart.json` áp bằng TS cho ra đúng `autumn_leaves_gm.effective.json` (do Python sinh bằng `apply_overrides`); `test_effective_fixture.py` kiểm file đó bằng chính `apply_overrides`.
- [ ] **Step 3: Chạy** `rtk npm test -- --run src/grid` và `rtk uv run pytest tests/unit/analysis_contract -q` — Expected: FAIL.
- [ ] **Step 4: Cài đặt**; mở rộng `contracts/gen_fixtures.py` để sinh `*.effective.json`.
- [ ] **Step 5: Chạy lại + lint + build** — Expected: PASS.
- [ ] **Step 6: Commit** `feat(web): add pure grid engine and effective-events overlay`

### Task 4: ChordFormatter Basic/Precise (3.8)

**Files:** Create `web/src/format/chord.ts`; Test `web/src/format/chord.test.ts`

**Interfaces:** `type ChordMode = 'basic' | 'precise'`; `formatChord(event: Pick<ChordEvent,'chord'|'symbol'>, mode: ChordMode): string`.

- [ ] **Step 1: Viết test fail** — bảng (precise = `symbol`): `Cmaj9→C`, `Am7→Am`, `G7→G`, `Bm7b5→Bdim`, `Caug→C`, `G9sus4→G`, `Dsus2→D`, `Em→Em`, `F#dim→F#dim`, `C/E→C/E`, `Am7/G→Am/G`, `G7/B→G/B`; mọi tổ hợp 12 root × 5 quality × 15 extension (± bass) không ném và `basic` luôn khớp `^[A-G]#?(m|dim)?(/[A-G]#?)?$`; `basic` idempotent theo nghĩa kết quả không đổi khi gọi lại trên chord đã đơn giản hoá; chord có `bass` khác root không bao giờ mất bass ở Basic.
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/format` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo D4 (không gọi API, không phụ thuộc React).
- [ ] **Step 4: Chạy lại + lint** — Expected: PASS. **Step 5: Commit** `feat(web): add basic/precise chord formatter`

### Task 5: Loop controller (3.6 — phần logic)

**Files:** Create `web/src/loop/loop.ts`; Test `web/src/loop/loop.test.ts`

**Interfaces:**
- `class LoopController`: `range: {startMs: number; endMs: number} | null`; `set(startMs, endMs): boolean` (từ chối khi `endMs ≤ startMs` hoặc âm); `setFromBars(grid: Grid, a: number, b: number): boolean` (nhận cả hai thứ tự, bar `b` được **gồm** trong loop, kẹp trong `[0, bars.length-1]`); `clear(): void`; `noteSeek(toMs: number): void`; `tick(positionMs: number): number | null` — trả vị trí cần tua về (`startMs`) khi vừa **đi qua** `endMs` từ trong loop; không bao giờ trả giá trị khi vừa tua ra ngoài.

- [ ] **Step 1: Viết test fail** — chạm B ⇒ trả A (`tick` liên tiếp 7 990, 8 010 với B=8 000); A==B và B<A bị từ chối, trạng thái cũ giữ nguyên; `setFromBars(grid, 16, 9)` = `setFromBars(grid, 9, 16)` = từ đầu bar 9 đến hết bar 16 (index 0-based theo grid: người dùng thấy "bar 9" = index 8); loop vượt cuối bài kẹp ở `durationMs`; tua ra ngoài (`noteSeek(20 000)` khi loop 8 000–16 000) rồi `tick(20 100)` ⇒ `null`; tua *vào* loop rồi chạm B ⇒ vẫn lặp; `clear()` giữa vòng ⇒ `tick` luôn `null`; đổi tốc độ không ảnh hưởng (controller không biết tốc độ; test rằng cùng chuỗi vị trí cho cùng kết quả); sau wrap `tick(startMs+5)` ⇒ `null` (không lặp liên tục).
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/loop` — Expected: FAIL. **Step 3: Cài đặt.** **Step 4: Chạy lại + lint.** **Step 5: Commit** `feat(web): add A/B loop controller`

### Task 6: Metronome — beat source, tap tempo, click, scheduler (3.7)

**Files:** Create `web/src/metronome/{beats.ts,tap.ts,click.ts,scheduler.ts}`; Test `web/src/metronome/{beats,tap,scheduler}.test.ts`, `web/e2e/metronome.perf.spec.ts` (T14)

**Interfaces:**
- `interface Beat { ms: number; accent: boolean }`; `interface BeatSource { beatsBetween(fromMs: number, toMs: number): Beat[] }`; `songBeats(grid: Grid): BeatSource` (accent = beat đầu bar; từ `Bar.beatsMs`); `manualBeats(bpm: number, beatsPerBar: number): BeatSource` (BPM kẹp 30–280, beat 0 tại 0 ms).
- `class TapTempo { tap(nowMs: number): number | null; reset(): void }` — BPM = 60 000 / trung vị 4–8 khoảng cách gần nhất; im lặng > 2 s ⇒ bắt đầu lại; kết quả kẹp 30–280; chưa đủ 2 lần bấm ⇒ `null`.
- `type Timbre = 'woodblock' | 'beep'`; `scheduleClick(ctx: BaseAudioContext, when: number, opts: {accent: boolean; timbre: Timbre; volume: number}): void` (dùng `OscillatorNode` + `GainNode` có envelope, `when` tính theo giờ `ctx`).
- `interface PlaybackClock { sample(): {songMs: number; ctxTime: number; rate: number} }`; `class MetronomeScheduler(ctx, source: BeatSource, clock: PlaybackClock, opts?: {lookaheadMs = 120; tickMs = 25; timbre; volume})` với `start()`, `stop()`, `reset()` (gọi khi seek / wrap loop / đổi tốc độ), `setSource(source)`.

- [ ] **Step 1: Viết test fail** — `songBeats`: 4/4@120 ⇒ beat mỗi 500 ms, accent mỗi 4; `beatsBetween` nửa mở `[from,to)` không trùng khi ghép cửa sổ liên tiếp; 6/8 ⇒ 2 beat/bar; `downbeats_ms` không đều ⇒ beat chia đều trong từng bar; `manualBeats(60,4)`; BPM 10 ⇒ 30, 999 ⇒ 280. `TapTempo`: 4 lần bấm cách 500 ms ⇒ 120; bấm lệch ±20 ms vẫn ≈120 (trung vị); nghỉ 3 s rồi bấm ⇒ bắt đầu lại; kết quả luôn trong 30–280. `MetronomeScheduler` với `AudioContext` giả ghi lại `when`: mỗi tick chỉ xếp beat trong `[songMs, songMs + lookahead·rate)`; `when = ctxTime + (beat.ms − songMs)/1000/rate` (0.75× ⇒ khoảng cách giữa hai click giãn đúng 1/0.75); **không xếp trùng** một beat qua nhiều tick; sau `reset()` xếp lại từ vị trí mới; **tab ẩn** (tick bị bỏ 3 s rồi `sample()` nhảy xa) ⇒ **không** xếp các beat đã trôi qua, chỉ các beat từ `songMs` trở đi; `stop()` huỷ hẹn giờ và không xếp thêm.
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/metronome` — Expected: FAIL. **Step 3: Cài đặt.** **Step 4: Chạy lại + lint.** **Step 5: Commit** `feat(web): add metronome beat sources, tap tempo and look-ahead scheduler`

---

## Phần C — Transport đa nguồn

### Task 7: Transport interface và HtmlAudioTransport (3.4)

**Files:** Create `web/src/transport/{types.ts,html.ts}`, `web/src/test/fakeMedia.ts`; Test `web/src/transport/transport.contract.test.ts`, `web/src/transport/html.test.ts`

**Interfaces:**
```ts
type TransportState = 'idle'|'loading'|'ready'|'playing'|'paused'|'ended'|'error'
interface TransportError { code: 'autoplay_blocked'|'unavailable'|'embed_blocked'|'network'; message: string }
interface Transport {
  readonly kind: 'html'|'youtube'|'soundcloud'
  readonly capabilities: { rate: boolean }
  readonly state: TransportState
  load(): Promise<void>; play(): Promise<void>; pause(): void
  seek(ms: number): void; setRate(rate: 0.5|0.75|1): void
  currentTimeMs(): number; durationMs(): number | null
  on(event: 'state'|'error', fn: (e?: TransportError) => void): () => void
  destroy(): void
}
class HtmlAudioTransport implements Transport { constructor(audio: HTMLAudioElement, source: () => Promise<string>) }
```
`source()` trả URL (ticket cho `kind=file`, `audio_url` cho `direct`). `runTransportContract(name, factory)` là bộ test dùng chung cho cả ba adapter.

- [ ] **Step 1: Viết test fail (contract, tham số hoá theo adapter)** — `load` rồi `state=ready`; `play` ⇒ `playing`, `pause` ⇒ `paused`; `seek(5000)` ⇒ `currentTimeMs()` ≈ 5000; seek âm ⇒ 0, quá cuối ⇒ `durationMs`; `setRate(0.75)` được áp dụng (nếu `capabilities.rate`), giá trị khác {0.5, 0.75, 1} bị từ chối; `on('state')` nhận đủ chuyển trạng thái và hàm huỷ đăng ký hoạt động; `destroy()` dừng phát, gỡ listener, các lời gọi sau đó không ném; lỗi nền ⇒ `on('error')` với `code` hợp lệ và `state='error'`. **HtmlAudio riêng:** `preservesPitch === true` (cả `mozPreservesPitch`/`webkitPreservesPitch` nếu có); `play()` bị `NotAllowedError` ⇒ `error{autoplay_blocked}`, state về `paused`; lỗi media khi đang phát (ticket hết hạn) ⇒ gọi lại `source()` **một lần**, gán URL mới, tua về vị trí cũ và phát tiếp, không lặp vô hạn nếu lần 2 vẫn lỗi (`error{network}`); `ended` ⇒ state `ended`, `play()` sau đó phát lại từ 0; seek khi chưa `ready` được ghi nhớ và áp dụng khi sẵn sàng; `currentTimeMs` luôn là số nguyên.
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/transport` — Expected: FAIL. **Step 3: Cài đặt** (`FakeAudio` trong `fakeMedia.ts`: `EventTarget` có `play/pause/currentTime/duration/playbackRate/preservesPitch/src/error`). **Step 4: Chạy lại + lint + build.** **Step 5: Commit** `feat(web): add transport interface and HTML audio adapter`

### Task 8: YouTubeTransport, SoundCloudTransport, chọn nguồn (3.4)

**Files:** Create `web/src/transport/{youtube.ts,soundcloud.ts,select.ts,useTransport.ts}`; Test `web/src/transport/{youtube,soundcloud,select}.test.ts`, mở rộng `transport.contract.test.ts`

**Interfaces:**
- `new YouTubeTransport(host: HTMLElement, videoId: string, loadApi: () => Promise<YTNamespace>)`; `new SoundCloudTransport(iframe: HTMLIFrameElement, loadApi: () => Promise<SCNamespace>)` — API ngoài được tiêm để test bằng fake; mặc định `loadApi` nạp `https://www.youtube.com/iframe_api` / `https://w.soundcloud.com/player/api.js` một lần.
- `createTransport(playback: PlaybackSource, deps): Transport` — `file` ⇒ Html (+ ticket qua `POST /charts/{id}/audio-ticket`), `direct` ⇒ Html với `audio_url`, `youtube`/`soundcloud` ⇒ adapter tương ứng; `useTransport(chart)` hook vòng đời (tạo/huỷ).
- `SoundCloudTransport.capabilities.rate = false`; `currentTimeMs()` nội suy giữa các sự kiện `PLAY_PROGRESS` bằng `performance.now()` để mượt ở 60 fps.

- [ ] **Step 1: Viết test fail** — chạy lại **bộ contract** cho YouTube và SoundCloud (fake API); YouTube: lỗi `2|5|100|101|150` ⇒ `error{embed_blocked}` (101/150) hoặc `unavailable` (100/2/5) kèm thông điệp gợi ý tải file lên (AF1); `onStateChange` 1/2/0/3 ánh xạ `playing/paused/ended/loading`; sau `buffering` (3) rồi `playing` thì `currentTimeMs` lấy lại từ `getCurrentTime()` (không dùng giá trị nội suy cũ); `setPlaybackRate(0.75)` chỉ gọi khi `getAvailablePlaybackRates()` chứa 0.75, ngược lại bỏ qua và `capabilities.rate=false`. SoundCloud: nội suy đúng giữa hai `PLAY_PROGRESS` cách 250 ms; `seek` gọi `seekTo` rồi cập nhật mốc nội suy; `setRate` là no-op có ghi nhận. `createTransport`: đúng adapter cho từng `PlaybackSource.kind`; `file` xin ticket đúng chart; ticket 404 ⇒ `error{unavailable}`; script API không nạp được ⇒ `error{network}` và không treo `load()`.
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/transport` — Expected: FAIL. **Step 3: Cài đặt.** **Step 4: Chạy lại + lint + build.** **Step 5: Commit** `feat(web): add YouTube and SoundCloud transports with source selection`

---

## Phần D — Spike quyết định render (3.2)

### Task 9: SmartGrid prototype và đo NFR-01 (3.2, 3.5)

**Files:** Create `web/src/practice/{SmartGrid.tsx,usePlayhead.ts}`, `web/src/spike/SyncSpike.tsx` (route dev-only `/__spike/sync`), `web/e2e/sync.perf.spec.ts`; Modify `web/playwright.config.ts` (project `firefox` chỉ cho `*.perf.spec.ts`), `web/src/routes.tsx`

**Interfaces:**
- `SmartGrid({ grid, mode, current?, loop?, onSeek(ms), onSelect(fromBar, toBar), gridRef })` — mỗi bar là một `<button>` memo hoá (`data-bar`, `aria-label` "Bar 9, Dm7 G7"); chord hiển thị qua `formatChord(…, mode)`. Highlight **không** là prop.
- `usePlayhead(transport, grid, root: RefObject<HTMLElement>, opts?)` — vòng `requestAnimationFrame` đọc `transport.currentTimeMs()`, tính `grid.barAtMs`, chỉ khi bar đổi mới gỡ/gán `data-current="true"`; không `setState`. Tự hạ tần suất khi khung hình trung bình > 34 ms (30 → 15 fps, 7a1); khi `visibilitychange` ⇒ `visible` cập nhật ngay một lần; trả `{ fps }` qua ref.
- Trang spike: dựng click-track 4 phút (mỗi bar một tiếng "tick" cao hơn ở beat 1) **trong bộ nhớ** bằng `OfflineAudioContext` → WAV Blob URL, lưới tổng hợp 4/4@120 (120 bar), phát qua `HtmlAudioTransport`, mỗi frame ghi `{perfNow, audioMs, shownBar}` và tính `window.__syncStats = { samples, p50, p95, max, meanFps }` với `offset = 0` nếu `shownBar` đúng bar chứa `audioMs`, ngược lại khoảng cách (ms) từ `audioMs` tới bar đang sáng.

- [ ] **Step 1: Viết test fail** — unit (jsdom, `FakeAudio`): `usePlayhead` đặt `data-current` đúng bar; không đụng DOM khi bar không đổi (spy `setAttribute` = 0 lần giữa hai frame cùng bar); **0 lần re-render** của `SmartGrid` qua 120 frame mô phỏng (đếm render); hạ xuống 30 fps khi `performance.now` giả nhảy 40 ms/frame; `visibilitychange` ⇒ cập nhật tức thì; vùng trước beat đầu ⇒ không bar nào có `data-current`; `SmartGrid`: bấm bar 9 ⇒ `onSeek(barStartMs(8))`; bấm vùng trước beat đầu không có (không có nút); 400 bar render < 100 ms trong jsdom; không có `<button>` nào bị re-render khi `mode` không đổi và chỉ `current` đổi. Perf e2e `sync.perf.spec.ts` (Chromium **và** Firefox, 3 lần × 30 s mặc định, `SYNC_FULL=1` ⇒ 4 phút): `p95 ≤ 100`, `meanFps ≥ 50`, in số đo ra output.
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/practice` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** SmartGrid (DOM + `data-current`, CSS transition ≤ 80 ms) và `usePlayhead`; cài Firefox (`rtk npx playwright install firefox`) và thêm project.
- [ ] **Step 4: Chạy** `rtk npm test -- --run` + `rtk npm run test:e2e -- sync.perf` — Expected: PASS. **Cổng quyết định:** nếu p95 > 100 ms ở trình duyệt nào, **dừng** và ghi quyết định (canvas hoặc `will-change` + compositor-only), không đi tiếp T10–T13 trước khi đạt.
- [ ] **Step 5: Ghi số đo** (p50/p95/max/fps, 2 trình duyệt, ngày chạy, máy) vào `.agents/memory/memory.md`; ghi rõ embed YouTube/SoundCloud **chưa đo thật** (D6). **Commit** `feat(web): add DOM smart grid with rAF playhead and measure NFR-01`

---

## Phần E — Practice Workspace

### Task 10: Thư viện và bộ nạp chart (3.1 FE)

**Files:** Create `web/src/pages/LibraryPage.tsx`; Modify `web/src/routes.tsx`, `web/src/api/{analyze,types}.ts`, `web/src/i18n/messages.ts`, `web/src/test/server.ts`; Test `web/src/pages/LibraryPage.test.tsx`, `web/src/api/charts.test.ts`

**Interfaces:** `listCharts(client, {cursor?, limit?}): Promise<ChartPage>`; `getChart(client, id)` gửi `If-None-Match` từ cache và xử lý `304` (TanStack `queryFn` giữ bản cũ); `getAudioTicket(client, chartId): Promise<{url: string; expiresAt: Date}>`. `LibraryPage` dùng `useInfiniteQuery`.

- [ ] **Step 1: Viết test fail** — danh sách thẻ (tên, nghệ sĩ, nhãn nguồn Upload/Link/Demo, thời lượng `m:ss`, BPM, giọng); skeleton khi tải; "Tải thêm" theo `next_cursor` và biến mất khi hết; trạng thái rỗng có liên kết tới Workspace; lỗi ⇒ toast `toastFor`; thẻ là liên kết tới `/charts/:id`; `getChart` nhận `304` giữ nguyên dữ liệu và gửi `If-None-Match` ở lần sau; `getAudioTicket` trả URL tuyệt đối theo `baseUrl` của client; i18n vi/en.
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/pages src/api` — Expected: FAIL. **Step 3: Cài đặt** (thay `SectionPage library`). **Step 4: Chạy lại + lint + build.** **Step 5: Commit** `feat(web): add library page with paginated charts`

### Task 11: PracticePage — nối chart, grid, transport (3.3–3.5)

**Files:** Create `web/src/practice/PracticePage.tsx`, `UnlockModal.tsx`; Modify `web/src/pages/ChartPage.tsx` (xoá), `web/src/routes.tsx`; Test `web/src/practice/PracticePage.test.tsx`, thay `web/src/pages/ChartPage.test.tsx`

**Interfaces:** `PracticePage` tải chart ⇒ `applyOverrides` ⇒ `buildGrid` (memo theo `chart.version`) ⇒ `useTransport(chart)` ⇒ `SmartGrid` + `usePlayhead`; khung media (video/iframe cho embed, ẩn với `<audio>`); `UnlockModal({ onUnlock })` hiện khi `play()` trả `autoplay_blocked` hoặc `AudioContext.state === 'suspended'` (2a).

- [ ] **Step 1: Viết test fail** — mở chart ⇒ tiêu đề, giọng, BPM và lưới đúng số bar; chart có `overrides` hiển thị hợp âm đã sửa (dùng fixture `autumn_leaves_gm.edited`); chart 404 ⇒ thông báo + liên kết Thư viện; `autoplay_blocked` ⇒ modal; bấm "Bật âm thanh" ⇒ `resume()` + phát tiếp; chart `youtube` hiển thị khung video và **không** xin ticket; lỗi `embed_blocked` ⇒ banner "video không khả dụng" + nút "Tải tệp lên thay thế" (AF1) dẫn tới Workspace; chuyển trang khi đang phát ⇒ `transport.destroy()` được gọi, rAF dừng (không còn callback sau unmount); đổi chart khác không rò rỉ lưới cũ.
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/practice src/pages` — Expected: FAIL. **Step 3: Cài đặt.** **Step 4: Chạy lại + lint + build.** **Step 5: Commit** `feat(web): turn the chart page into the practice workspace`

### Task 12: Thanh điều khiển, loop, click-to-seek, phím tắt (3.4, 3.6)

**Files:** Create `web/src/practice/{TransportBar.tsx,LoopControls.tsx}`; Modify `SmartGrid.tsx`, `PracticePage.tsx`, `i18n/messages.ts`; Test `web/src/practice/{TransportBar,LoopControls,shortcuts}.test.tsx`

**Interfaces:** `TransportBar` — Play/Pause, thời gian `m:ss / m:ss` cập nhật 4 Hz bằng `textContent` (không state), chọn tốc độ 0.5/0.75/1, cờ `usePlayhead`; phím: Space ⇒ phát/dừng, ←/→ ⇒ ∓1 bar, Shift+←/→ ⇒ ∓5 s (D5), bị bỏ qua khi `event.target` là `input|textarea|select|[contenteditable]`. `LoopControls` — chọn loop bằng kéo trên lưới (con trỏ) **hoặc** Shift+bấm bar (bàn phím/chạm), vẽ dải loop, nút "Xoá loop"; vòng `tick` chạy trong cùng rAF và gọi `transport.seek(start)` rồi `metronome.reset()`.

- [ ] **Step 1: Viết test fail** — Space bật/tắt (không khi focus trong ô BPM; không toggle đôi khi focus ở nút); ←/→ nhảy đúng bar theo `grid` và kẹp ở 0/cuối bài; kéo từ bar 9 tới 16 (cả kéo ngược) ⇒ `LoopController.range` = bar 9..16; Shift+bấm hai bar tương đương; loop chạy: mô phỏng `currentTimeMs` đi qua B ⇒ `seek(A)` đúng một lần và độ trễ từ lúc vượt B tới lúc gọi `seek` ≤ 1 frame (giờ giả); đổi tốc độ khi đang lặp giữ nguyên loop; tua ra ngoài loop không kéo về; "Xoá loop" ⇒ phát tiếp bình thường (5a); click bar 9 ⇒ `seek(barStartMs(8))` và highlight cập nhật ngay trong frame kế; thời gian hiển thị không gây re-render; chọn tốc độ gọi `setRate` và hiển thị `0.75×`; nguồn SoundCloud ⇒ khối tốc độ bị vô hiệu kèm giải thích (D6).
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/practice` — Expected: FAIL. **Step 3: Cài đặt.** **Step 4: Chạy lại + lint + build.** **Step 5: Commit** `feat(web): add transport bar, click-to-seek, A/B loop and shortcuts`

### Task 13: Metronome UI, Basic/Precise, count-in (3.7, 3.8)

**Files:** Create `web/src/practice/MetronomePanel.tsx`; Modify `PracticePage.tsx`, `SmartGrid.tsx`, `i18n/messages.ts`; Test `web/src/practice/MetronomePanel.test.tsx`

**Interfaces:** `MetronomePanel` — bật/tắt, chế độ **Thủ công** (BPM 30–280 gõ/kéo, Tap Tempo, số beat/bar) và **Đồng bộ bài** (dùng `songBeats(grid)`, theo tốc độ và loop), chọn âm sắc (woodblock/beep), âm lượng độc lập; nút **Basic/Precise** cạnh thanh công cụ (đổi `mode` của `SmartGrid` tức thì, lưu `localStorage['chordsense.chordMode']`); count-in = 1 bar tick trước khi `play()` thật (chỉ Pro). `AudioContext` tạo ở lần bấm đầu tiên (đáp ứng autoplay policy) và dùng chung với `UnlockModal`.

- [ ] **Step 1: Viết test fail** — bật metronome tạo `MetronomeScheduler` với `PlaybackClock` ghép `transport.currentTimeMs()` + `ctx.currentTime` + rate hiện tại; Đồng bộ bài dùng beat của lưới (bar đầu accent); đổi tốc độ / loop wrap / seek ⇒ `reset()` được gọi; tắt ⇒ `stop()`; BPM nhập 10 ⇒ 30, 999 ⇒ 280, chữ ⇒ giữ giá trị cũ; Tap Tempo cập nhật BPM; Basic/Precise đổi nhãn `G9sus4`→`G` ngay, **không có request mạng** (MSW "no request") và nhớ lựa chọn sau tải lại; chuyển chế độ không re-render các bar không đổi nhãn; count-in phát `beats_per_bar` tick rồi mới gọi `transport.play()`; i18n vi/en.
- [ ] **Step 2: Chạy** `rtk npm test -- --run src/practice` — Expected: FAIL. **Step 3: Cài đặt.** **Step 4: Chạy lại + lint + build.** **Step 5: Commit** `feat(web): add metronome panel, chord mode toggle and count-in`

### Task 14: Phân quyền Free/Pro và prompt nâng cấp (3.9 — FE)

**Files:** Create `web/src/practice/entitlements.ts`, `UpgradePrompt.tsx`; Modify `TransportBar.tsx`, `LoopControls.tsx`, `MetronomePanel.tsx`; Test `web/src/practice/entitlements.test.ts`, mở rộng các test panel

**Interfaces:** `entitlementsFor(role: Role): { speed: boolean; songSyncMetronome: boolean; countIn: boolean; loopSpeed: boolean }` (Pro = `paid|teacher|student`); `UpgradePrompt({ feature })` tái dùng kiểu `QuotaBanner` + liên kết `#upgrade`.

- [ ] **Step 1: Viết test fail** — bảng role × tính năng (8 role); Free: chọn 0.75× bị chặn, hiện prompt, tốc độ giữ 1×; metronome Đồng bộ bài và count-in hiện khoá + prompt, Thủ công vẫn dùng được (4b1); loop đơn vẫn chạy ở 1×; Pro: mở hết; **không** có request ghi nào (loop chưa lưu cloud — D3); đổi role ở phiên mới áp dụng đúng.
- [ ] **Step 2: Chạy** — Expected: FAIL. **Step 3: Cài đặt.** **Step 4: Chạy lại + lint + build.** **Step 5: Commit** `feat(web): gate speed, song-sync metronome and count-in by tier`

---

## Phần F — Nghiệm thu

### Task 15: E2E, NFR-04, tài liệu, DoD

**Files:** Create `web/e2e/{practice.spec.ts,metronome.perf.spec.ts}`; Modify `.agents/context/{context.md,implementation-plan.md}`, `.agents/memory/memory.md`

- [ ] **Step 1: Viết spec fail** (`E2E_BACKEND=1`; chạy Chromium, riêng `*.perf` thêm Firefox):
  - `practice.spec.ts`: đăng ký → bài demo `ii_v_i_c` → mở chart → **Play** (qua modal unlock nếu có) → ô nhịp sáng thay đổi theo thời gian; bấm ô **bar 9** của bài `blues_f` ⇒ `audio.currentTime` ≈ thời điểm bar 9 (±100 ms); kéo bar 9–16 ⇒ phát qua B tự quay về A (đo thời gian wrap ≤ 100 ms trong trang bằng mốc `performance.now`); chọn 0.75× ⇒ `audio.playbackRate === 0.75` và `audio.preservesPitch === true`; Basic/Precise đổi nhãn; Space/←/→; user Free thấy prompt nâng cấp ở 0.75× (đăng ký mới là Free); chart link YouTube hiển thị khung video (API ngoài bị chặn bằng `page.route` ⇒ banner "không khả dụng" + nút tải file lên); ticket hết hạn (route trả 401 một lần) ⇒ phát tiếp đúng vị trí.
  - `metronome.perf.spec.ts`: `page.evaluate` nạp `/src/metronome/click.ts` + `beats.ts`, render 32 beat bằng `OfflineAudioContext` với lưới thật của `blues_f`, phát hiện onset (ngưỡng biên độ), so với `beat.ms`: lệch **tối đa ≤ 10 ms** ở Chromium và Firefox, accent cao hơn beat thường.
- [ ] **Step 2: Chạy** `E2E_BACKEND=1 rtk npm run test:e2e` — Expected: FAIL tới khi mọi task trước xong, sau đó PASS (chạy 3 lần liên tiếp để loại spec không ổn định).
- [ ] **Step 3: Nghiệm thu DoD**: (a) `rtk uv run pytest -q` + ruff; (b) `rtk npm run lint && rtk npm test -- --run && rtk npm run build`; (c) p95 NFR-01 ≤ 100 ms trên cả hai trình duyệt, **số đo ghi trong `memory.md`**; (d) NFR-04: wrap ≤ 100 ms, click ≤ 10 ms; (e) không re-render theo frame (test đếm render T9 + một lần xem Chrome performance trace/React Profiler và ghi kết quả); (f) bar 9–16 lặp được, 0.75× không méo cao độ (cờ `preservesPitch`; nghe thử bằng tai một lần — ghi vào `memory.md`), metronome trùng beat; (g) chạy tay một link YouTube và một SoundCloud thật, ghi độ lệch quan sát và cờ `calibration offset` cần đặt (D6).
- [ ] **Step 4: Cập nhật** `context.md` (M3 ✅, rủi ro: embed chưa đo thật, loop chưa lưu cloud), `memory.md` (ADR D1–D8 + số đo), đánh dấu M3 trong `implementation-plan.md`.
- [ ] **Step 5: Commit** `test(e2e): cover the practice workspace and measure NFR-01/04; docs: close milestone 3`

---

## Thứ tự & phụ thuộc

`T1 ∥ T2` (BE độc lập) · `T3 → T4 → T5 → T6` (FE thuần, có thể song song với BE) · `T7 → T8` · **`T9` cần T2 (không bắt buộc, spike dùng audio trong bộ nhớ), T3, T4, T7 — là cổng quyết định** · `T10` cần T1/T2 · `T11` cần T3, T7, T8, T9, T10 · `T12` cần T5, T11 · `T13` cần T4, T6, T11 · `T14` cần T12, T13 · `T15` cuối.

Mốc có thể demo sớm: **sau T9** thấy lưới sáng theo click-track ngay trong trình duyệt và có số đo NFR-01; **sau T11** mở được chart thật và nghe; **sau T12** tập được với loop.

## Self-review (đã chạy)

- **Phủ spec 3.1–3.9:** 3.1 ⇒ T1, T2, T10; 3.2 ⇒ T9; 3.3 ⇒ T3; 3.4 ⇒ T7, T8, T11, T12; 3.5 ⇒ T9; 3.6 ⇒ T5, T12; 3.7 ⇒ T6, T13; 3.8 ⇒ T4, T13; 3.9 ⇒ T14 (FE; phần BE hoãn — D3). "TDD cần viết trước" của plan gốc: Grid (T3), `msToBar/barToMs` property (T3), Loop (T5), Metronome offline (T6 logic + T15 `OfflineAudioContext`), Formatter (T4), Range 206 (T2), Playwright click-seek ±100 ms (T15), transport contract 3 adapter (T7/T8), YouTube lỗi 101/150 (T8), spike từng loại nguồn (T9 + D6).
- **Khác plan gốc có chủ đích:** D1 (ticket), D2 (không dùng `MediaElementAudioSourceNode`), D3, D4, D5, D8; spike đặt *sau* khi có grid engine và transport cơ bản (cần chúng để đo) nhưng *trước* mọi UI còn lại.
- **Nhất quán tên:** `Grid`/`Bar`/`BarChord`, `buildGrid`, `applyOverrides`, `formatChord`, `LoopController` (`set`, `setFromBars`, `noteSeek`, `tick`), `Beat`/`BeatSource`/`songBeats`/`manualBeats`, `PlaybackClock`, `MetronomeScheduler.reset`, `Transport` (`capabilities.rate`), `createTransport`, `usePlayhead`, `entitlementsFor` dùng thống nhất.
- **Rủi ro biết trước:** Firefox headless không chắc phát được audio ổn định (nếu không, ghi nhận và đo bằng chế độ có đầu); YouTube/SoundCloud chỉ kiểm bằng mock + thử tay; time-stretch của `preservesPitch` khác nhau giữa trình duyệt; độ trễ đầu ra `<audio>` so với `AudioContext` có thể cần calibration offset; ticket qua query string lọt vào log truy cập — TTL ngắn và gắn chart + user giảm rủi ro, cần lưu ý khi triển khai thật.

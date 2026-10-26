# memory.md — Quyết định kiến trúc, lỗi đã fix, quy tắc dự án

> Chỉ ghi những gì **không suy ra được từ code/git log**. Append-only, mới nhất ở cuối.
> Mẫu: `YYYY-MM-DD · [ADR|FIX|RULE] · Tiêu đề` → **Why** / **How to apply**.
> Đọc `../context/context.md` trước để biết trạng thái hiện tại.

## RULE — Bất biến của dự án

- **RULE · RTK**: mọi lệnh shell có tiền tố `rtk` (`rtk uv run pytest -v`, `rtk npm run build`, `rtk git …`).
- **RULE · Phân vai** (`AGENTS.md`): Claude Code = TDD + thuật toán + commit nguyên tử; Antigravity = kiến trúc, review diff, cập nhật tài liệu/LaTeX/diagram. Claude Code không sửa LaTeX/diagram.
- **RULE · Git**: `<type>(<scope>): <desc>`; không commit `.env`, khoá, `.wav/.mp3`.
- **RULE · Event-based timeline**: chord lưu theo mili-giây tuyệt đối, không lưu Grid/Bar (đổi tempo sẽ vỡ).
- **RULE · AI baseline bất biến**: chỉnh sửa người dùng đi vào lớp `overrides`; luôn có "Reset to AI".
- **RULE · Real-time ở client**: Basic/Precise, A/B loop, metronome, Harmonic Engine, chỉnh sửa hợp âm không gọi mạng.
- **RULE · Policy ngoài code**: quota/giá đọc từ `PolicyProvider`, không hard-code (FR-18/28).

## ADR

- **2026-10-22 · ADR · Pydantic là nguồn sự thật của contract**
  Why: một nơi định nghĩa, sinh OpenAPI → TS types, fixtures dùng chung, tránh lệch FE/BE/AI.
  How: sửa model trong `DE/src/analysis_contract/`, chạy lại script sinh `contracts/openapi.json`; drift check trong test sẽ báo nếu quên.
- **2026-10-22 · ADR · AI contract không chứa Roman numeral**
  Why: FR-04 quy định Harmonic Engine chạy ở client; tránh hai bản cài đặt lệch nhau.
  How: AI trả key + chord events; FE tính Roman/Scale; `harmony_golden.json` là bộ test chung.
- **2026-10-22 · ADR · ORM = SQLAlchemy 2 async + Alembic** (không Prisma như `ChordSense.MD`) — theo `CLAUDE.md`.
- **2026-10-22 · ADR · Job runner sau port** (`JobRunnerPort`): `InProcess` cho dev/test, Celery+Redis cho môi trường thật.
- **2026-10-22 · ADR · Pitch-preserving speed dùng `<audio>.preservesPitch`**, không dùng `AudioBufferSource.playbackRate` (đổi cao độ).

- **2026-10-22 · ADR · Phát lại audio theo nguồn (`PlaybackSource`)**
  Why: giống Chordify — link YouTube/SoundCloud phát bằng player nhúng chính thức; tránh lưu/redistribute audio có bản quyền. Upload/demo dùng `<audio>`.
  How: `Transport` interface + 3 adapter (HtmlAudio / YouTube IFrame / SoundCloud Widget). Server xoá audio tạm của URL sau phân tích. Đo NFR-01 riêng từng nguồn.
- **2026-10-22 · ADR · `bass` trong ChordLabelDTO là optional**: AI (DE) chưa nhận diện slash chord ⇒ `null`; chỉ có giá trị khi user sửa. Chưa chốt với team AI.
- **2026-10-22 · ADR · Dev DB = PostgreSQL 18 native (DB `chordsense_app`)**, prod/compose = 16 ⇒ SQL phải tương thích cả hai. Compose DE map 5432 → trùng cổng với PG18.

- **2026-10-22 · ADR · Contract v1.0 (task 1.1)**: `DE/src/analysis_contract/` → `contracts/openapi.json` → `web/src/api/generated/schema.d.ts`. Thêm so với bản phác thảo: `BeatGrid.beats_per_bar` (6/8 theo dõi bằng nốt chấm = 2 beat/ô; `tempo_bpm` đếm theo beat của tracker), `ChordEvent.symbol` bị validator ép khớp `chord` có cấu trúc, `Chart.ai_events` bất biến + `overrides` (set/delete/insert, tối đa 1 override/event), `apply_overrides()` là ngữ nghĩa chuẩn của "effective timeline" (bản TS ở M4 phải qua cùng fixtures).
  How: sửa model → `rtk uv run python -m src.analysis_contract.export` → `npm run gen:api` (test drift sẽ đỏ nếu quên bước 1). Fixtures sinh bằng `rtk uv run python ../contracts/gen_fixtures.py` (chạy từ `DE/`).

- **2026-10-22 · ADR · Persistence (task 1.2)**: SQLAlchemy **Core** tables ở `DE/src/persistence/tables.py` (schema `app`), migration Alembic **viết tay** ở `DE/migrations/versions/` (thư mục tên `migrations`, không phải `alembic/` để khỏi che package `alembic`). Test integration so `compare_metadata` giữa DB đã migrate và `tables.py` ⇒ hai bên không lệch. Revision đã merge/áp dụng thì **đóng băng**, thay đổi mới = revision mới.
  Điểm đáng nhớ: `analyses` (key = content_hash) là kho kết quả AI dùng chung để cache không tốn quota; `jobs` có CHECK `(state='failed') = (error IS NOT NULL)`; `playlist_items` unique `(playlist_id, order_index)` **DEFERRABLE INITIALLY DEFERRED** để đổi chỗ trong 1 transaction; `usage_counters` PK `(user_id, period_date)` cho upsert nguyên tử; email unique theo `lower(email)`; role lưu 7 giá trị (Guest không lưu).
  How: `rtk uv run alembic upgrade head` (từ `DE/`, đọc `DE/.env`); integration test dùng `TEST_DATABASE_URL` (tên DB phải kết thúc `_test`, fixture từ chối DB khác).

- **2026-10-22 · ADR · Identity & auth (task 1.3)**: bounded context `DE/src/identity/{domain,application,infrastructure}` + lớp HTTP `DE/src/api/`. JWT HS256 (chỉ `sub/type/jti/iat/exp`; **role và is_active luôn đọc từ DB mỗi request** ⇒ đổi role/khoá tài khoản có hiệu lực ngay). Thuật toán bị ghim (`alg:none`, HS384/512 bị từ chối); hết hạn so với `clock` được tiêm (test xác định). Refresh **rotate + revoke**, tái dùng token cũ ⇒ `token_revoked`. Mật khẩu bcrypt cost ≥ 12 (settings ép; test dùng 4), > 72 byte bị từ chối thay vì cắt ngầm; login email lạ vẫn tốn 1 lần verify (chống dò email). Quota: `QuotaService` đọc `PolicyProvider`, reset 00:00 UTC, `try_consume` là **một câu lệnh SQL nguyên tử** (đã test 20 request đồng thời, hạn mức 5 ⇒ đúng 5). `refund()` không idempotent theo receipt — M2 phải chặn bằng cờ `jobs.quota_charged`.
  FE: access token chỉ ở **bộ nhớ**, refresh token ở `localStorage` (`createApiClient`); 401 `token_expired|invalid_token|not_authenticated` ⇒ **một** lần refresh dùng chung giữa các request đồng thời rồi retry đúng 1 lần; mất mạng KHÔNG đăng xuất, bị server từ chối (4xx) thì có. Thêm `POST /auth/logout` (204, im lặng với token xấu) để thu hồi refresh token.
  Chưa làm (cố ý): khoá tài khoản sau 5 lần sai (FR-32, thuộc gateway protection), tier Education pooled quota.
- **2026-10-22 · RULE · Test auth dùng `tests/support/{identity,api}.py`** (container in-memory + app FastAPI nhỏ có route thăm dò cho ma trận role×endpoint, NFR-10). Role quyết định ở `Role`/`PRO_ROLES` (`models.py`); `Guest` không lưu DB.

- **2026-10-26 · ADR · App factory (task 1.4)**: `DE/src/api/app.py::create_app(container=None, settings=None)`; không truyền container ⇒ lifespan tạo engine từ `database_url()` + `build_container` (`src/api/wiring.py`, PG repos; `InMemoryRevocationStore` chỉ cho dev). Chạy: `rtk uv run uvicorn src.api.app:create_app --factory --port 8000` (Vite proxy `/api` → 8000).
  Lỗi luôn dạng `{code,message}`; 422 thêm `detail:[{field,message}]`; 404/405 → `not_found`/`method_not_allowed`; exception lạ → 500 `internal_error` (không lộ chi tiết, traceback chỉ vào log). Middleware: `X-Request-ID` (chỉ chấp nhận `[A-Za-z0-9._-]{1,64}`, ngược lại sinh mới) + 1 dòng log structlog `request` mỗi request. CORS: `API_CORS_ORIGINS` (JSON list, mặc định 5173/4173), cấm `*`. `/healthz` chỉ báo process sống, chưa ping DB.
  FE: `ApiError.fieldErrors/requestId`; `toastFor(error)` (`web/src/api/errors.ts`) là nguồn duy nhất map mã lỗi → `{kind,message}`; `auth/messages.errorMessage` dùng lại. Component toast thật làm ở 1.5.

- **2026-10-26 · ADR · Workspace Shell (task 1.5)**: `web/src/shell/ShellLayout` (sidebar Thư viện/Workspace/Từ điển → `/library` `/workspace` `/dictionary`, `/` → `/workspace`) + header (`QuotaBadge`, nút ngôn ngữ, email, đăng xuất). Providers gom ở `web/src/Providers.tsx`: I18n → Toast → ApiClientContext → QueryClient (`retry:false`) → Auth. Quota = TanStack Query key `['quota', userId]`, `staleTime` 30s; M2 phải `invalidateQueries` khoá này sau mỗi lần phân tích (hiện `quotaKey` chưa export).
  i18n tự viết nhẹ (không thêm thư viện): `web/src/i18n/messages.ts` là từ điển `vi`/`en` có kiểu (`en: Messages` ép đủ khoá), lưu `localStorage['chordsense.lang']`, đặt `<html lang>`. **Chưa dịch**: trang Login/Register (`auth/messages.ts`) và `toastFor` (api/errors.ts) vẫn chỉ tiếng Việt ⇒ khi làm tiếp phải chuyển sang từ điển chung.
  Test: `server.ts` có handler quota mặc định (mọi màn hình sau đăng nhập đều gọi nó); `setup.ts` xoá `localStorage` mỗi test. E2E thật: `E2E_BACKEND=1 npm run test:e2e` (PowerShell: `$env:E2E_BACKEND=1`) tự bật uvicorn ở `../DE`, cần PG như `DE/.env`; không đặt biến ⇒ spec `shell.spec.ts` bị skip.

- **2026-10-26 · ADR · Ingestion M2 (D1–D7 của kế hoạch M2)**: context `DE/src/ingestion/{domain,application,infrastructure}` + `DE/src/api/routers/{analyze,charts,demo,jobs_ws}.py`; dịch vụ ingestion nằm ở `app.state.ingestion` (`IngestionContainer`) cạnh `app.state.container` của identity.
  **D1** `content_hash` = sha256 WAV chuẩn hoá (chỉ biết sau decode) nên POST không đợi nó: `source_fingerprints` (sha256 byte upload → content_hash, migration `0002`) cho cache-hit tức thì lúc POST (chỉ khi `DATA_DIR/audio/<hash>.wav` còn); cùng nhạc khác encoding thì trừ lúc POST rồi **hoàn** trong job. **D2** chưa có huỷ job (contract không có `cancelled`): FE có "Dừng theo dõi"/"Thử lại". **D3** `GET /charts/{id}` kéo sớm từ 3.1. **D4** timeout 90 s nằm trong `JobExecutor`. **D5** giữ WAV chuẩn hoá của chart file/demo ở `DATA_DIR/audio/<hash>.wav`; audio upload gốc và audio URL xoá ngay; URL YouTube/SoundCloud chỉ lưu embed id. **D6** 100 MB = 100 MiB, 10 phút = 600 000 ms. **D7** WS xác thực bằng subprotocol `['bearer', token]`; từ chối bằng close code 4401/4404 (handshake luôn được chấp nhận để trình duyệt đọc được mã).
  Job: `queued → processing → done|failed` (+ `queued → done` cho cache-hit); quota trừ lúc nhận, hoàn đúng một lần qua `jobs.quota_charged` (`try_mark_refunded`) về đúng **ngày đã trừ** (`jobs.charged_on`). Kịch bản mock theo stem tên file: `demo-<id>`, `fail-ai`, `fail-timeout`, `slow` (chỉ để dev/test).
  SSRF: `ingestion/application/ssrf.py::check_url` (chỉ http/https cổng 80/443, không credentials, chặn IP literal mọi dạng + DNS trả về private, ghim IP đầu) + `fetcher.py::SafeFetcher` (kết nối tới IP đã ghim, `Host`+SNI giữ tên, kiểm lại **mỗi** redirect, tối đa 3). Phần TLS thật chưa chạy được khi offline (kiểm bằng `MockTransport`).
  Hạ tầng: Celery+Redis là tuỳ chọn (`INGEST_JOB_RUNNER=celery`, extra `jobs`); `redis` ghim `<8` vì kombu không có tuỳ chọn protocol và Redis 5.0 không nói RESP3; mỗi task Celery tự dựng executor mới (NullPool) vì mỗi task có event loop riêng. Windows: gọi `localhost` chậm ~2,1 s khi uvicorn chỉ nghe IPv4 ⇒ dùng `127.0.0.1` (Vite proxy, Playwright, README).
  FE: `web/src/analyze/` — `useJob` (WS → polling 2 s; bỏ socket không mở trong 5 s; bỏ qua cập nhật cũ), job đang theo dõi lưu ở `sessionStorage['chordsense.activeJob']` để F5 không mất; tất cả chuỗi mới nằm trong từ điển i18n (`analyze.*`, `demo`, `chart`).

- **2026-10-26 · RULE · Job không bao giờ kẹt (review M2)**: `IngestSettings.job_deadline_s` (300 s) bọc toàn bộ `JobExecutor._execute` (job_timeout_s 90 s chỉ bọc phần phân tích); giới hạn Celery = deadline + 5/+15; `SoftTimeLimitExceeded` do tín hiệu Celery ném từ *ngoài* event loop nên `run_job` bắt nó và chạy `executor.fail_job` trong loop mới; `_fail_safely` không ném (DB sập ⇒ job còn dở, vòng `run_recovery_forever` xử lý sau). Hoàn quota luôn xảy ra **trước** khi publish trạng thái cuối (FE đọc lại quota ngay khi nhận sự kiện). `@app.task(shared=False)` để mỗi Celery app giữ giới hạn riêng.
- **2026-10-26 · RULE · e2e không hermetic nếu âm thanh lặp**: cache nội dung dùng chung mọi user/mọi lần chạy ⇒ audio trùng được phục vụ miễn phí và làm lệch số quota; WAV trong e2e phải ngẫu nhiên.

## FIX — Lỗi đã gặp & cách xử lý

- **2026-10-22 · FIX · msw 3 đổi tên `onUnhandledRequest` → `onUnhandledFrame`**
  Why: tên cũ bị bỏ qua âm thầm (mặc định chỉ cảnh báo) ⇒ request chưa mock không làm test đỏ. TypeScript báo lỗi nên mới phát hiện.
  How: `server.listen({ onUnhandledFrame: 'error' })` trong `web/src/test/setup.ts`; mọi test phải khai báo handler cho request nó phát ra.
- **2026-10-22 · FIX · Đăng nhập xong bị đưa về `/` thay vì trang đang định vào**
  Why: `GuestOnly` render lại ngay khi trạng thái thành `authenticated` và `Navigate to="/"` thắng `navigate(destination)` của form.
  How: `GuestOnly` tự đọc `location.state.from` qua `destinationFrom()` (`auth/destination.ts`). Mọi guard "chuyển hướng khi đã đăng nhập" phải làm vậy.

- **2026-10-22 · FIX · `PostgresSettings` (DE) không đọc `DE/.env`**
  Why: chỉ `Settings` cha có `env_file`; lớp con tạo bằng `default_factory` chỉ đọc biến môi trường thật ⇒ chạy ngoài Docker rơi về mặc định `group26`/`group26`.
  How: `src.persistence.database_url()` tự `load_dotenv(DE/.env)` (không ghi đè env thật). Chưa sửa `config.py` của DE (thuộc team DE); nếu code khác cần đọc `.env` ngoài Docker sẽ gặp lại.

- **2026-10-22 · FIX · `ChordLabel.display_name` (DE) gộp `Cmaj9` và `C9` thành "C9"**
  Why: `major+nat9` và `dominant+nat9` cùng ra "C9" (tương tự nat11/nat13) — sai nhạc lý, ảnh hưởng chế độ Precise.
  How: contract dùng `chord_symbol()` riêng (major + 9/11/13 ⇒ `maj9/11/13`); **chưa sửa `display_name` của DE** vì thuộc team DE/AI — cần báo họ. Ngoài ra từ vựng DE chưa biểu diễn được `G9sus4` (ví dụ của FR-14): `sus` chỉ có "Csus"/"Gsus9"; Basic/Precise ở FE phải làm trên trường có cấu trúc.

- **2026-10-22 · FIX · `crystaldba/postgres-mcp` không chạy được trên Windows máy này**
  Why: bản mới (0.3.0) phụ thuộc `pglast==7.2.0` không có wheel Windows ⇒ `uvx` lùi về bản 0.1.0 là một package khác (nhận DSN qua `argv[1]`); khi thả `mcp` mới nhất thì lỗi `Server has no attribute 'list_resources'`.
  How: dùng `@bytebase/dbhub` (npx, DSN qua env `DSN`). Chỉ-đọc được đảm bảo bằng quyền DB của role `chordsense_ro` (chỉ `SELECT`), không dựa vào cờ của MCP.

## Môi trường (ghi nhận 2026-10-22)

- Máy dev Windows: Node 22, npm 12, Python 3.13.5, ffprobe 8.0, git, `uv` 0.13.5 (cài bằng `pip install uv` ngày 2026-10-22), PostgreSQL 18 (service `postgresql-x64-18`, `psql` chưa trên PATH). **Chưa có**: Docker (user sẽ cài), redis-server.
- `landing_page/` là JSX đơn khối không có test; FE mới đặt ở `web/` (Q1 đã chốt).
- Plugin `superpowers@claude-plugins-official` bật ở project scope (`.claude/settings.json`); MCP khai báo trong `.mcp.json` (cần user duyệt lần đầu). Trên Windows MCP chạy bằng `npx` phải bọc `cmd /c`.

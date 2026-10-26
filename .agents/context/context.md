# context.md — Trạng thái hiện tại của dự án

> Cập nhật cuối mỗi task. Cập nhật lần cuối: **2026-10-26**. Branch: `tienkee`.
> Kế hoạch đầy đủ: [implementation-plan.md](implementation-plan.md). Quyết định/quy tắc: [../memory/memory.md](../memory/memory.md).

## Giai đoạn
**M2 hoàn tất (2026-10-26)** — upload / link / bài demo → stepper realtime → chart mock lưu vào thư viện. Tiếp theo: **M3** (Chord Viewer & Audio Sync). Kế hoạch chi tiết đã chạy: [docs/superpowers/plans/2026-10-26-m2-ingestion-mock-pipeline.md](../../docs/superpowers/plans/2026-10-26-m2-ingestion-mock-pipeline.md).

M2 gồm: `DE/src/ingestion/` (state machine job, SSRF guard, SafeFetcher, Normalizer ffmpeg, MockAnalysisEngine, JobExecutor, Submission, demo catalog, dọn file tạm, Celery+Redis), `DE/src/api/routers/{analyze,charts,demo,jobs_ws}.py`, migration `0002`; FE `web/src/analyze/` (Source Selector, useJob WS→polling, JobTracker, QuotaBanner, DemoTracks), `web/src/pages/{WorkspacePage,ChartPage}.tsx`. Kiểm chứng: `DE` pytest xanh (680+), `web` vitest 220 + lint + build, Playwright 9 e2e với backend thật (`E2E_BACKEND=1`).

**Chạy thử tay:** `rtk uv run python scripts/gen_demo_audio.py` (một lần, từ `DE/`) → `rtk uv run uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000` + `rtk npm run dev` (từ `web/`). Chế độ Celery: đặt `INGEST_JOB_RUNNER=celery`, chạy thêm `rtk uv run celery -A src.ingestion.worker worker --pool=solo` (cần `uv sync --extra jobs` và Redis; bản Redis 5.0 trên máy này → redis-py ghim `<8`).

(M1 — lịch sử) Task 1.0 ✅, 1.1 ✅, 1.2 ✅, 1.3 ✅ (2026-10-22), 1.4 ✅, 1.5 ✅ (2026-10-26) ⇒ **M1 hoàn tất**. Tiếp theo: M2 (Ingestion + Mock Pipeline).

Task 1.5: Workspace Shell — layout sidebar/header, `QuotaBadge` (tier + còn x/y, skeleton, trạng thái hết lượt/không giới hạn/lỗi), toast, i18n vi/en (lưu localStorage), 3 trang trống có skeleton. `web` 69 test, lint (0 cảnh báo), build xanh; e2e thật (đăng ký → Shell → badge `Free · còn 5/5` → đổi English → reload → đăng xuất) xanh với `E2E_BACKEND=1`.

Task 1.4: `create_app()` (`DE/src/api/app.py`) gắn auth/users thật lên PG, `/healthz`, lỗi chuẩn hoá (422 có `detail`, 404/405/500), `X-Request-ID` + log structlog, CORS cấu hình. `DE` 306 pytest xanh (19 mới, gồm 2 integration trên `chordsense_test`), ruff sạch; smoke bằng uvicorn trên `chordsense_app` ok. FE: `ApiError.fieldErrors/requestId` + `toastFor`; `web` 51 test, lint, build, e2e xanh. Chạy backend: `rtk uv run uvicorn src.api.app:create_app --factory --port 8000` (từ `DE/`).

Task 1.3: BE `identity` + `api` (register/login/refresh/logout, `GET /users/me`, `GET /users/me/quota`, `require_roles`, `PolicyProvider`), `DE` 287 pytest xanh (không đổi DB mới), ruff sạch. FE: client HTTP có interceptor refresh, `AuthProvider`, route guard, trang Login/Register, `web` 43 test + lint + build + e2e smoke xanh. `openapi.json` giờ gồm cả schema auth.

Task 1.2: Alembic `0001` đã áp lên DB dev `chordsense_app` (8 bảng trong schema `app`; `chordsense_ro` chỉ đọc được, ghi bị từ chối) và chạy lên/xuống/lên sạch trên `chordsense_test`. `DE`: 147 pytest xanh (31 mới (23 integration cần PG, 8 unit)), ruff sạch.

Task 1.1: `analysis_contract` (models, `apply_overrides`, port `AnalysisEngine`), 7 fixture kết quả + 3 chart + jobs/errors/requests trong `contracts/fixtures/`, `contracts/openapi.json`, `web/src/api/{generated,types}`. `DE`: 116 pytest xanh (76 mới), ruff sạch; `web`: 13 test, lint, build xanh.

Baseline 1.0:
- `DE/`: `uv sync --extra api --extra db`; `pytest` 40 passed; ruff **khoanh vào mã mới** (mã cũ có 190 lỗi lint / 71 file chưa format, đã `extend-exclude` trong `pyproject.toml`; bỏ dần khi dọn).
- `web/` (Vite 8 + React 19 + TS 6 + Tailwind v4 + Router/TanStack Query/Zustand; Vitest + Testing Library + MSW + Playwright/Chromium): `npm test`, `lint`, `build`, `test:e2e` đều xanh.
- DB PG18: `chordsense_app`, `chordsense_test` (schema `app`), role `chordsense_app` + `chordsense_ro`; MCP `playwright`/`context7`/`postgres(dbhub)` đã connected.

## Milestone
| # | Tên | Pri | Trạng thái |
| :-- | :-- | :-- | :-- |
| M1 | Foundation, DB, Auth, Workspace Shell | P0 | ✅ 1.0 ✅ · 1.1 ✅ · 1.2 ✅ · 1.3 ✅ · 1.4 ✅ · 1.5 ✅ |
| M2 | Ingestion + Mock Pipeline | P0 | ✅ 2.1–2.8 |
| M3 | Chord Viewer & Audio Sync | P1 | ⏳ (kế tiếp) |
| M4 | Harmony / Jazz Theory | P1 | ⏳ |
| M5 | Tích hợp AI thật | P2 | ⏳ |

## Endpoint đã xong
`GET /healthz` · `POST /api/v1/auth/{register,login,refresh,logout}` · `GET /api/v1/users/me`, `/users/me/quota` · `POST /api/v1/analyze/file` (multipart `file`, 202 `{job_id}`), `/analyze/url` (`{url}`), `/analyze/demo/{id}` (không trừ quota) · `GET /api/v1/analyze/status/{id}` · `WS /api/v1/ws/jobs/{id}` (subprotocol `bearer`, token) · `GET /api/v1/charts/{id}` (chỉ chủ sở hữu; M3.1 mở rộng) · `GET /api/v1/demo-tracks`. Mã lỗi: 413 `file_too_large`/`duration_too_long`, 415 `unsupported_format`, 422 `ssrf_blocked`/`url_unresolvable`, 429 `quota_exceeded`, 502 `ai_failure`, 503 `demo_unavailable`.

## Đang làm dở
_(không)_

## Blockers
1. Docker Desktop đã cài nhưng `docker` chưa trên PATH của session Claude; không cần Docker cho M2 vì Redis native chạy ở `127.0.0.1:6379` (5.0.14). Cổng 5432: không bật compose `postgres:16` của DE cùng lúc với PG18 native.
2. `openapi-typescript@7` yêu cầu TypeScript 5 (web dùng TS 6) ⇒ chạy qua `npx` trong script `gen:api`, không cài vào devDependencies.

## Rủi ro đang mở
- **Engine là mock**: kết quả lấy từ 7 fixture theo `hash % 7` (hoặc theo tên file `demo-*`/`fail-ai`/`fail-timeout`/`slow`). Toàn bộ 'phân tích' ở M2 là giả cho tới M5; `FakeMediaResolver` cũng chưa tải gì từ URL (yt-dlp thật hoãn đến 5.3).
- Job chỉ được phục hồi khi API khởi động (`recover_orphans`, tuổi > `job_timeout_s`): worker chết giữa chừng thì job treo tới lần khởi động kế tiếp.
- Celery trên Windows chỉ chạy `--pool=solo`; giới hạn thời gian cứng của Celery không có tác dụng ở đó, chỉ `asyncio.timeout` của executor bảo vệ.
- Cache theo byte upload chỉ đúng khi WAV chuẩn hoá còn trong `DATA_DIR/audio/` (xoá thư mục này ⇒ lần upload lại đi qua hàng đợi, trừ rồi hoàn quota).
- `InMemoryRevocationStore`: refresh token đã thu hồi **hợp lệ trở lại sau khi server khởi động lại**, và không chia sẻ giữa nhiều worker. Phải thay bằng Redis (M2.6) hoặc bảng PG trước khi chạy thật; `build_container` hiện vẫn dùng nó (chỉ dev).
- Refresh token nằm trong `localStorage` ⇒ lộ nếu có XSS. Chấp nhận cho MVP (access token ở bộ nhớ, rotate mỗi lần dùng); cân nhắc cookie `httpOnly` khi triển khai thật.
- `AUTH_JWT_SECRET` dev nằm trong `DE/.env` (gitignored). Môi trường thật phải đặt biến riêng ≥ 32 ký tự.

## Việc cần báo team DE/AI
- `ChordLabel.display_name` hiển thị `Cmaj9` và `C9` giống nhau ("C9"); từ vựng chưa có `sus4` + extension kiểu `G9sus4`; chưa có `bass` (slash chord). Xem memory.md.

## Việc kế tiếp
1. **M3** (xem implementation-plan.md §3): Chart API đầy đủ (list, ETag, `GET /charts/{id}/audio` hỗ trợ Range), Smart Grid, player đồng bộ, spike 3.2 (độ lệch < 100 ms). Trang `/charts/:id` hiện chỉ là bảng hợp âm tạm.
2. Thay `InMemoryRevocationStore` (Redis hoặc bảng PG) trước khi chạy thật; dịch nốt Login/Register và `toastFor` sang từ điển i18n chung.
3. Khi có team AI: thêm adapter thật vào `ENGINES` của `DE/tests/contract/test_analysis_engine_contract.py` (tiêu chí nghiệm thu).

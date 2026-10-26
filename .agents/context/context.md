# context.md — Trạng thái hiện tại của dự án

> Cập nhật cuối mỗi task. Cập nhật lần cuối: **2026-10-26**. Branch: `tienkee`.
> Kế hoạch đầy đủ: [implementation-plan.md](implementation-plan.md). Quyết định/quy tắc: [../memory/memory.md](../memory/memory.md).

## Giai đoạn
**M1 đang chạy.** Task 1.0 ✅, 1.1 ✅, 1.2 ✅, 1.3 ✅ (2026-10-22), 1.4 ✅, 1.5 ✅ (2026-10-26) ⇒ **M1 hoàn tất**. Tiếp theo: M2 (Ingestion + Mock Pipeline).

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
| M2 | Ingestion + Mock Pipeline | P0 | ⏳ (kế tiếp) |
| M3 | Chord Viewer & Audio Sync | P1 | ⏳ |
| M4 | Harmony / Jazz Theory | P1 | ⏳ |
| M5 | Tích hợp AI thật | P2 | ⏳ |

## Endpoint đã xong
`GET /healthz`, `POST /api/v1/auth/{register,login,refresh,logout}`, `GET /api/v1/users/me`, `GET /api/v1/users/me/quota` (chạy được trên PG thật qua `create_app()`).

## Đang làm dở
_(không)_

## Blockers
1. Docker Desktop đã cài (2026-10-26) nhưng `docker` chưa trên PATH của session Claude; máy đã có `redis-server` native ở 127.0.0.1:6379 — có thể dùng cho M2.6, chưa kiểm tra phiên bản. Cổng 5432: không bật compose `postgres:16` của DE cùng lúc với PG18 native.
2. `openapi-typescript@7` yêu cầu TypeScript 5 (web dùng TS 6) ⇒ chạy qua `npx` trong script `gen:api`, không cài vào devDependencies.

## Rủi ro đang mở
- `InMemoryRevocationStore`: refresh token đã thu hồi **hợp lệ trở lại sau khi server khởi động lại**, và không chia sẻ giữa nhiều worker. Phải thay bằng Redis (M2.6) hoặc bảng PG trước khi chạy thật; `build_container` hiện vẫn dùng nó (chỉ dev).
- Refresh token nằm trong `localStorage` ⇒ lộ nếu có XSS. Chấp nhận cho MVP (access token ở bộ nhớ, rotate mỗi lần dùng); cân nhắc cookie `httpOnly` khi triển khai thật.
- `AUTH_JWT_SECRET` dev nằm trong `DE/.env` (gitignored). Môi trường thật phải đặt biến riêng ≥ 32 ký tự.

## Việc cần báo team DE/AI
- `ChordLabel.display_name` hiển thị `Cmaj9` và `C9` giống nhau ("C9"); từ vựng chưa có `sus4` + extension kiểu `G9sus4`; chưa có `bass` (slash chord). Xem memory.md.

## Việc kế tiếp
1. M2 (xem implementation-plan.md §3): upload/link ingestion + mock pipeline. Trước M2.6 phải thay `InMemoryRevocationStore` (Redis native đang chạy ở :6379 hoặc bảng PG) và quyết định Redis/Celery thật.
2. Dịch nốt Login/Register và `toastFor` sang từ điển i18n chung; sau mỗi lần phân tích gọi `invalidateQueries(['quota', userId])`.

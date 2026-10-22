# context.md — Trạng thái hiện tại của dự án

> Cập nhật cuối mỗi task. Cập nhật lần cuối: **2026-10-22**. Branch: `tienkee`.
> Kế hoạch đầy đủ: [implementation-plan.md](implementation-plan.md). Quyết định/quy tắc: [../memory/memory.md](../memory/memory.md).

## Giai đoạn
**M1 đang chạy.** Task 1.0 ✅, 1.1 ✅, 1.2 ✅ (2026-10-22). Tiếp theo: 1.3 (Auth + RBAC).

Task 1.2: Alembic `0001` đã áp lên DB dev `chordsense_app` (8 bảng trong schema `app`; `chordsense_ro` chỉ đọc được, ghi bị từ chối) và chạy lên/xuống/lên sạch trên `chordsense_test`. `DE`: 147 pytest xanh (31 mới (23 integration cần PG, 8 unit)), ruff sạch.

Task 1.1: `analysis_contract` (models, `apply_overrides`, port `AnalysisEngine`), 7 fixture kết quả + 3 chart + jobs/errors/requests trong `contracts/fixtures/`, `contracts/openapi.json`, `web/src/api/{generated,types}`. `DE`: 116 pytest xanh (76 mới), ruff sạch; `web`: 13 test, lint, build xanh.

Baseline 1.0:
- `DE/`: `uv sync --extra api --extra db`; `pytest` 40 passed; ruff **khoanh vào mã mới** (mã cũ có 190 lỗi lint / 71 file chưa format, đã `extend-exclude` trong `pyproject.toml`; bỏ dần khi dọn).
- `web/` (Vite 8 + React 19 + TS 6 + Tailwind v4 + Router/TanStack Query/Zustand; Vitest + Testing Library + MSW + Playwright/Chromium): `npm test`, `lint`, `build`, `test:e2e` đều xanh.
- DB PG18: `chordsense_app`, `chordsense_test` (schema `app`), role `chordsense_app` + `chordsense_ro`; MCP `playwright`/`context7`/`postgres(dbhub)` đã connected.

## Milestone
| # | Tên | Pri | Trạng thái |
| :-- | :-- | :-- | :-- |
| M1 | Foundation, DB, Auth, Workspace Shell | P0 | 🔧 1.0 ✅ · 1.1 ✅ · 1.2 ✅ · 1.3–1.5 ⏳ |
| M2 | Ingestion + Mock Pipeline | P0 | ⏳ |
| M3 | Chord Viewer & Audio Sync | P1 | ⏳ |
| M4 | Harmony / Jazz Theory | P1 | ⏳ |
| M5 | Tích hợp AI thật | P2 | ⏳ |

## Endpoint đã xong
_(chưa có — backend `DE/` chưa có FastAPI app)_

## Đang làm dở
_(không)_

## Blockers
1. Docker Desktop (user đang cài) → Redis/Celery thật (M2.6) chưa chạy được; unit test dùng fake/in-process. Cổng 5432: không bật compose `postgres:16` của DE cùng lúc với PG18 native.
2. `openapi-typescript@7` yêu cầu TypeScript 5 (web dùng TS 6) ⇒ chạy qua `npx` trong script `gen:api`, không cài vào devDependencies.

## Việc cần báo team DE/AI
- `ChordLabel.display_name` hiển thị `Cmaj9` và `C9` giống nhau ("C9"); từ vựng chưa có `sus4` + extension kiểu `G9sus4`; chưa có `bass` (slash chord). Xem memory.md.

## Việc kế tiếp
1. Task 1.3: Auth + RBAC (bcrypt cost ≥ 12, JWT, `require_role`, `GET /users/me/quota`, `PolicyProvider`) — cần thêm dependency (bcrypt, PyJWT, email-validator) vào `pyproject`.
2. Task 1.4: app factory + lỗi chuẩn + `/healthz`; 1.5: Workspace Shell (FE).

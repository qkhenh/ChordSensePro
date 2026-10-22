# context.md — Trạng thái hiện tại của dự án

> Cập nhật cuối mỗi task. Cập nhật lần cuối: **2026-10-22**. Branch: `tienkee`.
> Kế hoạch đầy đủ: [implementation-plan.md](implementation-plan.md). Quyết định/quy tắc: [../memory/memory.md](../memory/memory.md).

## Giai đoạn
**M1 đang chạy.** Task 1.0 ✅ xong (2026-10-22). Tiếp theo: 1.1 (contract package + fixtures, test trước).

Baseline 1.0:
- `DE/`: `uv sync --extra api --extra db`; `pytest` 40 passed; ruff **khoanh vào mã mới** (mã cũ có 190 lỗi lint / 71 file chưa format, đã `extend-exclude` trong `pyproject.toml`; bỏ dần khi dọn).
- `web/` (Vite 8 + React 19 + TS 6 + Tailwind v4 + Router/TanStack Query/Zustand; Vitest + Testing Library + MSW + Playwright/Chromium): `npm test`, `lint`, `build`, `test:e2e` đều xanh.
- DB PG18: `chordsense_app`, `chordsense_test` (schema `app`), role `chordsense_app` + `chordsense_ro`; MCP `playwright`/`context7`/`postgres(dbhub)` đã connected.

## Milestone
| # | Tên | Pri | Trạng thái |
| :-- | :-- | :-- | :-- |
| M1 | Foundation, DB, Auth, Workspace Shell | P0 | 🔧 1.0 ✅ · 1.1–1.5 ⏳ |
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

## Việc kế tiếp
1. Task 1.1: `DE/src/analysis_contract/` + `contracts/fixtures/` (test trước) → sinh `contracts/openapi.json` → `npm run gen:api`.
2. Task 1.2: Alembic + schema `app`.

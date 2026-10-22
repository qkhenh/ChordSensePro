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

## FIX — Lỗi đã gặp & cách xử lý

- **2026-10-22 · FIX · `crystaldba/postgres-mcp` không chạy được trên Windows máy này**
  Why: bản mới (0.3.0) phụ thuộc `pglast==7.2.0` không có wheel Windows ⇒ `uvx` lùi về bản 0.1.0 là một package khác (nhận DSN qua `argv[1]`); khi thả `mcp` mới nhất thì lỗi `Server has no attribute 'list_resources'`.
  How: dùng `@bytebase/dbhub` (npx, DSN qua env `DSN`). Chỉ-đọc được đảm bảo bằng quyền DB của role `chordsense_ro` (chỉ `SELECT`), không dựa vào cờ của MCP.

## Môi trường (ghi nhận 2026-10-22)

- Máy dev Windows: Node 22, npm 12, Python 3.13.5, ffprobe 8.0, git, `uv` 0.13.5 (cài bằng `pip install uv` ngày 2026-10-22), PostgreSQL 18 (service `postgresql-x64-18`, `psql` chưa trên PATH). **Chưa có**: Docker (user sẽ cài), redis-server.
- `landing_page/` là JSX đơn khối không có test; FE mới đặt ở `web/` (Q1 đã chốt).
- Plugin `superpowers@claude-plugins-official` bật ở project scope (`.claude/settings.json`); MCP khai báo trong `.mcp.json` (cần user duyệt lần đầu). Trên Windows MCP chạy bằng `npx` phải bọc `cmd /c`.

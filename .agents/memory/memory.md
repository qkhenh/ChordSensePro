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

## FIX — Lỗi đã gặp & cách xử lý

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

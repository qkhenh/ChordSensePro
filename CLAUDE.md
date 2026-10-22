<!-- rtk-instructions v2 -->
# Command output

Command output here is condensed to save tokens, keeping every signal and
dropping costly noise. Treat it as the complete result: run commands
normally, and batch related commands into one call to avoid extra turns.
Truncated results state their recovery path in their own output. Re-run a
command as `rtk proxy <cmd>` only when its result is unusable: empty when
output was clearly expected, contradicting its exit code, or garbled.

# RTK Prefix Rule
Prefix all execution commands with `rtk`:
- `rtk git status`, `rtk git diff`, `rtk git commit -m "..."`
- `rtk uv run pytest`, `rtk uv run ruff check .`
- `rtk npm run build`, `rtk npm run lint`
<!-- /rtk-instructions -->

# ChordSense Pro — Claude Code Guidelines

## 1. Project Overview & Context
**ChordSense Pro** is an AI-powered, context-aware interactive practice workspace and harmonic analysis system for musicians.
- **Frontend / UI**: `landing_page/` (React 19, Vite 8, Tailwind CSS v4, Motion, Lucide React, Oxlint).
- **Backend & Data Pipeline**: `DE/` (Python 3.11+, `uv`, FastAPI, Pydantic 2, SQLAlchemy/Asyncpg, PyTorch, Librosa, Demucs, Celery/Airflow DAGs).
- **Music Theory & Specifications**: `markdown/` (`ChordSense.MD`, `Lý Thuyết Hợp Âm Jazz.md`, `use_case_specifications.md`).
- **Academic Research**: `latex_last_project/` (LaTeX research paper & documentation).

---

## 2. Dual-Agent Collaboration Role: Claude Code CLI
In this repository, you work in tandem with **Antigravity IDE (Gemini Pro)**:
- **Your Role (The Senior Craftsman & TDD Specialist)**:
  - Deep algorithmic logic (Harmonic analysis, Roman numeral analysis, audio signal processing).
  - Test-Driven Development (TDD): Write unit tests first, implement features, ensure all tests pass.
  - Refactoring, bug-fixing, and atomic Git commits.
- **Antigravity's Role (The Architect & Researcher)**:
  - Analyzes large theoretical docs (`main.pdf`, jazz theory, use case specs).
  - Generates architecture specs, task briefs, and high-level roadmaps.
  - Reviews code diffs, maintains diagrams, and updates project documentation.

---

## 3. Standard Development Commands (Always use `rtk`)

### Backend (`DE/`)
Managed via `uv`:
```bash
# Run tests
rtk uv run pytest -v

# Run lint and format checks
rtk uv run ruff check .
rtk uv run ruff format --check .

# Apply auto-fixes
rtk uv run ruff check --fix .
rtk uv run ruff format .
```

### Frontend (`landing_page/`)
Managed via `npm`:
```bash
# Linting
rtk npm run lint

# Production build check
rtk npm run build
```

### Git Workflow
```bash
rtk git status
rtk git add <files>
rtk git commit -m "feat(module): description"
```

---

## 4. Code Quality & Implementation Rules
1. **TDD First**: Before modifying or creating core music theory algorithms or API endpoints, create/update corresponding tests in `DE/tests/`.
2. **Never Break Type Safety**:
   - Python: Strict Pydantic models for request/response payloads; full type annotations.
   - Frontend: Strict component prop checking, clean React 19 hook usage.
3. **Audio & Math Precision**:
   - Sample rates, FFT windows, hop sizes, and timing offsets must strictly adhere to specifications in `markdown/ChordSense.MD`.
   - Chord symbol representations must follow the jazz/harmonic conventions defined in `markdown/Lý Thuyết Hợp Âm Jazz.md`.
4. **Clean Handoff**:
   - Always run lint and tests before completing a task.
   - Make atomic, descriptive git commits so Antigravity can review changes easily.

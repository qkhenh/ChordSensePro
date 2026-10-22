# AGENTS.md — Dual-Agent Operating Protocol (ChordSense Pro)

This document establishes the collaboration protocol between **Antigravity IDE (Google AI Pro)** and **Claude Code CLI (Anthropic Pro)** within the `ChordSensePro` workspace.

---

## 1. Project Context: ChordSense Pro

**ChordSense Pro** is an AI-powered, context-aware interactive practice workspace and harmonic analysis system for musicians.
- **Core Value Proposition**: Bridges the gap between raw AI chord recognition (ACR) and musician-grade harmonic comprehension (Roman Numeral Analysis, Modal/Scale mapping, interactive practice sync player).
- **Subsystems**:
  - `landing_page/`: React 19 + Vite 8 + Tailwind CSS v4 web application.
  - `DE/`: Data engineering and backend pipeline (FastAPI, PyTorch, Librosa, Demucs, Celery/Airflow DAGs, PostgreSQL/Asyncpg, uv).
  - `markdown/`: Music theory notes, architectural specifications, use case docs (`ChordSense.MD`, `Lý Thuyết Hợp Âm Jazz.md`).
  - `latex_last_project/`: Academic research papers, thesis documentation (`main.pdf`, `main.tex`).
  - `diagrams/` & `build_diagram/`: System architecture and data flow diagrams.

---

## 2. Division of Responsibilities (Phân Vai Hiệp Đồng)

| Agent | Core Persona | Primary Domain & Tasks | Key Tools & Interface |
| :--- | :--- | :--- | :--- |
| **Antigravity** *(Google AI Pro)* | **The Architect & Researcher** | • Comprehensive context ingestion (PDFs, multi-chapter research, large markdown specs).<br>• System architecture design & CodeGraph exploration (`codegraph`, `openflowkit`).<br>• Drafting detailed Task Specs & Implementation Plans for Claude Code.<br>• Global code review, UI testing via browser subagent, updating documentation. | IDE Chat Panel, Artifacts (`.md`), CodeGraph, OpenFlow, Browser Subagent. |
| **Claude Code** *(Anthropic Pro)* | **The Senior Craftsman & TDD Engineer** | • Deep algorithmic reasoning (Harmonic engine math, audio DSP, jazz chord voice-leading).<br>• Test-Driven Development (TDD): Write unit tests, implement logic, iterate until 100% pass.<br>• Refactoring complex modules, fixing subtle edge cases & tracebacks.<br>• Atomic Git commits and branch hygiene. | Terminal CLI (`claude`), Shell execution via RTK. |

---

## 3. RTK (Rust Token Killer) Mandatory Rule

Both agents **must** use RTK to preserve context window and prevent subscription rate-limit exhaustion:
1. **Command Prefix**: Every shell command must be prefixed with `rtk`:
   - `rtk git status`, `rtk git diff`, `rtk git commit -m "..."`
   - `rtk uv run pytest`, `rtk uv run ruff check .`
   - `rtk npm run build`, `rtk npm run lint`
2. **Command Output Treatment**: Output is condensed by design. Do not re-run commands unless the result is corrupted. Use `rtk proxy <cmd>` only if raw unfiltered output is strictly necessary.
3. **Telemetry & Economics**: Check token savings with `rtk gain` and Claude Code spending vs savings with `rtk cc-economics`.

---

## 4. The 4-Step Collaborative Workflow (Quy Trình Phối Hợp)

```
[ Step 1: Design (Antigravity) ]
  • Analyzes theory docs & codebase via CodeGraph.
  • Formulates Task Specification in a dedicated brief or chat prompt.
              │
              ▼
[ Step 2: Implementation (Claude Code) ]
  • Receives the Task Specification in the integrated terminal.
  • Writes tests first in `DE/tests/` or frontend tests.
  • Implements code and runs self-healing test loops (`rtk uv run pytest`).
  • Commits changes cleanly (`rtk git commit`).
              │
              ▼
[ Step 3: Review & Verification (Antigravity) ]
  • Reviews `rtk git diff`.
  • Validates cross-module integration and potential side-effects.
  • Tests visual/web components if applicable.
              │
              ▼
[ Step 4: Sync & Documentation (Both) ]
  • Antigravity updates Markdown / LaTeX documentation to reflect code changes.
  • Git branch is ready for deployment / merge.
```

---

## 5. Development & Verification Commands

### Backend (`DE/`)
```bash
# Navigate & run tests
rtk uv run pytest -v

# Code quality
rtk uv run ruff check .
rtk uv run ruff format --check .

# Auto-fix linting
rtk uv run ruff check --fix .
rtk uv run ruff format .
```

### Frontend (`landing_page/`)
```bash
# Code checking
rtk npm run lint

# Production build
rtk npm run build
```

---

## 6. Coding & Handoff Standards

1. **No Silent Breakages**: If modifying existing functions or data models (Pydantic models, API schemas, Harmonic notation), existing tests must pass or be explicitly migrated.
2. **Music Theory Accuracy**: Harmonic notation, Roman numerals, chord intervals, and jazz scales must strictly conform to `markdown/Lý Thuyết Hợp Âm Jazz.md`.
3. **Git Hygiene**:
   - Format: `<type>(<scope>): <short description>`
   - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.
   - Never commit sensitive keys, `.env`, or large raw media artifacts (`.wav`, `.mp3` dumps).

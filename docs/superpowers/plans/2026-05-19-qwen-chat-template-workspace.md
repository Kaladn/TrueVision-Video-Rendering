# Qwen Chat Template Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Qwen a regular project chat beside a render/template workspace, with daily chat logs and reusable generation templates.

**Architecture:** Extend the existing studio server instead of adding a new backend. Store chat logs in one flat JSONL file per day under `storage/chats/`, store render templates as individual JSON files under `storage/templates/`, and expose simple GET/POST endpoints for the HTML studio. The UI keeps Qwen conversation separate from renderer execution while showing the current render template JSON in a neighboring pane.

**Tech Stack:** Python `http.server`, JSON/JSONL files, existing single-file HTML/CSS/JS studio, existing `unittest` suite.

---

### Task 1: Server Storage Contracts

**Files:**
- Modify: `scripts/truevision_studio_server.py`
- Test: `tests/test_truevision_studio_server.py`

- [x] Add `chats` and `templates` to `STORAGE_LANES`.
- [x] Add `append_chat_message()` to write one JSON object per line into `storage/chats/YYYY-MM-DD.jsonl`.
- [x] Add `read_chat_log()` to read today's file or a requested date.
- [x] Add `save_template()`, `list_templates()`, and `delete_template()` for flat template files.
- [x] Add `probe_media_duration()` and `build_generation_template_from_request()` so audio-backed templates get an exact timeline duration.

### Task 2: Server API Routes

**Files:**
- Modify: `scripts/truevision_studio_server.py`
- Test: `tests/test_truevision_studio_server.py`

- [x] Add `GET /api/chat/today`.
- [x] Add `POST /api/chat/log`.
- [x] Add `GET /api/templates`.
- [x] Add `POST /api/templates/save`.
- [x] Add `POST /api/templates/delete`.
- [x] Add `POST /api/media/probe`.

### Task 3: Studio UI Split

**Files:**
- Modify: `ui/truevision_state_media_studio.html`
- Test: `tests/test_truevision_state_media_studio_html.py`

- [x] Rename the center chat lane to normal Qwen chat language.
- [x] Add a render workspace window beside/under chat with `templateJson`.
- [x] Add audio path, duration sync, save template, refresh templates, and delete template controls.
- [x] Save every chat message to the daily chat endpoint.
- [x] Load today's daily chat on startup.
- [x] Show templates in a simple list with Load/Delete buttons.

### Task 4: Verification

**Files:**
- Modify: `README.md`

- [x] Document new server endpoints.
- [x] Run `git diff --check`.
- [x] Run `$env:PYTHONPATH='scripts;modules;.'; python -m unittest discover -s tests -v`.
- [x] Commit the finished package.

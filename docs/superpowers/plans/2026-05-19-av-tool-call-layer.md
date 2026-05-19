# AV Tool Call Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an audio/video-only tool-call layer so Qwen can request safe media actions through validated JSON instead of pretending to execute work.

**Architecture:** Introduce `truevision_runtime/av_tools/` with a registry, policy gate, runner, receipts, and recalibration helpers. The studio server exposes list/call endpoints that route tool requests through policy and write receipts for accepted and rejected calls. Tools are scoped to audio/video templates, previews, render preparation, markers, recalibration notes, and learning records.

**Tech Stack:** Python stdlib, existing storage layout, existing studio server, `unittest`.

---

### Task 1: AV Tool Contracts

**Files:**
- Create: `truevision_runtime/av_tools/av_tool_registry.py`
- Create: `truevision_runtime/av_tools/av_tool_policy.py`
- Create: `tests/test_av_tool_layer.py`

- [x] Define the Phase 1 AV tool registry.
- [x] Mark `video_execute_full_render` and `template_delete` as human-confirmed actions.
- [x] Reject unknown tools.
- [x] Reject unsafe template filenames and non-AV storage lanes.

### Task 2: Receipts And Recalibration

**Files:**
- Create: `truevision_runtime/av_tools/av_tool_receipts.py`
- Create: `truevision_runtime/av_tools/av_recalibration.py`
- Modify: `tests/test_av_tool_layer.py`

- [x] Write receipt JSON for every tool call.
- [x] Store recalibration notes and time markers as JSONL events.
- [x] Keep time-marker records structured by template/source artifact/time.

### Task 3: AV Tool Runner

**Files:**
- Create: `truevision_runtime/av_tools/av_tool_runner.py`
- Modify: `tests/test_av_tool_layer.py`

- [x] Implement `audio_probe_duration`.
- [x] Implement template create/load/save/patch/variant/list.
- [x] Implement marker and recalibration note tools.
- [x] Implement preview/full render preparation manifests.
- [x] Reject full render execution unless human confirmation is explicit.

### Task 4: Studio API

**Files:**
- Modify: `scripts/truevision_studio_server.py`
- Modify: `ui/truevision_state_media_studio.html`
- Modify: `README.md`
- Modify: `tests/test_truevision_studio_server.py`
- Modify: `tests/test_truevision_state_media_studio_html.py`

- [x] Add `GET /api/av-tools`.
- [x] Add `POST /api/av-tools/call`.
- [x] Add Qwen system prompt hints that tools are AV-only and validated server-side.
- [x] Document the tool-call endpoints.

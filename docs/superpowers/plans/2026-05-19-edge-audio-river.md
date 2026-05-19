# Edge Audio River Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-pass full-song audio-reactive video for `Edge Of The World (I Am Your Nightmare)` using a black-field, swirling color-river visual with no lettering or glyphs.

**Architecture:** A standalone Python renderer decodes the MP3 with FFmpeg, extracts frame-level audio features, renders deterministic river frames, muxes the original audio into the final MP4, and writes manifest/report/state JSONL outputs. This is a small TrueVision-shaped generation lane, not a MilkDrop engine integration.

**Tech Stack:** Python 3, NumPy, OpenCV, FFmpeg/FFprobe, existing repo testing with `unittest`.

---

### Task 1: Audio Feature Core

**Files:**
- Create: `scripts/truevision_edge_audio_river.py`
- Create: `tests/test_truevision_edge_audio_river.py`

- [ ] **Step 1: Write tests for feature extraction**

Create tests that synthesize a short NumPy audio buffer, call `measure_audio_features`, and assert that it returns per-frame `rms`, `bass`, `mid`, `high`, and `beat` values in the `0..1` range.

- [ ] **Step 2: Implement decode and feature extraction**

Implement `decode_audio_mono`, `measure_audio_features`, and `normalize_feature_series`.

- [ ] **Step 3: Verify**

Run:

```powershell
$env:PYTHONPATH="D:\TrueVision_Generation_Lab\scripts;D:\TrueVision_Generation_Lab\modules;D:\TrueVision_Generation_Lab"
python tests\test_truevision_edge_audio_river.py -v
```

Expected: all tests pass.

### Task 2: River Frame Renderer

**Files:**
- Modify: `scripts/truevision_edge_audio_river.py`
- Modify: `tests/test_truevision_edge_audio_river.py`

- [ ] **Step 1: Write tests for frame rendering**

Test that `render_river_frame` returns a `height x width x 3` `uint8` frame, contains non-black river pixels, and records no text/glyph layer in its metadata.

- [ ] **Step 2: Implement deterministic river render**

Render a black background with a multi-strand sine/snake river. Drive line width, hue speed, bloom, and shock rings from audio features.

- [ ] **Step 3: Verify**

Run:

```powershell
python tests\test_truevision_edge_audio_river.py -v
```

Expected: all tests pass.

### Task 3: Full Render Bundle

**Files:**
- Modify: `scripts/truevision_edge_audio_river.py`
- Modify: `tests/test_truevision_edge_audio_river.py`

- [ ] **Step 1: Write tests for manifest output**

Use a tiny synthetic WAV file and render a sub-second clip into a temporary directory. Assert that the manifest, report, JSONL state, and MP4 are written.

- [ ] **Step 2: Implement `generate_edge_audio_river`**

Write the renderer loop, stream frames into FFmpeg, write JSONL frame state, write thumbnail, write manifest and report.

- [ ] **Step 3: Verify**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass.

### Task 4: Render Track 1

**Files:**
- Output only under `outputs/edge_of_the_world_audio_river/`

- [ ] **Step 1: Render full song**

Run:

```powershell
python scripts\truevision_edge_audio_river.py `
  --audio "C:\Users\mydyi\OneDrive\Documents\Desktop\Album_Builds\Machine_Dread_Album_Sequenced\01_ordered_audio\01 - Edge Of The World (I Am Your Nightmare).mp3" `
  --lyrics "C:\Users\mydyi\OneDrive\Documents\Desktop\Full Album Lyrics_sound.txt" `
  --output-root outputs\edge_of_the_world_audio_river `
  --width 1280 `
  --height 720 `
  --fps 30
```

Expected: full-song MP4 with audio, no text/glyphs, manifest, report, thumbnail, and frame-state JSONL.

### Task 5: Commit

**Files:**
- Add: `scripts/truevision_edge_audio_river.py`
- Add: `tests/test_truevision_edge_audio_river.py`
- Add: `docs/superpowers/plans/2026-05-19-edge-audio-river.md`

- [ ] **Step 1: Run final verification**

```powershell
python -m unittest discover -s tests -v
git diff --check
```

- [ ] **Step 2: Commit**

```powershell
git add docs\superpowers\plans\2026-05-19-edge-audio-river.md scripts\truevision_edge_audio_river.py tests\test_truevision_edge_audio_river.py
git commit -m "Add Edge audio river renderer"
```

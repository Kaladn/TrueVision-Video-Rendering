# TrueVision Video Rendering

TrueVision Video Rendering is an experimental local system for recording visual state, regenerating state-derived video, and testing deterministic frame reconstruction.

The project is source-first. Generated media, capture chunks, manifests, reports, and run artifacts are intentionally excluded from the repository.

## Core Boundary

```text
Forward TrueVision records observed audio/video state.
Reverse TrueVision replays, regenerates, or demonstrates state.
Generated media is synthetic state media, not evidence.
Raw frames are not implied unless explicitly enabled.
```

## Rendering Law

TrueVision render lanes must separate state work from final pixels.

```text
state/grid/primitive plan first
pixel output last
```

The rule:

```text
Do not solve production video by asking every output pixel what it should be.
Build deterministic state first:
  audio/video signals
  cell/grid fields
  temporal transition fields
  render primitives
  confidence/receipt data

Then rasterize or encode the final pixels.
```

Pixel-level inspection is allowed for validation, replay accuracy, and final
presentation. Pixel-first full-frame procedural rendering is a prototype path
only unless it is explicitly marked as such.

The current high-performance render lane follows this machine-use law:

```text
CPU: use available render threads for state/render work.
GPU: use hardware video encoding when available.
Manifest: record encoder, render threads, bitrate, frame count, duration,
state-log interval, wall time, memory, and hashes.
```

Plain version:

```text
TrueVision should think in state, not paint every pixel by hand.
The machine can still output pixels, but pixels are the final delivery format,
not the main reasoning system.
```

## Scientific Summary

TrueVision represents video as time-ordered visual state rather than as a prompt-only media product. A capture maps observed frames into addressed cell tensors containing channels such as color, luminance, edge density, texture energy, motion energy, and temporal deltas. TrueFrameGen then reconstructs higher-rate video by estimating transition fields between known states.

The current compiled frame-generation lane uses a SegmentField approach:

```text
source state A
source state B
-> estimate one A-to-B transition field
-> render all intermediate frames as continuous steps along that field
```

This differs from independent frame blending. The grid is an internal state lattice; final presentation is expected to be a continuous image forecast.

## Plain English Summary

This project watches video and saves a compact description of what changed over time. Later, it can rebuild or smooth the video from that saved description.

Instead of asking an AI to invent every frame from scratch, TrueVision keeps track of motion, color, brightness, edges, and timing. TrueFrameGen uses that information to fill in the missing frames between real recorded frames.

## Current Capabilities

- Python research tools for capture, replay, templates, audio-reactive rendering, and local studio control.
- Rust native screen-state capture that writes compact `.tvcells` chunks.
- Rust streaming TrueFrameGen renderer for bounded-memory high-FPS reconstruction.
- AV-only tool bus for local audio/video template, render, and recalibration workflows.
- Local model adapter shape for prompt-to-state translation behind schema validation.
- TrueVision Studio control-plane tools for source snaps, existing-state animation, glow intensity animation, spectrum/city rendering, frame diff, manifest browsing, preset reuse, and local Qwen orchestration.
- Document-state reader for treating page frames and glyph cells as replayable visual state.
- Storage library support for keeping heavy runtime data outside the repository.

## Repository Layout

```text
docs/                         Technical notes and tool language
native/truevision_capture_rs/ Rust capture and streaming frame generation
scripts/                      Research CLIs and studio server
tests/                        Unit and integration tests
trueframegen/                 Python temporal reconstruction modules
truevision_runtime/           AV tools, document state, storage, renderer, and LLM adapter
ui/                           Local studio HTML
storage/                      Ignored runtime lanes with tracked placeholders
outputs/                      Ignored generated output lane
```

## Studio Presets

Successful render lanes are preserved as reusable presets instead of one-off
scripts. Current built-in presets include:

```text
glitch_444_alive_poster
fade_away_memory_cathedral
house_remix_audio_city
storm_ember_city
mirror_maze_realism
edge_audio_river
```

## Runtime Data Policy

Heavy runtime data belongs outside git:

```text
captures
cell-state chunks
rendered videos
generated manifests
generated reports
temporary previews
```

The default external vault used during development is:

```text
E:\TruEVision Generation
```

Initialize a storage vault:

```powershell
python scripts\truevision_storage_library.py init --root "E:\TruEVision Generation"
```

## Native Capture

Build:

```powershell
cd native\truevision_capture_rs
cargo build --release
```

Example capture:

```powershell
.\target\release\truevision_capture_rs.exe `
  --duration 10 `
  --fps 10 `
  --resolution 2560x1440 `
  --grid 640x360 `
  --output-root "E:\TruEVision Generation\library\capture_units\10_second\runs" `
  --run-id "sample_capture"
```

## TrueFrameGen

Example 60 FPS reconstruction from captured state:

```powershell
.\native\truevision_capture_rs\target\release\trueframegen_stream_rs.exe `
  --run-dir "E:\TruEVision Generation\library\capture_units\10_second\runs\sample_capture" `
  --output-dir "E:\TruEVision Generation\library\trueframegen\sample_capture_60fps" `
  --duration 10 `
  --capture-fps 10 `
  --target-fps 60 `
  --motion-mode segment-field
```

## Full Pipeline

The operator path from human intent to rendered video is documented here:

[docs/HUMAN_TO_VIDEO_PIPELINE.md](docs/HUMAN_TO_VIDEO_PIPELINE.md)

The current lab status, proof ledger, rejected experiments, and TODO roundup are documented here:

[docs/BEAST_MODE_LAB_REPORT_2026-05-22.md](docs/BEAST_MODE_LAB_REPORT_2026-05-22.md)

## Local Studio

```powershell
python scripts\truevision_studio_server.py --storage-root "E:\TruEVision Generation"
```

Open:

```text
http://127.0.0.1:8765/
```

The studio is local-first. Model output is treated as a draft state request; only validated AV tool calls and templates are trusted.

## Development

Install Python dependencies:

```powershell
python -m pip install -e .
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Build Rust tools:

```powershell
cd native\truevision_capture_rs
cargo build --release
```

## Status

This is experimental research software. It is intended for local audio/video state-media rendering experiments, not forensic reconstruction, evidence generation, or production video tooling.

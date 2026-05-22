# TrueVision Generation Lab System Guide

This document explains what this repository is, why each major part exists,
and how the parts talk to each other.

## What This Repo Is

TrueVision Generation Lab is a local audio/video state-media research system.
It studies three related operations:

```text
record observed media as structured state
regenerate or smooth media from stored state
generate new synthetic media from validated state templates
```

The project is not a cloud video generator, not a general AI operating system,
not a desktop automation tool, and not forensic evidence software.

The central rule is:

```text
Record state.
Plan state.
Transform state.
Render pixels last.
Prove every run.
```

## Why It Exists

The goal is to make audio/video generation and reconstruction inspectable.

Instead of treating output as a magic prompt result, the repo keeps explicit
state records:

```text
audio features
cell/grid fields
temporal transition fields
render primitives
tool receipts
manifests
frame-state logs
```

That gives the operator a way to ask:

```text
What drove this output?
Which input was used?
Which tool ran?
Which renderer produced the video?
How long did it take?
What machine resources were used?
What was synthetic and what was observed?
```

## Core Boundary

```text
Forward TrueVision records observed audio/video state.
Reverse TrueVision replays, regenerates, or demonstrates state.
Generated media is synthetic state media, not evidence.
Raw frames are not implied unless explicitly enabled.
```

Generated videos are creative/render artifacts. They must not be described as
proof of real events.

## Major Subsystems

### 1. Native Capture And Render

Path:

```text
native/truevision_capture_rs/
```

Purpose:

```text
high-speed capture
compiled TrueFrameGen playback
full-song state-media rendering
hardware-assisted video encoding
```

Important binaries:

```text
truevision_capture_rs
trueframegen_stream_rs
truevision_weird_occlusion_rs
```

Why it exists:

Python was too slow for sustained capture and full-length render loops. Rust
owns the hot path so the machine can use many CPU threads and hardware encoders.

Current rule:

```text
Python may orchestrate.
Rust renders/captures.
FFmpeg encodes/muxes.
```

### 2. TrueFrameGen

Paths:

```text
trueframegen/
scripts/trueframegen_*.py
native/truevision_capture_rs/src/bin/trueframegen_stream_rs.rs
```

Purpose:

```text
reconstruct higher-FPS playback from known captured state
test 6-1-6 temporal causality
test SegmentField motion continuity
```

The intended correction is SegmentField:

```text
old wrong shape:
  each generated frame solves alone

current intended shape:
  one A-to-B transition field
  many generated frames walk the same bridge
```

Why it exists:

Frame generation should not smear or snap by treating every frame as an isolated
blend. It should build a transition plan between known states.

### 3. AV Tool Bus

Path:

```text
truevision_runtime/av_tools/
```

Files:

```text
av_tool_registry.py
av_tool_policy.py
av_tool_runner.py
av_tool_receipts.py
av_recalibration.py
```

Purpose:

```text
define allowed audio/video tools
reject out-of-scope requests
execute validated calls
write receipts for every write/action
store recalibration notes
```

Why it exists:

The local model can help plan, but it must not directly operate the system.
The trusted boundary is validated AV tool calls, not free-form model text.

### 4. LLM Adapter

Path:

```text
truevision_runtime/llm_adapter/
```

Purpose:

```text
wrap a local or remote model endpoint
add project rules and schema
ask for draft JSON state
validate or repair the draft
hand only valid state to the tool/render layer
```

Who it talks to:

```text
Studio server
-> LLM adapter
-> local Qwen / Ollama / compatible endpoint
-> schema validator
-> AV tool bus
```

Why it exists:

Qwen does not need to know TrueVision internally. The wrapper teaches it the job
at runtime and prevents unvalidated output from driving the renderer.

### 5. Studio Server And UI

Paths:

```text
scripts/truevision_studio_server.py
ui/truevision_state_media_studio.html
truevision_runtime/studio/
```

Purpose:

```text
local control surface
daily chat storage
template save/load/delete
render preset library
local Qwen proxy
AV tool call endpoint
record/render planning
```

Current truth:

The server and endpoints exist and are tested. The UI is not yet product-grade.
It is a working control plane prototype, not a finished studio application.

### 6. Template Renderer

Path:

```text
truevision_runtime/rendering/template_renderer.py
```

Purpose:

```text
render reusable JSON templates
extract audio features
write MP4, manifest, frame-state JSONL, report, thumbnail
```

Why it exists:

It is the Python research renderer for reusable template behavior. It is useful
for tests, previews, and design work. Production-length heavy render loops
should move to Rust/native lanes.

### 7. State Patterns

Path:

```text
truevision_runtime/state_patterns/
```

Purpose:

```text
small library of known audio/video state patterns
prompt-to-state context
reusable visual behavior vocabulary
```

Why it exists:

The AI should choose from known state patterns instead of inventing blind.

### 8. Document-State Reader

Path:

```text
truevision_runtime/document_state/
```

Purpose:

```text
treat pages as visual state
build page-frame records
match glyph state through a read-only lexicon
keep lifetime counts
produce deterministic document-state reads
```

Why it exists:

This creates a bridge toward visual-symbolic document/video reading without
claiming OCR text as absolute authority. The current reader is a controlled
state-reader lane, not full natural-page recognition at scale.

### 9. Research Scripts

Path:

```text
scripts/
```

Purpose:

```text
capture prototypes
still-image state snaps
path tracing experiments
audio-reactive river/city renderers
signature extraction
state replay
storage initialization
studio server
```

Rule:

Scripts may exist as experiments, but any successful lane should be promoted
into:

```text
runtime module
Studio preset
tool contract
template
documented command
test
```

### 10. Storage And Outputs

Paths:

```text
storage/
outputs/
E:\TruEVision Generation
```

Purpose:

```text
runtime chats
templates
events
receipts
artifacts
manifests
reports
generated videos
capture chunks
```

Policy:

Generated media and capture data are not committed to git. The repository keeps
system code, tests, docs, and small reusable samples.

## Who Talks To Who

### Human To Video

```text
Human direction/audio/lyrics/source notes
-> Studio UI or CLI
-> local Qwen/Codex draft
-> PromptToStateAdapter
-> schema validator
-> AV policy
-> template/preset/tool request
-> Rust or Python renderer
-> FFmpeg encoder/muxer
-> MP4 + manifest + frame-state JSONL + report
```

### Capture To Replay

```text
screen/video/still source
-> recorder/capture lane
-> cell-state chunks or NPZ
-> manifest/summary
-> replay or TrueFrameGen
-> reconstructed video + continuity report
```

### Recalibration

```text
human watches output
-> time marker note
-> recalibration record
-> template patch or variant
-> preview/full render
-> receipt + manifest
```

### Cross-System Harness

The broader system boundary is:

```text
AnchorWorks = face and language/count brain
SecureCore = safety, agents, logging, policy, system substrate
TrueVision Generation = state-media render lab
Harness = proof that they can cooperate without becoming one tangled thing
Connector point = validated state packets
```

TrueVision Generation should enter the wider system only as an optional
state-media lane.

## Primary File Map

```text
README.md
  public overview, core boundary, run commands

TODO.md
  active roadmap and cross-system pickup order

docs/HUMAN_TO_VIDEO_PIPELINE.md
  operator path from prompt/audio to video output

docs/BEAST_MODE_LAB_REPORT_2026-05-22.md
  proof ledger, rejected experiments, current limits

docs/TRUEVISION_AV_TOOLING_LIBRARY.md
  AV tool list, render law, runtime inventory

docs/TRUEVISION_NATIVE_POWER_POLICY.md
  what Python may do and what must be native

docs/DOCUMENT_STATE_READER.md
  document/page/glyph state-reader lane

native/truevision_capture_rs/src/bin/truevision_capture_rs.rs
  native screen-state capture

native/truevision_capture_rs/src/bin/trueframegen_stream_rs.rs
  compiled frame reconstruction / high-FPS playback

native/truevision_capture_rs/src/bin/truevision_weird_occlusion_rs.rs
  compiled full-song visual render lanes and presets

truevision_runtime/av_tools/
  validated AV tool bus

truevision_runtime/studio/studio_tooling.py
  reusable Studio tool contracts and render presets

truevision_runtime/llm_adapter/
  prompt-to-state contract and validator

truevision_runtime/rendering/template_renderer.py
  Python template renderer

trueframegen/
  Python temporal reconstruction modules

scripts/
  research CLIs and bridge commands

tests/
  unit and integration tests
```

## Current Proven Runs

The strongest current proof is the native full-song lane:

```text
scene: memory_cathedral
duration: 232.88s
frames: 6986
state records: 6986
resolution: 1280x720
fps: 30
encoder: h264_qsv
render threads: 32
speed: about 1.5x realtime
```

Earlier full-song proof:

```text
scene: warp_laser_field
duration: 191.44s
frames: 5743
state records: 5743
encoder: h264_qsv
render threads: 32
speed: about 1.216x realtime
```

Outputs remain ignored runtime artifacts.

## Known Limits

```text
UI is not product-grade yet.
True FFT/Goertzel spectrum analysis is still open.
GPU raster/generation is not implemented yet.
Python render paths are research/previews, not the production full-render shape.
Realistic from-scratch video is still research.
Natural page glyph recognition at scale is not proven in this repo.
Generated media is synthetic and not evidence.
```

## Development Checks

Python tests:

```powershell
$env:PYTHONPATH='scripts;modules;.'
python -m unittest discover -s tests -v
```

Rust build:

```powershell
cargo build --release --manifest-path native/truevision_capture_rs/Cargo.toml --bin truevision_weird_occlusion_rs --bin trueframegen_stream_rs
```

Generated data check before commit:

```powershell
git diff --cached --name-only |
  Select-String -Pattern '^(outputs|storage|\.vscode)/|\.mp4$|\.wav$|\.png$|\.jpg$|\.jpeg$|\.npz$|\.tvcells$'
```

## Plain English

TrueVision Generation Lab is a workshop for making and rebuilding video from
state instead of from mystery prompts. It listens to audio, records or plans
visual changes, renders video locally, and leaves a paper trail showing what
happened.

The point is not to pretend the computer sees like a person. The point is to
let the computer use its own strengths: counting, state tracking, deterministic
rules, fast local rendering, and repeatable manifests.

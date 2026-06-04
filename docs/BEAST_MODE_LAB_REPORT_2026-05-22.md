# TrueVision Beast Mode Lab Report

Date: 2026-05-22

## Bottom Line

TrueVision Video Rendering is no longer just a loose pile of experiments. The repo now contains a working local audio/video state-media lab with:

```text
native capture/state lanes
TrueFrameGen reconstruction lanes
AV-only tool policy and receipts
prompt-to-state adapter shape
local Studio surface
reusable render presets
document/page visual-state reader
hardware-assisted render proof
```

The hard correction from the last session is now law:

```text
state/grid/primitive plan first
pixel output last
```

We proved that the machine can render a full song faster than realtime when the lane uses 32 CPU render threads and Intel QSV hardware encoding. We also proved that pixel-first CPU-only rendering is the wrong production architecture.

## Current Repo Reality

This worktree is not clean. There are modified and untracked files that represent active development. Do not treat the current branch as release-ready until it is reviewed, committed, and pushed intentionally.

Current active areas:

```text
README.md
TODO.md
docs/
native/truevision_capture_rs/
scripts/
tests/
trueframegen/
truevision_runtime/
ui/
```

Important active docs:

```text
README.md
TODO.md
docs/TRUEVISION_AV_TOOLING_LIBRARY.md
docs/TECHNICAL_OVERVIEW.md
docs/PLAIN_ENGLISH_OVERVIEW.md
docs/TRUEVISION_NATIVE_POWER_POLICY.md
docs/TRUEVISION_STATE_GENERATION_PRIMITIVES.md
docs/DOCUMENT_STATE_READER.md
```

Important active native paths:

```text
native/truevision_capture_rs/src/bin/truevision_capture_rs.rs
native/truevision_capture_rs/src/bin/trueframegen_stream_rs.rs
native/truevision_capture_rs/src/bin/truevision_weird_occlusion_rs.rs
```

Important active runtime paths:

```text
truevision_runtime/av_tools/
truevision_runtime/llm_adapter/
truevision_runtime/rendering/
truevision_runtime/storage_library.py
truevision_runtime/studio/
truevision_runtime/document_state/
```

## Verification Snapshot

Current verification run:

```text
cargo build --release --manifest-path native/truevision_capture_rs/Cargo.toml --bin truevision_weird_occlusion_rs --bin trueframegen_stream_rs
```

Result:

```text
passed
warning only: unused spectrum_backdrop_haze in truevision_weird_occlusion_rs.rs
```

Current Python test run:

```text
$env:PYTHONPATH='scripts;modules;.'; python -m unittest discover -s tests -v
```

Result:

```text
129 tests passed
0 failed
```

## Proven

### 1. Core Boundary

Proven in docs and tests:

```text
Forward TrueVision records observed audio/video state.
Reverse TrueVision replays, regenerates, or demonstrates state.
Generated media is synthetic state media, not evidence.
Raw frames are not implied unless explicitly enabled.
```

This boundary is present in the README, tooling docs, prompt-to-state adapter rules, and render reports.

### 2. Native Capture Shape

Proven by docs/tests:

```text
Rust native capture emits .tvcells state chunks.
Capture hot loop stays dumb and fast.
Mapping/fingerprinting/reconstruction are separate post-capture workers.
Capture manifests state that raw frames are not saved by default.
```

This is the correct split:

```text
capture frame
-> extract compact cell features
-> write state chunk
-> return fast

later:
state chunks
-> temporal mapping / signatures / reconstruction
```

### 3. TrueFrameGen Exists

Proven by tests and Rust source:

```text
Python 6-1-6 temporal map exists.
Python causal cell map exists.
Python frame gap filler exists.
Rust streaming TrueFrameGen renderer exists.
SegmentField mode exists.
Bounded-cache streaming exists.
```

The important correction is SegmentField:

```text
old wrong shape:
each generated frame solves independently

correct shape:
one A-to-B transition field
many generated frames walk that same field
```

### 4. Render Law Is Now Documented

Proven in:

```text
README.md
docs/TRUEVISION_AV_TOOLING_LIBRARY.md
TODO.md
```

The law:

```text
Grid/state storage is not the same thing as pixel presentation.
Production render lanes must be state/grid/primitive first.
Pixels are final output, not the primary reasoning layer.
```

### 5. Native Power Policy Exists

Proven in:

```text
docs/TRUEVISION_NATIVE_POWER_POLICY.md
```

The policy:

```text
Python may orchestrate, validate, and report.
Python must not own sustained capture loops.
Python must not own full-length frame generation.
Python must not own high-resolution pixel transforms.
Heavy work belongs in Rust/native/GPU lanes.
```

### 6. Beast Mode Full-Song Render

Proven by completed output, manifest, per-frame state log, hashes, and report:

```text
Run ID:
glitch_444_house_center_warp_laserfield_full_beast_qsv_rs

Output:
outputs/weird_occlusion_rs/glitch_444_house_center_warp_laserfield_full_beast_qsv_rs/
```

Render proof:

```text
Duration: 191.44 seconds
Frames: 5743
Frame-state records: 5743
Resolution: 1280x720
FPS: 30
CPU render threads: 32
GPU encoder: ffmpeg h264_qsv
Bitrate: 24M
Wall time: 157.380 seconds
Speed vs realtime: 1.216x
Peak working set: 85.60 MiB
Output size: 549.59 MiB
```

This proved:

```text
Full-song generation can run faster than realtime on this machine.
Intel QSV hardware encoding works.
32-thread Rust rendering works.
Per-frame deterministic state logging works.
```

### 7. AV Tool Bus And Studio Tooling

Proven by tests:

```text
AV-only tool registry exists.
AV policy rejects unknown tools and path escapes.
Receipts are written for tool calls.
Studio-level tools exist.
Render preset library exists.
Local model controller has been removed.
```

Current Studio tool contracts:

```text
source_snap_tool
existing_state_animator
electric_glow_intensity_animator
spectrum_audio_reactive_city
frame_diff_replay_accuracy
manifest_browser
render_preset_library
```

### 8. External Prompt-To-State Adapter Shape

Future adapter boundary:

```text
Model output is draft JSON only.
Validator is the trust boundary.
Bad drafts are repaired with validation errors only.
The model does not execute directly.
```

This repo uses approved external API sessions for model assistance and validated
state/tool calls.

### 9. Document-State Reader

Proven by tests:

```text
Document/video-style page frames can be represented as visual state.
Reader uses read-only lexicon and lifetime counts.
Repeated visual state is deterministic.
```

This is a serious future lane:

```text
document page
-> visual/glyph state
-> recall/replay/graph/tutorial generation
```

### 10. Some Visual Presets Are Real

Current preset truth from `truevision_runtime/studio/studio_tooling.py`:

```text
glitch_444_alive_poster: proven
fade_away_memory_cathedral: ready
house_remix_audio_city: ready
abstract_symphony_soft_beams: needs_rework
center_warp_laserfield: draft
storm_ember_city: proven
mirror_maze_realism: proven
edge_audio_river: proven
```

The important distinction:

```text
proven = has a working lane/preset and tests or successful artifacts
ready = coherent preset, not yet fully promoted as a core lane
draft = working proof but not yet architecturally final
needs_rework = rejected visual direction or bad product fit
```

## Disproven Or Rejected

### 1. Pixel-First CPU Full Render

Disproven by the aborted full run:

```text
CPU-only, pixel-first full-frame procedural render was too slow.
It ran for roughly 35 minutes and still had no final manifest.
```

Conclusion:

```text
Do not use full-frame pixel-question loops as the production shape.
They are prototype/preview only unless explicitly marked.
```

### 2. Abstract Fog Symphony Direction

Rejected by user review:

```text
abstract_symphony_soft_beams
status: needs_rework
```

Reason:

```text
Too foggy, low contrast, not the desired visual language.
```

It remains useful only as a failed study.

### 3. UI As Product

Not proven.

Current truth:

```text
HTML exists.
Server exists.
Tests show many endpoints and controls exist.
But user experience was judged poor and parked.
```

Conclusion:

```text
The backend/tooling is more real than the UI.
Do not present the UI as finished.
```

### 4. True Spectrum Analysis

Not proven yet.

Current truth:

```text
Audio levels and waveform exist.
Spectrum bars are still low/mid/high facsimile in some lanes.
True FFT or Goertzel bins are still open TODO.
```

### 5. Realism From Scratch

Not solved.

Current truth:

```text
We can make strong procedural visuals.
We can animate existing poster intensity.
We can create stylized smoke/fog/rivers/lasers.
We have not proven high-realism generated video from scratch.
```

### 6. YOLO / Semantic CV Route

Rejected for this project stage.

Current truth:

```text
The chosen path is state detection, not YOLO/object detection.
YOLO-style semantic CV was judged too slow and off-mission for the hot path.
```

### 7. Python Full-Length Rendering

Rejected by power policy.

Current truth:

```text
Python scripts exist and tests pass.
Python is useful for research, wrappers, manifests, reports, and tests.
Python must not own full-length render loops.
```

## Current TODO Roundup

The repo TODO currently has 26 checked items and 28 open items.

### Repository Discipline

Done:

```text
Keep generated media and run artifacts out of git.
Keep runtime storage directories as placeholders only.
Keep the project scoped to audio/video state media.
```

Open:

```text
Add CONTRIBUTING.md.
Add a license after ownership and release intent are decided.
Add CI for Python tests and Rust build.
```

### Native Capture

Done:

```text
Rust native capture emits .tvcells state chunks.
Capture loop keeps mapping/fingerprinting out of the hot path.
Capture manifests state that raw frames are not saved by default.
```

Open:

```text
Add a stop/abort control for long captures.
Add selected-window capture.
Add GPU capture research branch only after CPU/native capture is stable.
```

### TrueFrameGen

Done:

```text
Python 6-1-6 temporal map exists.
Rust streaming renderer exists.
SegmentField mode exists: one A-to-B transition field, many generated frames.
```

Open:

```text
Promote SegmentField as the default proof path after clean source validation.
Add stronger cell-boundary deblocking for final presentation.
Add transition confidence maps for review.
Add long-run chunked rendering.
Validate libx264, h264_qsv, hevc_qsv, and h264_amf output quality.
```

### Rendering Language

Done:

```text
Template renderer exists.
Audio feature extraction tools exist.
Initial state-pattern library exists.
Document-state reader exists for page-frame and glyph-cell recall.
Render Law is documented: state/grid/primitive first, pixels last.
```

Open:

```text
Formalize the AV state template schema.
Add material channels for fog, smoke, water, glass, glow, and lighting pressure.
Add camera-motion primitives.
Add time-marker recalibration patches.
Connect document-state packets to shape/language rules for tutorials, charts, and graph generation.
```

### Local Studio

Done:

```text
Local HTML studio exists.
Local server exists.
AV-only tool registry and policy layer exist.
Local model adapter shape exists.
Reusable Studio tool contracts exist.
Proven render lanes are represented as reusable presets.
```

Open:

```text
Remove any remaining placeholder UI language.
Add native capture controls to the studio server.
Add template comparison view.
Add render status polling.
Add true FFT/Goertzel frequency-bin analyzer bars.
Wire Studio preset launch into the Rust renderer with preview/full render job execution receipts.
```

### Documentation

Done:

```text
Public README explains the project boundary.
Tool inventory exists.
State generation primitive notes exist.
```

Open:

```text
Add architecture diagram.
Add capture format specification for .tvcells.
Add TrueFrameGen algorithm notes with pseudocode.
Add plain-language walkthrough.
```

### Validation

Done:

```text
Unit tests cover core Python modules.
Rust build has been proven locally.
Full-song QSV/32-thread render produced per-frame deterministic state records.
```

Open:

```text
Add small fixture capture for reproducible tests.
Add benchmark command for capture FPS, frame time, CPU, and RAM.
Add generated-video quality metrics that do not require external services.
```

## What We Want To Accomplish

### Near Term

```text
1. Clean and commit the current work intentionally.
2. Promote Render Law into code paths, not just docs.
3. Add a benchmark command that records FPS, frame time, CPU, RAM, encoder, and output size.
4. Make Studio launch Rust preview/full renders with receipts.
5. Replace facsimile spectrum with true FFT or Goertzel frequency bands.
6. Add stop/abort control for long capture/render jobs.
```

### Core Engineering Direction

```text
1. Capture and state extraction stay native.
2. TrueFrameGen uses SegmentField as the default reconstruction path.
3. Renderers build state/grid/primitive plans before final rasterization.
4. GPU is used first for encoding, then later for actual shader/raster lanes.
5. Python remains orchestration, validation, reporting, tests, and external API glue.
```

### Product Direction

```text
TrueVision Studio should become an audio/video state-media lab:
  user talks through an approved external API session
  API response drafts AV state
  validator checks it
  tool bus executes allowlisted AV tools
  Rust/native lanes render/capture
  manifests and receipts prove what happened
```

### Research Direction

```text
1. Better use of captured video signatures.
2. Realistic fog/smoke/glow as state fields, not circles or triangles.
3. Document/video reader for reproducible page and glyph state.
4. 6-1-6 and SegmentField confidence maps.
5. Deterministic projection from source capture without copying raw pixels as the main truth.
```

## Final Honest State

This repo has a real technical core now:

```text
capture state
replay state
fill frames
stream reconstruction
validate tool calls
render full-song hardware-assisted output
prove runs with manifests, hashes, and state logs
```

But it is not yet a finished product:

```text
UI needs product work.
Render schema needs formalization.
True spectrum analysis is unfinished.
Realistic generation is still research.
GPU raster/generation is not implemented yet.
Current branch needs cleanup and commit discipline.
```

The strongest thing we learned:

```text
The idea works best when TrueVision behaves like a state machine, not like a painter.
```

The law going forward:

```text
Record state.
Plan state.
Transform state.
Render pixels last.
Prove every run.
```

## Weekend Cross-System Pickup

Before this lab is connected back to the broader systems, the pickup order is:

```text
1. Open the other workspace in SecureCore.
2. Inventory what is actually on the workbench.
3. Lock the cross-system harness shape.
4. Prove AnchorWorks / SecureCore / TrueVision health checks.
5. Only then bring TrueVision Generation in as an optional state-media lane.
```

Current handoff law:

```text
AnchorWorks = face and language/count brain
SecureCore = safety, agents, logging, policy, system substrate
TrueVision Generation = state-media render lab
Harness = proof that they can cooperate without becoming one tangled thing
Connector point = validated state packets
```

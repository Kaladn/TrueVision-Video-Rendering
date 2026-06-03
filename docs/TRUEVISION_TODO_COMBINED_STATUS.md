# TrueVision Combined TODO And Status Audit

Date: 2026-06-02

## Scope

This report combines and verifies TODO/status material from TrueVision-owned directories only.

Scanned active/code roots:

```text
D:\TrueVision_Generation_Lab
D:\CortexEvolved\systems\TrueVision
```

Scanned runtime/archive roots:

```text
E:\TruEVision Generation
E:\TruEVision_Generation_artifacts
```

Not scanned as general scope:

```text
D:\CortexEvolved root-level docs/code outside systems\TrueVision
```

## Source TODO Files

| Source | Status | Done | Open | Notes |
|---|---:|---:|---:|---|
| `D:\TrueVision_Generation_Lab\TODO.md` | active standalone roadmap | 85 | 83 | Main standalone TrueVision TODO. |
| `D:\CortexEvolved\systems\TrueVision\TODO.md` | embedded TrueVision roadmap | 85 | 91 | Same base TODO plus newer Effect Calibration and Dev Chat items. |
| `D:\TrueVision_Generation_Lab\docs\VIDEO_ITEMS_TODO.md` | parked generated-video note | 0 | 0 | Parks video output lanes; no checkbox work. |
| `E:\TruEVision Generation\...\TRUEVISION_STATE_MEDIA_POC\TODO.md` | historical POC archive | 0 | 74 | Archive TODO, not active source roadmap. |

## Verification Run

Active standalone lab:

```text
cd D:\TrueVision_Generation_Lab
$env:PYTHONPATH='D:\TrueVision_Generation_Lab\scripts;D:\TrueVision_Generation_Lab\modules;D:\TrueVision_Generation_Lab'
python -m unittest discover -s tests -v

Result: 336 tests OK
```

Embedded TrueVision root:

```text
cd D:\CortexEvolved\systems\TrueVision
$env:PYTHONPATH='D:\CortexEvolved\systems\TrueVision\scripts;D:\CortexEvolved\systems\TrueVision\modules;D:\CortexEvolved\systems\TrueVision'
python -m unittest discover -s tests -v

Result: 270 tests OK
```

Native Rust:

```text
cd D:\TrueVision_Generation_Lab\native\truevision_capture_rs
cargo build

Result: OK
```

```text
cd D:\CortexEvolved\systems\TrueVision\native\truevision_capture_rs
cargo build

Result: OK
```

Tool catalog status from standalone lab:

```text
active_callable: 41
contract_reference: 3
finalized_copy_only: 1
parked_experimental: 4
```

## Combined Status By Lane

### Repository Discipline

Status: started, mostly complete.

Complete or verified:

- Generated media and runtime artifacts are intended to stay out of git.
- Local product map, active tool surface, parked experiment map, AW/SC boundary contracts, and preflight script exist.
- Tests pass in both active TrueVision roots when launched with the required script/module path.

Still open:

- `CONTRIBUTING.md`
- license
- CI for Python tests and Rust build

### Native Capture

Status: started and test-backed.

Complete or verified:

- Rust native capture project exists and builds in both roots.
- `.tvcells` state-chunk capture lane is represented.
- Capture manifests claim no raw-frame retention by default.

Still open:

- stop/abort control for long captures
- selected-window capture
- GPU capture research branch

### TrueFrameGen

Status: started and test-backed.

Complete or verified:

- Python 6-1-6 temporal map exists.
- Rust streaming renderer exists.
- SegmentField mode exists.

Still open:

- promote SegmentField as default proof path
- stronger cell-boundary deblocking
- transition confidence maps
- long-run chunked rendering
- encoder quality validation for `libx264`, `h264_qsv`, `hevc_qsv`, and `h264_amf`

### Rendering / State Language

Status: started and test-backed.

Complete or verified:

- Template renderer, audio feature tools, initial state-pattern library, document-state reader, state-loop law, and weather/material starter channels exist.
- State language tests pass in the standalone lab.

Still open:

- formal AV state template schema
- remaining material channels for smoke, water, glow, and lighting pressure
- camera-motion primitives
- time-marker recalibration patches
- document-state packets connected to shape/language rules for tutorials, charts, and graph generation

### Atmosphere / Weather State Lane

Status: started and test-backed.

Complete or verified:

- Fog, mist, cloud, and rain-glass contracts exist.
- State channels for density, veil, scatter, edge softness, motion, curl, occlusion, droplet, refraction, and wetness exist.
- Native capture profiling with 6-1-6 windows exists.
- `atmosphere_toolset_create` and `atmosphere_profile_from_capture` exist.

Still open:

- run full fog/mist teacher capture and review density windows
- renderer hooks that consume profiles without hardcoded script behavior
- rain-on-glass reference capture
- cloud-volume reference capture

### Effect Calibration By Observed Diff

Status: plan-only in current active roots.

Evidence:

- The effect-calibration plan exists in `D:\CortexEvolved\systems\TrueVision\TODO.md`.
- Expected implementation files are not present:
  - `truevision_runtime\learning_intake\effect_calibration.py`
  - `tests\test_effect_calibration.py`
  - `scripts\truevision_effect_calibration.py`

Next work:

- Build observed-state, generated-state, diff, adjustment, and receipt schemas.
- Keep generated media marked derived-only and never evidence.

### Elemental Learning Intake

Status: started and heavily test-backed.

Complete or verified:

- Source-surface safe ops, coordinate capture plans, multi-sample planning, URL canonicalization, verified video-state receipts, coordinate map requirements, and purge-after-profile receipts are represented and tested.
- `element_creation_profile_from_capture` exists.
- Three-source process tests pass.
- Open-license dataset policy exists.

Still open:

- smoke curl/dissipation contracts
- element intake queue JSONL format
- source candidate records
- capture-plan builder without free desktop/browser control
- 42s smoke source plan
- real 3-video process run
- profile quality scoring
- profile comparison across fog/smoke/mist/clouds
- renderer-profile binding
- retention closeout policy execution
- DeepAction queue/cache/receipt/tiny proof

### Terrain Realism Teacher

Status: started and test-backed.

Complete or verified:

- Terrain plan, workspace contract, source classes, candidate ranking, disk guard, cleanup receipt, and human-review packet tests pass.

Still open:

- initialize external terrain workspace
- process first ocean-cliff source
- render 12-second wide-edge proof only after first promoted rule
- add QA comparison against current wide-edge proof
- future raytracing/pathtracing logger contracts

### TrueAudio Pre-Output Lane

Status: started, mostly complete, test-backed.

Complete or verified:

- TrueAudio is defined as sibling audio-state system.
- File and machine pre-sound logging tools exist.
- Replayable spectral-state capture and replay exist.
- Manifests and receipts include source hash, decoder path, frame count, duration, and no-raw-audio boundary.
- TrueSpeech segment detection and candidate lyric alignment exist.

Still open:

- run live machine-output capture with user-started playback and review receipt
- promote FFmpeg discovery helper into canonical resolver
- add TrueAudio-to-TrueVision sync contract

### Voice State Lane

Status: started, not complete.

Complete or verified:

- Speech/background segment detection exists.
- Source-file replayable state logging exists.
- Candidate lyric timing exists.

Still open:

- deterministic `voice_state_v1`
- `voice_extract_timeline`
- `voice_align_script`
- vocal-isolation / voice-vs-music calibration
- phoneme/word candidate lane
- editable voice timing JSON
- voice channel list
- voice manifest fields
- `voice_mix_bed` plan

### Cleanup Before State Vocal Services

Status: plan-only.

Open items:

- choose one canonical audio feature contract
- mark Rust `vocal_presence()` as heuristic
- normalize FFmpeg discovery
- formalize voice artifact storage
- keep voice lane audio/video only
- use `voice_state` naming unless service is intentional
- preserve FFmpeg -> TrueAudio -> TrueVision design

### Local Studio

Status: started and test-backed, but still not final product.

Complete or verified:

- Local HTML studio and server exist.
- AV-only tool registry and policy layer exist.
- Local engine adapter shape exists.
- Reusable Studio tool contracts and render presets exist.
- Studio server tests pass in active roots.

Still open:

- remove placeholder UI language
- native capture controls in studio server
- template comparison view
- render status polling
- true FFT/Goertzel analyzer bars
- Rust preset launch wiring with receipts
- voice/narration timing view
- line timing editor
- voiceover preview lane

### Documentation

Status: started.

Complete or verified:

- README, tool inventory, state generation primitive notes, repo system guide, and third-party notices exist.

Still open:

- architecture diagram
- `.tvcells` capture format specification
- TrueFrameGen algorithm notes with pseudocode
- plain-language walkthrough

### Validation

Status: started and strong, but not finished.

Complete or verified:

- Python tests pass when launched with proper `PYTHONPATH`.
- Rust builds pass.
- Timing audit, TrueAudio, TrueSpeech, AV tool bus, state-source law, tool-drop, and harness tests pass.

Still open:

- small fixture capture
- benchmark command for capture FPS/frame time/CPU/RAM
- generated-video quality metrics without external services
- tiny voice WAV fixture
- voice extraction determinism tests
- script-line timing save/load test
- voice manifest test

### Cross-System Harness Pickup

Status: plan-only in the TODOs.

Open items:

- open SecureCore workspace first
- inventory SC/AW/TV workbench
- lock cross-system harness shape
- prove AW, SC, and TV health checks separately
- bring TrueVision Generation in only as optional state-media lane

### Dev Chat / Language-State Surface

Status: started only in embedded TrueVision/Cortex work, not present in standalone lab.

Evidence:

- `D:\CortexEvolved\systems\TrueVision\truevision_runtime\dev_chat\chat.py` exists.
- Standalone lab does not currently have `truevision_runtime\dev_chat\chat.py`.
- Browser fetch issue is parked in `D:\CortexEvolved\systems\TrueVision\TODO.md`.

Still open:

- debug `truevision_dev_chat.html` fetch path
- confirm same-origin static UI and POST route
- inspect browser Network output
- add tiny UI fetch smoke test
- prove typed question -> contained renderer -> JSONL/receipt write through browser path

### Video Items

Status: parked.

Parked lanes:

```text
distant_love_state_soft_rs
stem_state_nightmare
stem_state_nightmare_rs
weird_occlusion_rs
```

Boundary:

```text
Generated media is visualization.
Video generation is parked until explicitly resumed.
```

### Historical State Media POC

Status: archived POC, not active roadmap.

Complete in archive:

- 30s capture
- 90x160x16 cell-state recording
- NPZ chunks
- JSONL temporal records
- manifest/summary/replay reports
- deterministic replay from stored cell state
- synthetic scene formula using same state shape
- first full-power frame consuming non-RGB channels
- still image to video-shaped state adapter
- handoff bundles

Open archive phases:

- freeze/verify bundle identity
- language cleanup
- real capture study
- geometry sidecar
- replay quality upgrades

Archive warning:

```text
Do not use this POC TODO as active source-roadmap authority unless explicitly resumed.
```

## Current Read

TrueVision is not just plan text. The active lab has a large amount of working, test-backed machinery:

```text
state capture
state replay
document-state movie
state recognition
geometry/shape profiles
meter grid
atmosphere profiling
TrueAudio/TrueSpeech state
studio/tool bus
tool_drop and harness
native Rust render/capture lanes
```

The main unfinished areas are not "does anything exist?" They are:

```text
effect calibration loop
voice-state formalization
profile-to-render binding
native/studio ergonomics
fixture/benchmark/quality validation
cross-system harness pickup
dev chat browser fetch
```

## Recommended Next Order

1. Fix dev-chat fetch only if the language-state chat is the current target.
2. Otherwise keep UI parked and finish state-tool language contracts in the standalone lab.
3. Build `effect_calibration.py` by TDD; it is the clearest plan-only gap.
4. Promote FFmpeg discovery into one canonical resolver before more audio/voice tools.
5. Add `.tvcells` capture format spec because it supports both state player and capture confidence.
6. Add small fixture capture and benchmark command so future claims stop depending on ad hoc runs.

## Law

```text
TODO checkboxes are not proof.
Files are not proof.
Passing tests plus receipts/manifests are proof.
Plans stay plans until code, tests, and receipts exist.
```

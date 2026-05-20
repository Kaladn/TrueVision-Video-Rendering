# TrueVision AV Tooling Library

This is the working inventory for the TrueVision Generation Lab audio/video
tooling. It separates actual tools from planned terminology and future research.

## Boundary

```text
Audio/video state media only.
No desktop automation.
No security enforcement.
No evidence claims for generated media.
```

Core law:

```text
Forward TrueVision records observed audio/video state.
Reverse TrueVision replays, regenerates, or demonstrates state.
Generated media is synthetic state media, not evidence.
```

## External Utilities Used

```text
python
ffmpeg
ffprobe
OpenCV / cv2 when present
numpy
Rust / cargo
Windows GDI native capture
PowerShell command runner
git
```

These are implementation utilities. They are not general autonomous tools.

## Studio And Runtime Tools

| Tool | Path | Status | Purpose |
| --- | --- | --- | --- |
| Studio server | `scripts/truevision_studio_server.py` | exists | Local UI/API bridge for chat, templates, storage, and AV tools. |
| Studio UI | `ui/truevision_state_media_studio.html` | exists | Single-page local workspace for prompt/chat/render control. |
| AV registry | `truevision_runtime/av_tools/av_tool_registry.py` | exists | Lists allowed audio/video tools. |
| AV policy | `truevision_runtime/av_tools/av_tool_policy.py` | exists | Validates allowed tool names and arguments. |
| AV runner | `truevision_runtime/av_tools/av_tool_runner.py` | exists | Executes validated AV-only tool calls. |
| AV receipts | `truevision_runtime/av_tools/av_tool_receipts.py` | exists | Logs tool execution receipts. |
| Recalibration | `truevision_runtime/av_tools/av_recalibration.py` | exists | Stores time-marker notes and patch proposals. |
| Storage library | `truevision_runtime/storage_library.py` | exists | Keeps repo storage and external vault layout tidy. |
| Template renderer | `truevision_runtime/rendering/template_renderer.py` | exists | Generic template-driven renderer. |

## AV Tool Bus Names

These are the current validated tool names in the runtime registry:

```text
audio_probe_duration
audio_analyze_levels
audio_extract_features
template_from_audio_signals
template_create
template_load
template_save
template_patch
template_create_variant
template_delete
time_marker_add
time_marker_list
recalibration_add_note
recalibration_apply
video_render_preview
video_prepare_full_render
video_execute_full_render
manifest_generate
receipt_create
learning_record_save
storage_list_artifacts
storage_list_templates
```

## Capture And Replay Tools

| Tool | Path | Status | Purpose |
| --- | --- | --- | --- |
| TrueVision recorder | `scripts/truevision_resonance_recorder.py` | exists | Python screen/state recorder. |
| Native capture | `native/truevision_capture_rs` | exists | Rust screen capture and cell-state emitter. |
| Native clarity test | `scripts/truevision_native_clarity_test.py` | exists | Runs capture and replay clarity tests. |
| State replay | `scripts/truevision_state_replay.py` | exists | Replays stored TrueVision cell-state data. |
| Region snip | `scripts/truevision_region_snip.py` | exists | Selects/snaps screen regions for capture prep. |
| Still capture | `scripts/truevision_still_image_capture.py` | exists | Converts still images into video-shaped TrueVision state. |
| Storage initializer | `scripts/truevision_storage_library.py` | exists | Creates clean external vault directories. |

## Generation And Render Tools

| Tool | Path | Status | Purpose |
| --- | --- | --- | --- |
| Scene generator | `scripts/truevision_state_scene_generator.py` | exists | Declared state scene generation. |
| Full power frame | `scripts/truevision_full_power_frame.py` | exists | High-detail still/frame experiment. |
| Path tracer | `scripts/truevision_path_tracer.py` | exists | Path-tracing render lane. |
| Render template | `scripts/truevision_render_template.py` | exists | Renders reusable JSON templates. |
| Edge audio river | `scripts/truevision_edge_audio_river.py` | exists | Audio-reactive river visualizer. |
| Edge v3 | `scripts/truevision_edge_world_v3.py` | exists | Edge/smoke/river thematic renderer. |
| Basement narrative | `scripts/truevision_basement_stick_narrative.py` | exists | Story/lyrics/audio narrative renderer. |

## Signature And TrueFrameGen Tools

| Tool | Path | Status | Purpose |
| --- | --- | --- | --- |
| Signature profile extractor | `scripts/truevision_signature_profile_extract.py` | exists | Extracts motion/look profiles from captured state. |
| Lightning/flash extractor | `scripts/truevision_extract_lightning_signature.py` | exists | Extracts 6-1-6 peak hot-cell lighting signatures. |
| Edge projection | `scripts/truevision_project_edge_from_capture.py` | exists | Projects capture dynamics across Edge Of The World audio. |
| TrueFrameGen fill | `scripts/trueframegen_fill.py` | exists | Fills missing frames from surrounding state. |
| 6-1-6 temporal map | `trueframegen/temporal_616.py` | exists | 6-prior / center / 6-future temporal mapping. |
| Causal cell map | `trueframegen/causal_cell_map.py` | exists | Cell-level cause/effect estimation. |
| State interpolator | `trueframegen/state_interpolator.py` | exists | Interpolates missing cell state. |
| Gap filler | `trueframegen/frame_gap_filler.py` | exists | Builds missing frame state. |
| Missing frame render | `trueframegen/render_missing_frame.py` | exists | Renders missing frame output. |
| Continuity verifier | `trueframegen/verify_replay_continuity.py` | exists | Checks continuity and error. |
| Temporal projector | `trueframegen/temporal_causality_projector.py` | exists | Projects captured 6-1-6 motion over audio timeline. |
| Lightning signature | `trueframegen/lightning_signature.py` | exists | Scores spikes and extracts hot cells. |

## Current Lighting Signature Lane

Current flow:

```text
TrueVision capture
-> score intensity spikes
-> find peak playback frame
-> pull 6 prior / peak / 6 future cells
-> save hot-cell signature JSON
-> fire signature on music peaks and bass pressure
```

Current known issue:

```text
The first signature overlay can become a white blob.
```

Next correction:

```text
signature hot cells
-> thin / skeletonize into branching energy
-> weight by edge continuity and direction
-> reject filled-area blobs
-> render as branching light pressure, not a solid overlay
```

Use actual lightning reference footage for a real lightning signature. A fog or
atmosphere recording can only create a peak-flash signature.

## Heavy Storage Lanes

External vault:

```text
E:\TruEVision Generation
```

Important lanes:

```text
library/source_audio/wav
library/source_audio/mp3
library/source_video/mp4
library/source_stills/jpg
library/truevision_captures
library/capture_units/20_minute/incoming
library/capture_units/20_minute/runs
library/capture_units/20_minute/profiles
library/capture_units/20_minute/reports
library/signature_profiles/fog
library/signature_profiles/smoke
library/signature_profiles/lighting
library/signature_profiles/camera_motion
library/renders/previews
library/renders/full
library/trueframegen
```

## Terminology Research Queue

These are terms to research and map into the tool language later:

```text
luminance
chroma
contrast
gamma
tone mapping
color grading
histogram equalization
CLAHE
edge detection
gradient field
optical flow
motion vectors
temporal coherence
temporal denoising
frame interpolation
super-resolution
deblocking
sharpening
unsharp masking
bilateral filtering
volumetric fog
participating media
phase function
light scattering
ray marching
depth cueing
parallax
motion blur
bloom
glow
exposure
specular reflection
diffuse reflection
ambient occlusion
path tracing
screen-space effects
particle advection
fluid simulation
signed distance fields
procedural noise
fractal Brownian motion
curl noise
reaction-diffusion
beat detection
onset detection
spectral flux
low/mid/high band energy
dynamic range
compression
sidechain-style visual response
```

## Tooling Rules

```text
Chat thinks.
Templates preserve.
Tools validate.
Renderer executes.
Manifest records.
Receipts constrain.
Learning improves.
```

No tool is allowed to claim a generated render is observed evidence.

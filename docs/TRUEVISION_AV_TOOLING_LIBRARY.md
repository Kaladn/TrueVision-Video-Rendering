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

## Render Law

This is now a project law:

```text
Grid/state storage is not the same thing as pixel presentation.
Production render lanes must be state/grid/primitive first.
Pixels are final output, not the primary reasoning layer.
```

Bad production shape:

```text
for every output pixel:
  solve the whole scene again
```

Allowed only as:

```text
prototype
visual sketch
validation pass
final rasterization
```

Correct production shape:

```text
audio/video signal
-> state/grid fields
-> temporal fields / primitive lanes
-> deterministic frame plan
-> final raster/encode
-> manifest + receipts + hashes
```

Hardware-use law:

```text
Use CPU cores for parallel state/render work.
Use GPU hardware encoders when available.
Record encoder, render threads, bitrate, frame count, duration,
state-log interval, wall time, memory, and deterministic hashes.
```

Plain-language law:

```text
TrueVision does not earn trust by hand-painting pixels.
It earns trust by preserving state, showing what drove the output,
and proving the run with manifests and hashes.
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

## TrueAudio Boundary

TrueAudio is a sibling audio-state lane used alongside TrueVision.

```text
TrueVision = visual state.
TrueAudio = audio state before playback/output.
Renderer = consumes validated state from both when needed.
```

TrueAudio now has two capture paths:

```text
file pre-output:
  source WAV/MP3/etc
  -> FFmpeg PCM decode
  -> derived state

file replayable:
  source WAV/MP3/etc
  -> FFmpeg PCM decode
  -> replayable spectral state
  -> raw PCM discarded

machine pre-output:
  local machine output mix
  -> Windows WASAPI loopback
  -> derived state
```

Both paths derive compact audio state, then discard the PCM. They do not save
raw audio, do not claim speech recognition, and do not identify speakers.

TrueAudio state can be replayed only as sonification:

```text
state JSONL
-> deterministic level/band/stereo/transient sonification
-> WAV proof
```

This is useful for hearing whether the log carried rhythm, pressure, and stereo
shape. It is not source-audio recovery.

Close audio replay uses a different contract:

```text
machine output mix
-> replayable spectral state
-> WAV reconstruction
```

This still does not save raw PCM or a source audio file, but it does save enough
derived spectral state to make audio replay possible. The manifest must mark
this as replayable derived audio state.

Current TrueAudio state channels include:

```text
rms_left
rms_right
rms_norm
dbfs
peak_abs
stereo_balance
stereo_width
zero_crossing_rate
bass
mid
high
attack
decay
transient
silence
```

Hard law:

```text
Decode before sound.
Log derived state.
Replay logs as sonification only, not as recovered original audio.
Close replay requires the explicit replayable spectral-state tool and must say so.
```

## TrueSpeech Boundary

TrueSpeech is a state conversion lane built on TrueAudio.

Current implemented lane:

```text
replayable TrueAudio state
-> spectral speech/background detector
-> frame confidence JSONL
-> timestamped speech-like segments
-> manifest + receipt
```

Candidate lyric timing can then use provided lyrics:

```text
speech-like segments
+ provided lyrics
-> candidate line timings
-> manifest + receipt
```

This is not speech-to-text yet. It does not produce ASR transcripts,
verified word timings, speaker identity, or semantic claims. Candidate lyric
alignment means the words came from the provided lyrics and the timing came
from audio state.

Planned split:

```text
TrueSpeech In:
audio state -> speech/background timing -> phoneme/word candidates later

TrueSpeech Out:
text/anchors -> phoneme plan -> voice state -> replayable audio state later
```

Fast ingestion law:

```text
Machine loopback capture is realtime.
Replayable state analysis can run faster than realtime.
FFmpeg decode can run faster than realtime for file ingestion.
```

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
| Studio tooling | `truevision_runtime/studio/studio_tooling.py` | exists | Reusable Studio tool contracts and render preset library. |
| Storage library | `truevision_runtime/storage_library.py` | exists | Keeps repo storage and external vault layout tidy. |
| Template renderer | `truevision_runtime/rendering/template_renderer.py` | exists | Generic template-driven renderer. |
| TrueAudio runtime | `trueaudio_runtime/` | exists | Logs derived pre-output audio state from decoded PCM. |
| TrueAudio CLI | `scripts/trueaudio_log_pre_sound.py` | exists | Command-line entry point for TrueAudio state logging. |
| TrueAudio machine CLI | `scripts/trueaudio_log_machine_pre_sound.py` | exists | Command-line entry point for local machine output-mix logging. |
| TrueAudio state replay CLI | `scripts/trueaudio_replay_state.py` | exists | Renders deterministic WAV sonification from TrueAudio state logs. |
| TrueAudio file replayable CLI | `scripts/trueaudio_log_file_replayable.py` | exists | Builds replayable spectral state from a source audio file through FFmpeg decode. |
| TrueAudio replayable capture CLI | `scripts/trueaudio_log_machine_replayable.py` | exists | Captures machine output as replayable spectral state, not raw PCM. |
| TrueAudio replayable replay CLI | `scripts/trueaudio_replay_replayable.py` | exists | Reconstructs close WAV audio from replayable spectral state. |
| TrueSpeech detection CLI | `scripts/truespeech_detect_segments.py` | exists | Detects speech/background segments from replayable TrueAudio state. |
| TrueSpeech lyric alignment CLI | `scripts/truespeech_align_lyrics_candidate.py` | exists | Aligns provided lyrics to speech-state windows as candidates, not ASR truth. |
| Atmosphere tools CLI | `scripts/truevision_atmosphere_tools.py` | exists | Lists, profiles, and writes reusable fog/mist/cloud/rain-glass state toolsets. |

## AV Tool Bus Names

These are the current validated tool names in the runtime registry:

```text
audio_probe_duration
audio_analyze_levels
audio_extract_features
trueaudio_log_pre_sound
trueaudio_log_machine_pre_sound
trueaudio_replay_state
trueaudio_log_file_replayable
trueaudio_log_machine_replayable
trueaudio_replay_replayable
truespeech_detect_segments
truespeech_align_lyrics_candidate
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
source_snap_tool
existing_state_animator
electric_glow_intensity_animator
spectrum_audio_reactive_city
atmosphere_profile_from_capture
atmosphere_toolset_create
source_surface_capture_plan
source_surface_multi_sample_plan
source_surface_video_state_receipt
element_creation_profile_from_capture
frame_diff_replay_accuracy
manifest_browser
render_preset_library
local_qwen_controller
```

## TrueVision Studio Tool Set

These are the reusable Studio-level controls. They are tool contracts, not
throwaway scripts.

| Tool | Purpose |
| --- | --- |
| Source Snap Tool | Creates still/video source-state packets for record, regen, or generation reference. |
| Existing-State Animator | Animates only existing source-state regions without adding composition. |
| Electric/Glow Intensity Animator | Pulses existing lightning, glow, halo, tower, waveform, or analyzer regions by intensity. |
| Spectrum/Audio-Reactive City Tool | Maps audio pressure to skyline, windows, glow, fog, and beat-synced frame pressure. |
| Atmosphere Capture Profiler | Reads native `.tvcells` captures and derives fog/mist/cloud/rain-glass profiles with 6-1-6 windows. |
| Weather / Atmosphere Toolset | Creates reusable state tools for fog, mist, cloud volume, and rain drops on glass. |
| TruDepth Contract Layer | Defines Volumetric State Field, Effect State Profile, Effect State Transform, and the TruDepth Logging Array contracts. |
| Source-Surface Capture Planner | Builds deterministic approved-source capture plans where recording starts before play and stops from source video time. |
| Source-Surface Multi-Sample Planner | Splits large approved videos into four section samples so the system learns across the source without retaining bulky teacher state. |
| Source-Surface Video-State Receipt | Verifies canonical address-bar navigation, resolved URL/title/duration, nonblank state, profile creation, and teacher-state purge before capture can count. |
| Element Creation Profile Tool | Converts temporary teacher state into compact creation-useful behavior signatures, then purges bulky observed state after verification. |
| Frame Diff / Replay Accuracy Tool | Measures source-vs-regenerated drift through manifests, state, or frame artifacts. |
| Manifest Browser | Reads render/capture manifests through the Studio instead of manual folder digging. |
| Render Preset Library | Promotes successful render lanes into reusable presets and templates. |
| Local Qwen Controller | Lets Qwen plan AV state and request tools through validation and receipts. |

Current reusable presets:

```text
glitch_444_alive_poster
fade_away_memory_cathedral
house_remix_audio_city
storm_ember_city
mirror_maze_realism
edge_audio_river
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
| Atmosphere weather tools | `scripts/truevision_atmosphere_tools.py` | exists | Creates reusable fog, mist, cloud, and rain-glass state toolsets from defaults or native capture profiles. |

## Atmosphere / Weather Toolset

This lane turns reference captures and successful experiments into reusable
state tools. It is not object detection and it is not a fluid simulator yet.
It produces deterministic state profiles that renderers can consume.

TruDepth law:

```text
Copy behavior, not pixels.
Transform state, not identity.
Validate before render.
```

The machine-readable TruDepth contract lives at:

```text
truevision_runtime/learning_intake/trudepth_contracts.py
```

The human contract is documented at:

```text
docs/TRUDEPTH_LAW_AND_CONTRACTS.md
```

Current elements:

```text
fog_density_field
mist_veil_field
cloud_volume_field
rain_glass_field
```

Core state channels:

```text
density
veil_opacity
scatter_bloom
edge_softness
motion_pressure
curl_pressure
occlusion_pressure
droplet_density
droplet_streak
refraction
surface_wetness
```

TruDepth logging-array channels:

```text
density_slice_near
density_slice_mid
density_slice_far
density_delta
depth_layer
depth_confidence
occlusion_pressure
light_scatter
bloom_bleed
reveal_rate
edge_recovery
contrast_recovery
texture_birth
motion_parallax
parallax_direction_16
angular_energy_16
softness
persistence_frames
validation_flags
```

Capture-learning flow:

```text
native TrueVision .tvcells capture
-> limited state-frame sample
-> density / softness / motion / bloom profile
-> 6-1-6 temporal windows
-> reusable atmosphere toolset template
-> manifest
```

Hard boundary:

```text
No raw frames required.
No semantic labels required.
No claim that generated atmosphere is evidence.
Grid/state is storage and planning. Pixels come last.
```

## Elemental Learning Intake

Elemental Learning Intake is the controlled self-training loop for visual
elements. It lets TrueVision gather observed video state, build 6-1-6 profiles,
and promote compact signatures into reusable render state.

It is not a browser bot and not a general automation layer.

```text
element list
-> approved source candidate
-> capture plan
-> native TrueVision capture
-> 6-1-6 profile
-> compact learned signature
-> renderer binding
-> manifest + receipt
```

Today the lane is visual only. Correlated sound signatures come later after the
visual element profiles are stable.

Cross-system shape:

```text
TrueVision Generation = capture/profile/render
AnchorWorks = later names, counts, lexicons, neighborhood meaning
SecureCore = later policy, approvals, retention, receipts
connector = validated state packets
```

Source-surface timing law:

```text
approved visible source
-> declare player region
-> start native capture
-> wait pre-roll
-> hit approved play button
-> run until source video duration plus post-roll
-> write plan and receipt
```

The source-surface planner is:

```text
source_surface_capture_plan
```

It accepts a visible source URL/title, video duration, player region, FPS,
state resolution, grid, pre-roll, and post-roll. It outputs a native capture
command with `--region`, a timeline, a manifest plan, and a receipt. It does
not click account controls, comments, downloads, likes, subscriptions, or
recommendations.

Creation-profile closeout law:

```text
temporary teacher capture
-> element_creation_profile_from_capture
-> creation-useful behavior signature
-> profile hash verification
-> purge bulky observed teacher chunks
-> keep profile, manifest, receipt, purge report
```

The creation profile keeps fields useful for rendering an element from scratch:

```text
shape behavior
growth/decay
edge softness
density/opacity
bloom/intensity
occlusion behavior
rhythm/pulse
transition behavior
camera relation
renderer binding
```

It does not keep the observed movie as durable memory. The durable result is
how to create the element, not state recall of the teacher source.

Smoke teacher default:

```text
42s source video
0.25x playback
~168s observed duration
~180s capture window
15 FPS native capture
640x360 grid
no audio
```

See:

```text
docs/ELEMENTAL_LEARNING_INTAKE_PLAN.md
```

Example:

```powershell
$env:PYTHONPATH='.'
python scripts\truevision_atmosphere_tools.py create `
  --storage-root storage `
  --run-id fog_mist_density_field_v1 `
  --capture-manifest "E:\TruEVision Generation\library\capture_units\fog_mist_density_field_v1\incoming\<run>\<run>_manifest.json"
```

## Signature And TrueFrameGen Tools

| Tool | Path | Status | Purpose |
| --- | --- | --- | --- |
| Signature profile extractor | `scripts/truevision_signature_profile_extract.py` | exists | Extracts motion/look profiles from captured state. |
| Lightning/flash extractor | `scripts/truevision_extract_lightning_signature.py` | exists | Extracts 6-1-6 peak hot-cell lighting signatures. |
| Meter Grid | `scripts/truevision_meter_grid.py` | exists | Extracts cell meter summaries, candidate event profiles, PNG tuning graphs, and section-ranking support. |
| Edge projection | `scripts/truevision_project_edge_from_capture.py` | exists | Projects capture dynamics across Edge Of The World audio. |
| TrueFrameGen fill | `scripts/trueframegen_fill.py` | exists | Fills missing frames from surrounding state. |
| TrueFrameGen upsample | `scripts/trueframegen_upsample.py` | exists | Generates in-between state frames inside the source timeline for higher FPS playback. |
| TrueFrameGen live upsample | `scripts/trueframegen_live_upsample.py` | exists | Reads native `.tvcells` chunks before the final manifest and renders trailing high-FPS output. |
| TrueFrameGen live pipeline | `scripts/trueframegen_live_pipeline.py` | exists | Starts native capture and launches TrueFrameGen after a trailing delay while capture continues. |
| Rust TrueFrameGen stream | `native/truevision_capture_rs/src/bin/trueframegen_stream_rs.rs` | exists | Compiled bounded-cache renderer from `.tvcells` to 60fps MP4 through ffmpeg; supports `libx264`, QSV, AMF, D3D12, Vulkan encoder names, `temporal-map`, `recursive-midpoint`, center-crop proofs, confidence traces, and RGB neighbor smoothing. |
| 6-1-6 temporal map | `trueframegen/temporal_616.py` | exists | 6-prior / center / 6-future temporal mapping. |
| Causal cell map | `trueframegen/causal_cell_map.py` | exists | Cell-level cause/effect estimation. |
| State interpolator | `trueframegen/state_interpolator.py` | exists | Interpolates missing cell state. |
| Gap filler | `trueframegen/frame_gap_filler.py` | exists | Builds missing frame state. |
| Frame upsampler | `trueframegen/frame_upsampler.py` | exists | Converts low-FPS captures to high-FPS output without extending duration. |
| Live upsampler | `trueframegen/live_upsampler.py` | exists | Watches native chunk state so generation can overlap capture. |
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

The first TrueVision-native distillation lives here:

```text
docs/TRUEVISION_STATE_GENERATION_PRIMITIVES.md
```

Use that document as the system vocabulary before adding new render controls or
tool names.

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

# TrueVision Generation Lab TODO

This file is the project handoff. If the thread goes cold, start here.

## Core Law

```text
Forward TrueVision witnesses.
Reverse TrueVision replays, regenerates, or demonstrates state.
Generated media is synthetic state media, not evidence.
Raw frames are not implied unless explicitly saved.
SecureCore guards later.
AnchorWorks routes later.
The lab proves the audio/video state-media language first.
```

## Current State

- [x] Standalone project root exists at `D:\TrueVision_Generation_Lab`.
- [x] Project is separated from SecureCore and AnchorWorks.
- [x] Static studio UI exists at `ui/truevision_state_media_studio.html`.
- [x] Local studio server exists at `scripts/truevision_studio_server.py`.
- [x] Local Qwen/Ollama proxy route exists.
- [x] Catbot conversation lane exists.
- [x] One flat daily chat file exists under `storage/chats/YYYY-MM-DD.jsonl`.
- [x] Template save/load/delete lane exists under `storage/templates/`.
- [x] Manual minutes recording setup exists in the UI.
- [x] State-aware countdown requirements are in the UI/tests.
- [x] Region snip tool exists.
- [x] Recorder/replay/generator scripts exist.
- [x] Still-image TrueVision capture exists.
- [x] Path tracing lane exists.
- [x] Edge audio river visualizer exists.
- [x] The Basement stick-figure narrative renderer exists.
- [x] Signature profile extractor exists.
- [x] Basement renderer can consume a signature profile bundle.
- [x] AV-only tool registry exists.
- [x] AV-only policy validation exists.
- [x] AV tool receipts exist.
- [x] Catbot can request AV tools through validated JSON.
- [x] Chat-origin AV tool calls force `human_confirmed=false`.
- [x] Delete and full render execution are gated.
- [x] Model-neutral PromptToStateAdapter base exists.
- [x] WAV feature extraction AV tool exists.
- [x] ffmpeg audio level analysis AV tool exists.
- [x] Initial state-pattern library exists.
- [x] Test suite was green after signature extractor addition: `73 tests`.
- [x] COD full-screen 20-minute signature capture exists.
- [x] COD signature profiles were extracted.
- [x] Basement full arc was rendered with COD motion/look signature.

## Morning Re-Entry

- [ ] Start the studio:

```powershell
cd D:\TrueVision_Generation_Lab
python scripts\truevision_studio_server.py
```

- [ ] Open:

```text
http://127.0.0.1:8765/
```

- [ ] Confirm Qwen settings:

```text
Provider: Ollama native
Proxy: enabled
Proxy endpoint: http://127.0.0.1:8765/api/local-llm/chat
Model: qwen3-coder:30b or the local Intel Qwen model name
```

- [ ] Ask Catbot a plain project question and confirm it answers without writing files.
- [ ] Ask Catbot for an AV tool request and confirm a receipt appears in `storage/receipts/`.
- [ ] Run tests before new work:

```powershell
$env:PYTHONPATH="D:\TrueVision_Generation_Lab\scripts;D:\TrueVision_Generation_Lab\modules;D:\TrueVision_Generation_Lab"
python -m unittest discover -s tests -v
```

## Layer 0: Project Discipline

- [ ] Keep this project audio/video only.
- [ ] Do not add general desktop automation.
- [ ] Do not add security enforcement logic.
- [ ] Do not add email/browser/app-control tools.
- [ ] Do not allow autonomous clicking.
- [ ] Do not allow autonomous typing.
- [ ] Do not let Qwen execute directly.
- [ ] Keep every state-changing action behind the local policy runner.
- [ ] Keep every write receipt-backed.
- [ ] Keep every render manifest-backed.

## Layer 1: In-Repo Storage

- [x] Create `storage/` runtime root.
- [x] Keep generated runtime data ignored by default.
- [x] Keep only storage directory structure in git.
- [x] Create storage lanes:

```text
storage/
  artifacts/
  chats/
  events/
  inbox/
  manifests/
  outbox/
  presets/
  receipts/
  reports/
  state_chunks/
  templates/
  tmp/
```

- [ ] Add `storage/README.md` explaining each lane.
- [ ] Add storage retention rules for previews, manifests, receipts, and state chunks.
- [ ] Add a storage report command that lists disk usage by lane.
- [ ] Add a cleanup command for generated previews only.
- [ ] Keep cleanup gated by explicit human confirmation.

## Layer 2: AV Tool Bus

- [x] Add `truevision_runtime/av_tools/av_tool_registry.py`.
- [x] Add `truevision_runtime/av_tools/av_tool_policy.py`.
- [x] Add `truevision_runtime/av_tools/av_tool_runner.py`.
- [x] Add `truevision_runtime/av_tools/av_tool_receipts.py`.
- [x] Add `truevision_runtime/av_tools/av_recalibration.py`.
- [x] Expose `GET /api/av-tools`.
- [x] Expose `POST /api/av-tools/call`.
- [x] Write receipts for accepted calls.
- [x] Write receipts for rejected calls.
- [x] Reject unknown tools.
- [x] Reject path escape attempts.
- [x] Gate `template_delete`.
- [x] Gate `video_execute_full_render`.

- [ ] Add JSON schema files for every AV tool.
- [ ] Add tool argument examples under `docs/av_tools/examples/`.
- [ ] Add `render_job_status`.
- [x] Add `audio_extract_features`.
- [x] Add `audio_analyze_levels`.
- [x] Add `template_from_audio_signals`.
- [ ] Add `video_render_score`.
- [ ] Add `video_extract_keyframes`.
- [ ] Add `video_composite_layers`.
- [ ] Add `video_apply_time_warp`.
- [ ] Add `video_color_grade`.
- [ ] Add `variant_compare`.
- [ ] Add a policy test for every tool.

## Layer 3: Qwen Transformation Helper

- [x] Catbot can chat through local Qwen.
- [x] Catbot system prompt tells Qwen the AV-only boundary.
- [x] Catbot can parse a strict AV tool JSON request.
- [x] Browser posts Qwen tool requests to `/api/av-tools/call`.
- [x] UI displays accepted/rejected tool result messages.

- [ ] Add a visible "Tool Request" panel in the UI.
- [ ] Show the exact JSON Qwen proposed before execution.
- [ ] Add operator buttons:

```text
Run safe tool
Reject
Save as template draft
Ask Qwen to revise
```

- [ ] Keep full render and delete buttons separate from chat.
- [ ] Add a "why rejected" view for policy failures.
- [ ] Add examples Qwen can learn from:

```text
prompt -> state template
time marker note -> template patch
render failure -> corrected template
audio duration -> timeline match
```

- [ ] Store Qwen transformation examples under:

```text
storage/events/qwen_transform_examples.jsonl
```

- [ ] Add a local prompt pack:

```text
docs/qwen/TRUEVISION_AV_TOOL_RULES.md
docs/qwen/PROMPT_TO_STATE_LANGUAGE.md
docs/qwen/RECALIBRATION_EXAMPLES.md
docs/qwen/RENDER_FAILURE_EXAMPLES.md
```

- [x] Add initial model-neutral adapter contract docs:

```text
docs/qwen/TRUEVISION_PROMPT_TO_STATE_CONTRACT.md
docs/qwen/WAV_TO_VIDEO_WORKFLOW.md
docs/qwen/STATE_PATTERN_LIBRARY.md
```

## Layer 4: Prompt-To-State Language

- [ ] Define `truevision_state_template_v1`.
- [x] Add minimal prompt-to-state validator.
- [x] Add minimal repair loop.
- [x] Add model-neutral prompt context builder.
- [x] Add initial state-pattern library context.
- [ ] Define required template fields:

```text
template_id
created_at
source_kind
prompt
duration_seconds
fps
timeline
scene
camera
layers
audio_reactivity
motion_rules
materials
lighting
render_constraints
safety_boundary
manifest
```

- [ ] Define optional template fields:

```text
lyrics_theme
style_fingerprint
computer_vision_hints
path_tracing_hints
physics_hints
geometry_hints
linear_algebra_hints
trigonometry_hints
electronics_signal_hints
recalibration_notes
learning_links
```

- [ ] Add template validator.
- [ ] Add template normalizer.
- [ ] Add template patcher.
- [ ] Add template variant creator.
- [ ] Add template diff command.
- [ ] Add template-to-render-job compiler.
- [ ] Reject templates that claim evidence or raw reconstruction.

## Layer 5: Audio River Visualizer

- [x] Generate Edge-style audio river video.
- [x] No lettering/glyph visual layer.
- [x] Thin river into letterbox-like black bands.
- [x] Stamp the program that made the state-change video.

- [ ] Locate and document the current awesome video path in `storage/artifacts/INDEX.md`.
- [ ] Add exact song duration matching.
- [ ] Add lyric-sheet theme ingestion.
- [ ] Keep theme simple:

```text
black field
thin river of color
joined flow
waking up
people as one
ancient Windows snake-screen-saver feeling
Milkdrop-like sound reaction
no words
no glyphs
```

- [ ] Add audio feature cache.
- [x] Add ffmpeg level/peak/valley observer.
- [x] Add section energy extraction.
- [x] Add signal-to-pattern template creation.
- [ ] Add amplitude band mapping.
- [ ] Add low/mid/high color lane mapping.
- [ ] Add beat bloom mapping.
- [ ] Add chorus intensity mapping.
- [ ] Add calm section smoothing.
- [ ] Add render preview at 10-15 seconds.
- [ ] Add full-song render job preparation.
- [ ] Add full render execution behind approval.
- [ ] Save every visualizer setup as a reusable template.

## Layer 5B: Stick-Figure Narrative Video

- [x] Add `scripts/truevision_basement_stick_narrative.py`.
- [x] Read the operator-supplied story file.
- [x] Read the operator-supplied album lyrics file.
- [x] Render a full-song arc from `The Basement.mp3`.
- [x] Keep the story literal instead of abstract-only:

```text
storm
basement door
hallway window creature
Frank falling
dragged descent
red rift
Nether World
mirror warning
sword awakening
demon battle
mother rescue
rift escape
seal and ascend
```

- [x] Keep lyric text and dialogue cards off-screen.
- [x] Drive lighting and pressure from audio features.
- [x] Write frame-state JSONL.
- [x] Write manifest, report, thumbnail, visual-only video, and audio-muxed video.
- [x] Add optional `--signature-profile` to apply captured motion/look grammar.
- [ ] Add this renderer behind an AV tool call.
- [ ] Add template controls for scene beat percentages.
- [ ] Add Qwen template compiler examples for narrative state sequencing.
- [ ] Add operator time-marker recalibration for story beats.

## Layer 5C: Signature Library

- [x] Capture 20 minutes of full-screen COD-style video state:

```text
storage/artifacts/signature_captures/cod_fullscreen_20m_signature_v2
```

- [x] Preserve no raw frames.
- [x] Preserve 160x90x16 cell state chunks.
- [x] Extract reusable signature profiles:

```text
motion_profile.json
camera_shake_profile.json
edge_density_profile.json
contrast_color_profile.json
energy_timing_profile.json
cut_rhythm_profile.json
signature_profile_bundle.json
```

- [x] Render `the_basement_full_arc_cod_signature_v1` with the extracted profile.
- [ ] Add signature profile listing to the UI.
- [ ] Add signature profile selection to template creation.
- [ ] Add signature strength controls:

```text
camera_shake_strength
motion_blur_strength
edge_shimmer_strength
contrast_grade_strength
flash_response_strength
```

- [ ] Add signature profile AV tools:

```text
signature_profile_extract
signature_profile_list
signature_profile_apply_to_template
```

## Layer 6: Recalibration Workflow

- [x] Add `recalibration_add_note`.
- [x] Add `recalibration_apply`.
- [x] Add `time_marker_add`.
- [x] Add `time_marker_list`.

- [ ] Add UI time marker input:

```text
timecode
note
target parameter
direction
confidence
source artifact
template id
```

- [ ] Support notes like:

```text
1:12 river too thick
2:04 colors should calm down
3:30 chorus should bloom harder
```

- [ ] Convert notes to structured recalibration records.
- [ ] Let Qwen propose a patch from notes.
- [ ] Show the patch before applying.
- [ ] Save old template before patching.
- [ ] Save patched template as a new version.
- [ ] Compare before/after variants.
- [ ] Record whether the operator liked the patch.

## Layer 7: Capture And Snip

- [x] Add region snip tool.
- [x] Snap selected region to 16:9.
- [x] Prepare recorder command.
- [x] Add manual recording duration in minutes.
- [x] Add countdown fields in UI.

- [ ] Finish interactive snip-to-record workflow.
- [ ] Add "watch selected region" start button.
- [ ] Add state-aware countdown overlay:

```text
dark background -> bright countdown
bright background -> dark countdown
busy background -> outline/shadow/backplate
similar color conflict -> invert or pulse
```

- [ ] Log countdown metadata:

```text
countdown_enabled
countdown_seconds
overlay_position
contrast_mode_used
record_start_time
```

- [ ] Add multi-monitor metadata.
- [ ] Add selected-region preview.
- [ ] Add region preset list.
- [ ] Add region preset rename.
- [ ] Add region preset delete behind confirmation.
- [ ] Add capture dry-run command.
- [ ] Add 1-minute capture test.
- [ ] Add 1-hour capture plan.
- [ ] Add storage math report for long captures.

## Layer 8: TrueVision Still And Video State

- [x] Capture still image into video-shaped TrueVision data.
- [x] Recreate still from data only.
- [x] Prove stored state can approximate the source layout.

- [ ] Add focus reconstruction from stored state only.
- [ ] Do not reread original photo/video during reconstruction.
- [ ] Add deterministic focus methods:

```text
Lanczos upsample
CLAHE/local contrast
edge-gated unsharp mask
bilateral cleanup
chroma-safe saturation restore
cell-boundary deblocking
```

- [ ] Add state-density profiles:

```text
draft
preview
full
archive
```

- [ ] Add richer per-cell channels:

```text
rgb_mean
rgb_min
rgb_max
luma_mean
luma_variance
edge_strength
dominant_hue
saturation
motion_delta
texture_energy
local_contrast
confidence
```

- [ ] Add optional raw-frame lane with explicit enable flag.
- [ ] Log when detail is missing and cannot be honestly recovered.

## Layer 9: Rendering Backend

- [x] Basic state replay exists.
- [x] Scene generator exists.
- [x] Path tracer exists.
- [x] Audio river generator exists.

- [ ] Make `video_render_preview` produce an actual preview artifact.
- [ ] Make `video_prepare_full_render` produce a complete render job manifest.
- [ ] Make `video_execute_full_render` execute approved jobs only.
- [ ] Add render queue records under `storage/events/render_jobs.jsonl`.
- [ ] Add render outputs under `storage/artifacts/`.
- [ ] Add render manifests under `storage/manifests/`.
- [ ] Add render reports under `storage/reports/`.
- [ ] Add render failure receipts.
- [ ] Add frame count, duration, fps, resolution, and hash to every manifest.
- [ ] Add hardware snapshot to full render reports.
- [ ] Add process memory snapshot to full render reports.

## Layer 10: Learning Twin

- [ ] Add observer twin records.
- [ ] Add renderer twin records.
- [ ] Observer twin learns from recorded/captured state.
- [ ] Renderer twin learns from render success/failure.
- [ ] Store lessons as evidence-linked AV records, not vague memory.
- [ ] Track which channels improved output.
- [ ] Track which channels caused artifacts.
- [ ] Track prompt language that produced useful templates.
- [ ] Track prompt language that produced bad templates.
- [ ] Let Qwen read learning summaries.
- [ ] Do not let Qwen rewrite learning records directly.
- [ ] Add `learning_record_save`.
- [ ] Add `learning_summary_build`.

## Layer 11: SecureCore Slot-In Shape

- [ ] Keep backend ports shaped so SecureCore can replace local adapters later.
- [ ] Add writer port.
- [ ] Add artifact port.
- [ ] Add policy port.
- [ ] Add notification port.
- [ ] Add event envelope port.

Target shape:

```text
truevision_runtime/
  ports/
    writer_port.py
    artifact_port.py
    policy_port.py
    notification_port.py
    event_envelope.py
```

- [ ] Add local adapters first:

```text
truevision_runtime/
  adapters/
    local_writer_adapter.py
    local_artifact_adapter.py
    local_policy_adapter.py
    local_notification_adapter.py
```

- [ ] Add SecureCore adapters only after local contracts are stable:

```text
truevision_runtime/
  adapters/
    securecore_writer_adapter.py
    securecore_artifact_adapter.py
    securecore_policy_adapter.py
    securecore_notification_adapter.py
```

- [ ] Keep SecureCore import optional.
- [ ] Do not require SecureCore to run the lab.
- [ ] Do not rewrite lab code when SecureCore integration begins.

## Layer 12: Rust / Compiled Lane

- [ ] Keep Python as the state-language prototype.
- [ ] Add Rust only when Python contracts are boring and stable.
- [ ] First Rust target:

```text
truevision-capture-worker.exe
```

- [ ] Rust responsibilities:

```text
high-rate screen/window capture
selected-region capture
timing precision
ring buffer
chunk writer
hash-chain writer
low-overhead disk spool
native tray/CLI control
```

- [ ] Python responsibilities:

```text
state language
templates
reports
learning rules
Qwen transformation helper
render prototypes
policy tests
```

## Layer 13: Quality Gates

- [ ] Every new tool gets a test.
- [ ] Every storage write gets a receipt.
- [ ] Every render gets a manifest.
- [ ] Every generated artifact gets a hash.
- [ ] Every destructive action requires explicit human confirmation.
- [ ] Every Qwen tool request is visible before sensitive execution.
- [ ] Every prompt-to-state output validates before render.
- [ ] Every capture states whether raw frames were saved.
- [ ] Every regen states its source and limits.

## Next Three Sessions

### Session 1: Make The Tool Bus Feel Real

- [ ] Add visible tool-request panel.
- [ ] Add safe-run button for non-gated AV tools.
- [ ] Add policy rejection view.
- [ ] Make `video_render_preview` produce a small real preview.
- [ ] Commit.

### Session 2: Make The River Workflow Real

- [ ] Index the current Edge river video path.
- [ ] Add lyrics/theme ingestion.
- [ ] Add exact song duration matching.
- [ ] Add time marker UI.
- [ ] Add recalibration patch preview.
- [ ] Commit.

### Session 3: Make Capture Useful

- [ ] Finish snip-to-record in the UI.
- [ ] Finish state-aware countdown.
- [ ] Run a 1-minute selected-region capture.
- [ ] Pull one data slice.
- [ ] Regen from that slice.
- [ ] Commit.

## Do Not Forget

```text
Chat thinks.
Templates preserve.
Renderer executes.
Manifest proves.
Receipts constrain.
Learning improves.
SecureCore slots in later.
```

# TrueVision Generation Lab

Standalone project root for TrueVision state-media generation experiments.

This project was split out from:

```text
D:\SecureCore_Workspace\SecureCore\TRUEVISION_STATE_MEDIA_POC
```

It is intentionally separate from SecureCore and AnchorWorks:

```text
SecureCore guards, logs, approves, and preserves.
AnchorWorks maps shape/count/routing language.
TrueVision Generation Lab builds the state-media generation tools.
```

## Core Boundary

```text
Forward TrueVision witnesses.
Reverse TrueVision replays or demonstrates state.
Generated state media is synthetic, not evidence.
Raw frames are not implied unless explicitly saved.
```

## What This Project Does

```text
observed video/photo
-> addressed visual state
-> temporal cell tensors
-> deterministic replay
-> state-driven generation experiments
-> manifest/report/output trail
```

The renderer must consume state. Prompt text never becomes authority by itself.

## Current Base Build

```text
ui/
  truevision_state_media_studio.html

scripts/
  truevision_studio_server.py
  truevision_storage_library.py
  truevision_resonance_recorder.py
  truevision_state_replay.py
  trueframegen_fill.py
  truevision_state_scene_generator.py
  truevision_full_power_frame.py
  truevision_path_tracer.py
  truevision_still_image_capture.py
  truevision_region_snip.py
  truevision_edge_audio_river.py
  truevision_edge_world_v3.py
  truevision_basement_stick_narrative.py
  truevision_signature_profile_extract.py
  truevision_extract_lightning_signature.py
  truevision_project_edge_from_capture.py
  truevision_render_template.py

templates/
  hard_has_no_meaning_here_mirror_made.json

truevision_runtime/
  rendering/
    template_renderer.py

trueframegen/
  temporal_616.py
  causal_cell_map.py
  state_interpolator.py
  frame_gap_filler.py
  render_missing_frame.py
  verify_replay_continuity.py
  temporal_causality_projector.py
  lightning_signature.py

modules/
  screen_grid_mapper.py

native/
  truevision_capture_rs/
    Rust native TrueVision screen recorder

screen_resonance_state.py

tests/
  unit tests for recorder, replay, scene generation, still capture,
  path tracing, grid math, and region snipping

presets/
  operator-selected region presets

connected_artifacts/
  generated local artifacts, ignored by default

outputs/
  ad hoc render/capture outputs, ignored by default
```

Tool inventory and terminology queue:

```text
docs/TRUEVISION_AV_TOOLING_LIBRARY.md
```

## External Storage Vault

Heavy captures and long renders should go to the external runtime vault, not the repo:

```text
E:\TruEVision Generation
```

Initialize the tidy audio/video library layout:

```powershell
cd D:\TrueVision_Generation_Lab
python scripts\truevision_storage_library.py init --root "E:\TruEVision Generation"
```

Run the studio against that vault:

```powershell
$env:TRUEVISION_STORAGE_ROOT="E:\TruEVision Generation"
python scripts\truevision_studio_server.py
```

or:

```powershell
python scripts\truevision_studio_server.py --storage-root "E:\TruEVision Generation"
```

The library keeps file types separated:

```text
library/source_audio/wav
library/source_audio/mp3
library/source_video/mp4
library/source_stills/jpg
library/truevision_captures/
library/capture_units/20_minute/
library/signature_profiles/fog
library/signature_profiles/smoke
library/signature_profiles/lighting
library/signature_profiles/camera_motion
library/renders/previews
library/renders/full
library/trueframegen/
```

Twenty-minute clips are the default signature-learning unit. They are long enough for fog drift, camera rhythm, lighting drift, and motion texture, but short enough to index, compare, rerun, and delete cleanly.

## Template Rendering

Reusable renderers live under `truevision_runtime/rendering/`. New videos should usually be JSON templates, not new song-specific Python files.

Example:

```powershell
python scripts\truevision_render_template.py templates\hard_has_no_meaning_here_mirror_made.json
```

The first high-power template lane is `mirror_maze_realism`: mirror corridors, cracked glass, smoke, shards, live-wire motion, silhouette blocking, bloom/grain finishing, and manifest-backed CPU/RAM/encoder stats.

## TrueFrameGen

TrueFrameGen fills missing frames from existing TrueVision cell-state captures. It does not rewrite the recorder and it does not claim missing raw reality was recovered.

```text
recorded frames
-> TrueVision cell state
-> 6-1-6 temporal map
-> missing-frame cause/effect estimate
-> generated in-between state
-> rendered frame
-> verification pass
```

Hard law:

```text
TrueVision records.
6-1-6 explains temporal causality.
TrueFrameGen fills only the missing state between known states.
```

Example:

```powershell
python scripts\trueframegen_fill.py --run-dir D:\path\to\truevision_capture --radius 6
```

## Native Rust Capture

The Python recorder is still useful for compatibility and experiments, but high-density capture belongs in native code. The first native lane is:

```text
native/truevision_capture_rs
```

Build:

```powershell
cd D:\TrueVision_Generation_Lab\native\truevision_capture_rs
cargo build --release
```

Example native capture:

```powershell
cd D:\TrueVision_Generation_Lab
.\native\truevision_capture_rs\target\release\truevision_capture_rs.exe `
  --duration 2 `
  --fps 9 `
  --resolution 2560x1440 `
  --grid 640x360 `
  --output-root "E:\TruEVision Generation\library\capture_units\20_minute\incoming" `
  --run-id "truevision_rs_1440p_640x360_test"
```

Native chunks use:

```text
tvcells_f32le_v1
```

Python replay supports both legacy compressed `.npz` chunks and native `.tvcells` chunks.

Proof run:

```text
Python 1440p / 640x360: 5 frames over 2.542s
Rust   1440p / 640x360: 18 frames over 2.061s
```

Prepare a playback clarity test without recording:

```powershell
python scripts\truevision_native_clarity_test.py
```

Run it only after the screen/video is ready:

```powershell
python scripts\truevision_native_clarity_test.py --execute
```

GPU boundary notes:

```text
reports/NATIVE_CAPTURE_GPU_BOUNDARY.md
```

## Open The Studio

Start the local studio proxy, then open the browser route:

```powershell
cd D:\TrueVision_Generation_Lab
python scripts\truevision_studio_server.py
```

```text
http://127.0.0.1:8765/
```

The proxy forwards local Qwen/Ollama requests to:

```text
http://127.0.0.1:11434/api/chat
```

Server-backed studio operations:

```text
GET  /api/files
POST /api/state/request
POST /api/state/plan
POST /api/record/prepare
POST /api/local-llm/chat
POST /api/assistant/message
GET  /api/chat/today
POST /api/chat/log
GET  /api/templates
POST /api/templates/save
POST /api/templates/delete
POST /api/media/probe
GET  /api/av-tools
POST /api/av-tools/call
```

Catbot action words:

```text
plain conversation -> queue qwen_chat and answer through local Qwen, with no storage write
compile / generate / draft / qwen -> save request, then queue Qwen compile when local Qwen is selected
save / persist / write / store -> save current request
prepare / record / capture / command -> save request and prepare recorder command
files / list / refresh -> refresh storage file list
```

Qwen chat and generation templates:

```text
storage/chats/YYYY-MM-DD.jsonl
storage/templates/*.json
```

Plain chat stays plain chat. Templates preserve generation setup, media timing, renderer target, and validated state plan. When an audio path has a known duration, template timelines can match the song duration exactly.

AV tool calls are audio/video only:

```text
audio_probe_duration
audio_analyze_levels
audio_extract_features
template_from_audio_signals
template_create / template_load / template_save / template_patch
time_marker_add / recalibration_add_note
video_render_preview / video_prepare_full_render
```

Qwen may request these tools as JSON, but the local server validates the tool name, arguments, confirmation state, and flat filenames before anything runs. Every accepted or rejected call writes a receipt under `storage/receipts/`.

Prompt-to-state adapter contract:

```text
human prompt + project context + schema
-> model draft JSON
-> validator
-> repair loop if invalid
-> canonical TrueVision AV state JSON
-> AV tool runner
```

WAV files can drive videos through the audio river renderer:

```powershell
python scripts\truevision_edge_audio_river.py `
  --audio "D:\path\to\song.wav" `
  --output-root "outputs\wav_river" `
  --run-id "song_river" `
  --fps 12
```

WAV files can also produce state signals first:

```text
audio_analyze_levels
-> levels, peaks, valleys, rising/falling energy, section energy
-> state pattern library
-> geometry/color/motion template
```

Story-and-song narrative videos can be rendered through the Basement stick renderer:

```powershell
python scripts\truevision_basement_stick_narrative.py `
  --audio "C:\Users\mydyi\Downloads\The Basement.mp3" `
  --story "C:\Users\mydyi\OneDrive\Documents\Desktop\The Basement.txt" `
  --lyrics "C:\Users\mydyi\OneDrive\Documents\Desktop\Full Album Lyrics_sound.txt" `
  --run-id "the_basement_full_arc_v1" `
  --width 1280 `
  --height 720 `
  --fps 12
```

That renderer keeps the output literal to the story scenery: storm, basement door, hallway window creature, Frank falling, dragging descent, red rift, Nether World, sword awakening, rescue, sealing, and ascension. It does not render lyric text or dialogue cards.

Capture-derived motion/look signatures can be extracted from a completed TrueVision recording:

```powershell
python scripts\truevision_signature_profile_extract.py `
  --capture-dir "storage\artifacts\signature_captures\cod_fullscreen_20m_signature_v2" `
  --output-dir "storage\artifacts\signature_profiles\cod_fullscreen_20m_signature_v2" `
  --profile-id "cod_fullscreen_20m_signature_v2"
```

The extractor writes reusable, non-evidence profiles:

```text
motion_profile.json
camera_shake_profile.json
edge_density_profile.json
contrast_color_profile.json
energy_timing_profile.json
cut_rhythm_profile.json
signature_profile_bundle.json
```

The Basement renderer can consume the bundle as an abstract motion/look signature:

```powershell
python scripts\truevision_basement_stick_narrative.py `
  --audio "C:\Users\mydyi\Downloads\The Basement.mp3" `
  --story "C:\Users\mydyi\OneDrive\Documents\Desktop\The Basement.txt" `
  --lyrics "C:\Users\mydyi\OneDrive\Documents\Desktop\Full Album Lyrics_sound.txt" `
  --run-id "the_basement_full_arc_cod_signature_v1" `
  --signature-profile "storage\artifacts\signature_profiles\cod_fullscreen_20m_signature_v2\signature_profile_bundle.json"
```

Edge Of The World v3 renders the edge/smoke/river-below concept with machine-cost telemetry:

```powershell
python scripts\truevision_edge_world_v3.py `
  --run-id "edge_of_the_world_v3_edge_smoke_river_codsig_v1" `
  --width 1280 `
  --height 720 `
  --fps 24 `
  --signature-profile "storage\artifacts\signature_profiles\cod_fullscreen_20m_signature_v2\signature_profile_bundle.json"
```

The v3 manifest records wall time, process CPU seconds, average logical CPU percent, and process memory snapshots. The current full render is:

```text
outputs\edge_of_the_world_v3\edge_of_the_world_v3_edge_smoke_river_codsig_v1\edge_of_the_world_v3_edge_smoke_river_codsig_v1_full_audio.mp4
```

## Edge 6-1-6 Projection

TrueFrameGen can use an atmospheric TrueVision capture as a temporal teacher,
then project that learned 6-1-6 motion history across the full Edge Of The
World song. This is projection from observed state dynamics, not a copied video
loop.

```powershell
python scripts\truevision_project_edge_from_capture.py `
  --capture-run-dir "E:\TruEVision Generation\library\capture_units\20_minute\incoming\YOUR_1_MINUTE_CAPTURE" `
  --run-id "edge_of_the_world_616_projected_atmosphere_v1" `
  --resolution 2560x1440 `
  --fps 12
```

Preview first:

```powershell
python scripts\truevision_project_edge_from_capture.py `
  --capture-run-dir "E:\TruEVision Generation\library\capture_units\20_minute\incoming\YOUR_1_MINUTE_CAPTURE" `
  --run-id "edge_616_projection_preview" `
  --resolution 2560x1440 `
  --fps 12 `
  --max-seconds 5
```

Hell/edge-stacked style preview:

```powershell
python scripts\truevision_project_edge_from_capture.py `
  --capture-run-dir "E:\TruEVision Generation\library\capture_units\20_minute\incoming\YOUR_1_MINUTE_CAPTURE" `
  --run-id "edge_616_hell_power_walk_preview" `
  --resolution 2560x1440 `
  --fps 12 `
  --max-seconds 5 `
  --style hell_power_walk
```

Peak/lightning-style flash signatures are extracted from captured TrueVision
state, not hand-drawn first. The extractor scores playback frames for intensity
spikes, pulls the 6-prior / peak / 6-future cell neighborhood, and writes a
reusable lighting signature:

```powershell
python scripts\truevision_extract_lightning_signature.py `
  --capture-run-dir "E:\TruEVision Generation\library\capture_units\20_minute\incoming\YOUR_CAPTURE" `
  --signature-id "edge_teacher_peak_flash_signature"
```

Use that signature during projection so music peaks and bass drops fire the
captured cell pattern with red rhythm transitions:

```powershell
python scripts\truevision_project_edge_from_capture.py `
  --capture-run-dir "E:\TruEVision Generation\library\capture_units\20_minute\incoming\YOUR_CAPTURE" `
  --run-id "edge_616_hell_power_walk_signature_flash_preview" `
  --resolution 2560x1440 `
  --fps 12 `
  --max-seconds 10 `
  --style hell_power_walk `
  --lightning-signature "E:\TruEVision Generation\library\signature_profiles\lighting\edge_teacher_peak_flash_signature.json"
```

If the source capture is fog/atmosphere, the output is a peak-flash signature.
If the source capture is real lightning footage, the same tool becomes the
lightning-signature snagging lane.

## Run Tests

```powershell
cd D:\TrueVision_Generation_Lab
$env:PYTHONPATH="D:\TrueVision_Generation_Lab\scripts;D:\TrueVision_Generation_Lab\modules;D:\TrueVision_Generation_Lab"
python -m unittest discover -s tests -v
```

## Snip A Region

Manual region:

```powershell
python scripts\truevision_region_snip.py `
  --region 640,360,1280,720 `
  --preset-id center_video `
  --print-command
```

Interactive drag selection:

```powershell
python scripts\truevision_region_snip.py --select --preset-id video_window --print-command
```

Run the existing recorder against the selected region:

```powershell
python scripts\truevision_region_snip.py `
  --region 640,360,1280,720 `
  --preset-id center_video `
  --duration 30 `
  --fps 15 `
  --watch
```

The snip tool does not replace the recorder. It only creates a clean region preset and calls:

```text
truevision_resonance_recorder.py --region left,top,width,height --resolution 960x540 --grid 160x90 --blocks 16x9
```

## Region Snip Law

```text
User selects rough rectangle.
Tool snaps to 16:9.
Tool clamps to monitor bounds.
Tool records preset.
Recorder captures state.
No raw video by default.
```

## Rust / Compiled Lane

Python owns the proof language for now. Rust should come in when the shape is stable.

Good Rust targets:

```text
screen/window region picker
high-rate capture backend
background chunk writer
ring buffer / retention spool
hash-chain artifact writer
native tray/CLI controller
```

Do not rewrite the renderer or state language in Rust until the Python contracts are boring and proven.

## Next Order

```text
1. Finish region snip/watch workflow.
2. Add focus reconstruction from stored state only.
3. Add prompt-to-state compiler contracts.
4. Capture more real samples.
5. Only then consider compiled Rust capture backend.
```

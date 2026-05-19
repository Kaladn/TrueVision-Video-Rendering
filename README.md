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
  truevision_resonance_recorder.py
  truevision_state_replay.py
  truevision_state_scene_generator.py
  truevision_full_power_frame.py
  truevision_path_tracer.py
  truevision_still_image_capture.py
  truevision_region_snip.py
  truevision_edge_audio_river.py
  truevision_edge_world_v3.py
  truevision_basement_stick_narrative.py
  truevision_signature_profile_extract.py

modules/
  screen_grid_mapper.py

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

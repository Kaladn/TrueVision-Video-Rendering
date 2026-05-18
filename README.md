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
  truevision_resonance_recorder.py
  truevision_state_replay.py
  truevision_state_scene_generator.py
  truevision_full_power_frame.py
  truevision_path_tracer.py
  truevision_still_image_capture.py
  truevision_region_snip.py

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

The current studio is static HTML. It has no backend wiring yet.

```powershell
Invoke-Item D:\TrueVision_Generation_Lab\ui\truevision_state_media_studio.html
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

# Human Input To Video Output Pipeline

This is the intended TrueVision Generation Lab operating path.

## Law

```text
Record state.
Plan state.
Transform state.
Render pixels last.
Prove every run.
```

Generated media is synthetic state media. It is not evidence.

## Pipeline

```text
human intent
-> approved external API draft
-> PromptToStateAdapter contract
-> schema validation
-> AV-only tool policy
-> reusable preset or AV state template
-> audio feature extraction
-> native Rust render/capture lane
-> hardware encoder
-> MP4 + manifest + frame-state JSONL + report
```

## Human Input

The human supplies one or more of:

```text
prompt or creative direction
audio file
lyrics or story notes
source still/video reference
time-marker recalibration notes
render preset choice
```

The model is not trusted as an executor. It may draft a state request only.

## Adapter Contract

The adapter wraps the model endpoint with:

```text
system rules
allowed AV fields
project context
JSON schema
examples
repair loop
```

The app trusts only validated state JSON.

Bad model output:

```text
rejected or repaired
```

Good model output:

```text
promoted to template or tool request
```

## Tool Policy

Only AV tools are allowed.

Examples:

```text
audio_probe_duration
audio_extract_features
template_create
template_patch
video_render_preview
video_prepare_full_render
video_execute_full_render
manifest_generate
receipt_create
render_preset_library
```

Not allowed:

```text
general desktop automation
security enforcement
email/browser/app control
autonomous clicking
autonomous typing
generic AI assistant tools
```

## Template Or Preset

Templates preserve the intended output shape.

Sample templates:

```text
templates/center_warp_laserfield_sample.json
templates/storm_ember_city_no_cheat.json
```

Render presets live in:

```text
truevision_runtime/studio/studio_tooling.py
```

Current meaningful preset states:

```text
proven: working lane with successful tests or artifacts
ready: coherent lane, not fully promoted
draft: working proof, not final architecture
needs_rework: rejected visual direction or bad product fit
```

## Audio Analysis

Audio is observed, not treated as magic.

Current signals include:

```text
duration
rms / level
bass pressure
high pressure
beat pressure
waveform bins
```

Open work:

```text
true FFT or Goertzel frequency bins
```

## Native Render

Heavy render work belongs in Rust/native.

Example beast-mode command:

```powershell
native\truevision_capture_rs\target\release\truevision_weird_occlusion_rs.exe `
  --output-root outputs\weird_occlusion_rs `
  --run-id sample_center_warp_laserfield `
  --scene-mode warp_laser_field `
  --palette glitch_444 `
  --size 1280x720 `
  --fps 30 `
  --duration 30 `
  --audio "replace-with-local-audio-file.wav" `
  --sample-rate 44100 `
  --mux-audio true `
  --render-threads 32 `
  --video-encoder h264_qsv `
  --bitrate 24M `
  --state-log-every 1
```

Fallback encoders to test:

```text
h264_amf
h264_d3d12va
hevc_qsv
libx264
```

## Outputs

Runtime outputs belong outside git or in ignored output lanes.

Expected render bundle:

```text
output.mp4
output_manifest.json
output_frame_state.jsonl
output_report.md
proof_frame.png
```

The manifest/report should include:

```text
source audio path
render command
scene mode
duration
frame count
fps
encoder
bitrate
render threads
state log interval
wall time
memory use
hashes
boundary statement
```

## Validation

Before calling a lane proven:

```text
Rust build passes.
Python tests pass.
Output has real duration.
Manifest exists.
Frame-state line count matches frame count when required.
Run report exists.
Generated media remains outside git.
```

## Current Best Proof

Latest full-song proof:

```text
outputs/weird_occlusion_rs/glitch_444_house_center_warp_laserfield_full_beast_qsv_rs
```

Summary:

```text
191.44 seconds
5743 frames
5743 frame-state records
1280x720 at 30fps
32 CPU render threads
Intel QSV hardware encode
1.216x realtime
```

The output media is ignored and should not be committed.

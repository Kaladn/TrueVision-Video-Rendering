# TrueVision Geometry Generation Engine

Status: active local worker lane.

Purpose:

```text
logger outputs
-> filters
-> candidate geometry
-> data-carrying shape units
-> reusable big-shape library
-> proof overlay / receipt
```

This is not object truth recognition and not a decorative SVG layer. Geometry is the interpreted state layer built from logger outputs.

## Core Law

```text
A shape must carry its source truth.
Filters may reveal it, but raw local data owns it.
```

Every generated shape keeps these fields separate:

```text
geometry:
  points / bounds / arcs / planes / vectors

source_region:
  frame/time/cell coordinates or source artifact scope

raw_data_ref:
  manifest/profile/tvcells/frame-state references when available

true_local_metrics:
  brightness, color, delta, edge, motion, texture, saturation, etc.

filtered_metrics:
  bloom, glow, branch mask, direction fields, recognizer/filter family
```

Generated media is only a demonstration of the logged state. It is not evidence.

## Logger Lanes

The engine exposes the current logger lanes into the proof scene as overlays, meter panels, geometry marks, state panels, or receipt refs:

```text
native_frame_state
meter_grid
angular_seismic_16dir
state_focus_lens
truedepth
atmosphere_weather
motion_vectors
occlusion
light_shadow_vectors
element_creation_profiles
timing_progress_receipts
trueaudio_state
driving_awareness
worker_forge
```

## First Big-Shape Library

```text
road_plane
horizon_band
vanishing_corridor
fog_bank
depth_wall
light_cone
occlusion_slab
motion_stream
reflection_vector_field
atmosphere_volume
lightning_branch
water_plane
```

These are reusable state forms. They are not promoted object labels.

## CLI

```powershell
python scripts\truevision_geometry_engine.py `
  --run-id geometry_logger_overlay_all_lanes_v0 `
  --storage-root storage `
  --output-root outputs\geometry_generation `
  --render-preview `
  --duration 10 `
  --fps 30 `
  --width 1280 `
  --height 720 `
  --meter-grid-profile storage\artifacts\meter_grid\example_profile.json `
  --angular-seismic-profile storage\artifacts\angular_seismic\example_profile.json `
  --state-focus-profile storage\artifacts\state_focus_lens\example_profile.json `
  --element-creation-profile storage\artifacts\element_creation_profiles\example_profile.json
```

## Outputs

```text
storage/artifacts/geometry_generation/<run>_geometry_scene.json
storage/artifacts/geometry_generation/<run>_big_shape_library.json
storage/manifests/geometry_generation/<run>_manifest.json
storage/receipts/geometry_generation/<run>_receipt.json
outputs/geometry_generation/<run>/<run>_logger_overlay.mp4
```

## Boundary

```text
Recognition proposes.
Raw state proves.
Geometry binds.
Renderer demonstrates.
Receipts prove the run.
```

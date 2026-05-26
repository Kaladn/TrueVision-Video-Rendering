# TruDepth Law And Contracts

TruDepth is the TrueVision depth/effect logging layer for reusable visual
behavior. It is not a claim of optical light-field capture and it is not a
source-frame copier.

Core law:

```text
Copy behavior, not pixels.
Transform state, not identity.
Validate before render.
```

Plain-language lock:

```text
Meters teach the effect.
ARC transforms the effect.
Renderer speaks the transformed state.
```

## Why This Exists

Flat visual effects are easy to fake and easy to make ugly. Fog, smoke,
lightning, water, reflections, glow, and terrain depth become believable when
the system records their relationships:

```text
density
depth
occlusion
reveal
motion
light
edge recovery
```

TruDepth turns those relationships into contracts that can be logged, changed,
validated, and rendered.

## Volumetric State Field

Primitive name:

```text
Volumetric State Field
```

Required channels:

```text
density_slice
depth_layer
occlusion_pressure
light_scatter
reveal_rate
edge_recovery
motion_parallax
```

Minimum layer model:

```text
near
mid
far
```

For fog-road footage, this matters because the observed world resolves by
depth:

```text
far world = hidden
middle world = soft
near world = resolved
foreground = fast passing
```

## Effect State Profile

An effect profile is a compact behavior signature extracted from source
material. It keeps the manner, not the movie.

Meters:

```text
density_over_depth
edge_recovery_rate
contrast_rise
texture_birth
bloom_bleed
parallax_speed
reveal_distance
```

Retention law:

```text
teacher frames retained: false
teacher video retained: false
compact behavior signature retained: true
source hash required: true
```

## Effect State Transform

The ARC-style transform acts on state controls:

```text
direction
density
depth_bias
reveal_window
motion_vector
light_source
near_mid_far_weight
```

Allowed operators:

```text
rotate_direction
deepen_density
invert_depth_bias
compress_reveal_window
expand_reveal_window
redirect_motion
reweight_near_mid_far
change_light_source
```

Forbidden:

```text
source_pixel_copy
source_frame_copy
identity_promotion_without_validation
```

Example:

```text
fog_profile_from_road
-> rotate_direction upward
-> density +0.25
-> light_source warm_center
-> reveal_target figure
-> transformed_effect_state
```

## TruDepth Logging Array

The big logging array is per-frame, per-cell. It is not kept forever by
default. It exists to produce compact summaries, profiles, graphs, and receipts.

Required fields:

```text
schema_version
frame_index
time_sec
cell_x
cell_y
cell_id
source_profile_id
effect_type
transform_id
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

Durable outputs:

```text
per_region_depth_summary
effect_event_profile
transform_profile
validation_receipt
```

Retention:

```text
keep raw cell array by default: false
keep compact summaries: true
keep validation receipts: true
```

## Belongs Tests

Every transformed effect has to still belong to its effect family.

Fog must still:

```text
soften edges
reduce contrast with distance
reveal nearer objects first
drift slowly
occlude without hard borders
```

Lightning must still:

```text
spike fast
bloom outward
lift surrounding exposure
decay quickly
leave short afterglow
```

Ocean must still:

```text
persist as mass
move in bands
shimmer locally
keep horizon or plane behavior
```

## Promotion Rule

No transformed effect becomes a renderer rule unless:

```text
source_profile_hash_present
transform_profile_hash_present
no_source_frames_used
belongs_rules_checked
receipt_written
```

If validation fails:

```text
render rule promotion allowed: false
keep output as proof: false
human review required: true
```

## Code Contract

Machine-readable builders live in:

```text
truevision_runtime/learning_intake/trudepth_contracts.py
```

Current bundle schema:

```text
truevision_trudepth_contract_bundle_v1
```

The current fog reveal sample renderer embeds that bundle in its manifest so
prototype videos carry the same law as the future native lane.

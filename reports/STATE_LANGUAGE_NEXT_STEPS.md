# TrueVision State Language Next Steps

## Problem Found

The first generated walking scene failed visually because it was mostly:

```text
rgb_mean cells painted back to frame
```

That is not enough.

Real video state uses:

```text
color distribution
subcell variance
motion pressure
delta luma
edge density
texture
saturation spread
localized active regions
```

The renderer has to consume those channels, and the generator has to emit them honestly.

## Current State Formula

```text
SceneState(t)
  -> object/layer state
  -> high-detail source frame or direct field model
  -> 90x160x16 cell tensor
  -> manifest + records + compressed chunks
  -> replay renderer
```

## Better State Formula

```text
SceneState(t)
  -> camera state
  -> layer fields
  -> actor skeleton
  -> object masks
  -> material fields
  -> lighting fields
  -> motion vector fields
  -> subcell statistics
  -> cell tensor + sidecars
  -> replay compositor
```

## Needed Sidecars

The 16-feature cell tensor is good for telemetry, but not enough for strong generation by itself.

Add sidecars:

```text
object_id_layer
foreground_coverage_layer
depth_hint_layer
motion_dx_layer
motion_dy_layer
edge_orientation_layer
material_class_layer
texture_seed_layer
skeleton_joint_state
camera_state
lighting_state
```

These sidecars should remain state, not prompt text.

## Replay Renderer v2

The replay renderer should not fill cells with flat color.

It should do:

```text
base color from rgb_mean
microtexture from rgb_std and luma_std
contours from edge_density and edge_orientation
surface pressure from texture_energy
temporal smear from motion_energy and delta_luma_abs
color pressure from saturation_mean and hsv state
shape coherence from object_id and skeleton sidecar
depth ordering from depth_hint
```

## Good Claims

```text
We can record video as replayable visual state.
We can render observed state back into a video-like artifact.
We can generate synthetic state media by declaring scene state.
We can measure fidelity against the stored state layer.
```

## Bad Claims

```text
We can recreate raw original pixels without saving them.
We have photoreal state generation.
We can treat synthetic media as evidence.
We can infer missing evidence from generated state.
```

## Next Technical Cut

Build `truevision_state_replay_v2.py`:

```text
input:
  cell_state_npz
  optional sidecars

output:
  full-power frame or clip

must consume:
  rgb_mean
  rgb_std
  luma_std
  edge_density
  texture_energy
  motion_energy
  delta_luma_abs
  saturation_mean

must report:
  channels used
  channels ignored
  input hashes
  output hashes
  synthetic/evidence boundary
```

Then build `truevision_scene_formula_v2.py`:

```text
person walking in field
actor skeleton
foreground coverage
object id
depth hint
motion vector
material/texture sidecar
```

The goal is not a better drawing. The goal is a better state language.


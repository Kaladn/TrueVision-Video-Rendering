# Terrain Realism Teacher Plan

## Purpose

Terrain Teacher is the first realism-learning lane for TrueVision Generation.
It exists because the renderer can follow rhythm and mood, but still needs
physical world structure before it can make convincing cinematic places.

Core law:

```text
Reality first.
Cinema second.
Nightmare third.
```

The immediate target is `edge_nightmare_world --shot-type wide_edge_intro`.
The shot must read as a human standing on a real cliff edge above a real
abyss/ocean, not as an abstract symbol.

## Source Priority

Start with real geography and atmosphere:

```text
1. Ocean cliffs / sea caves / coastal erosion
2. Canyons / desert cliffs / ravines
3. Volcanoes / lava fields / ash clouds
4. Stormy ocean / horizon / wave behavior
5. Mountain ridges / abyss drops / fog layers
```

The machine must learn:

```text
ground
edge
drop
scale
horizon distance
rock texture
water behavior
fog layers
light source direction
camera height
```

## Search Lanes

Seed searches are stored by `terrain_teacher`:

```text
ocean cliffs drone footage 1 hour
coastal cliffs drone 4k long video
sea cave ocean cliff drone footage
cliff edge ocean waves cinematic footage
coastal erosion cliffs documentary 1 hour
how ocean cliffs form documentary
canyon drone footage 1 hour
grand canyon drone footage long
desert canyon cinematic drone footage
canyon formation documentary 1 hour
volcano eruption documentary 1 hour
lava field drone footage 4k
volcanic ash cloud time lapse
mountain ridge fog drone footage
```

Preferred source shape:

```text
30-90 minutes
transcript/captions when available
real geography or documentary source
not gear-review-only
not abstract visualizer
not montage bait
```

## Extracted Rules

Each source becomes compact physical rules, not retained video.

Extract:

```text
horizon behavior
foreground/midground/background separation
scale cues
texture behavior
atmosphere behavior
light source direction
occlusion patterns
terrain edge shapes
depth cues
renderer parameter suggestions
```

Durable rule files:

```text
terrain_teacher/learned/ocean_cliff_rules.jsonl
terrain_teacher/learned/canyon_depth_rules.jsonl
terrain_teacher/learned/volcano_glow_rules.jsonl
terrain_teacher/learned/fog_atmosphere_rules.jsonl
```

## Retention

Retention law:

```text
capture/sparse sample
-> extract physical rules
-> write review packet
-> human approval
-> promote compact rule
-> purge raw teacher media/cache
```

Disk guard defaults:

```text
MAX_TOTAL_CACHE_GB = 10
MAX_ACTIVE_VIDEO_GB = 2
MAX_FRAME_SAMPLES_PER_VIDEO = 120
MAX_KEEP_FRAMES_AFTER_JOB = 12
DELETE_RAW_VIDEO_AFTER_ANALYSIS = true
DELETE_AUDIO_AFTER_TRANSCRIPT = true
```

No raw-media hoarding. No autonomous promotion.

## First Renderer Target

```text
scene_mode: edge_nightmare_world
shot_type: wide_edge_intro
duration_seconds: 12
full_song_render_allowed: false
```

QA metrics:

```text
subject_readability
ground_plane_visibility
edge_visibility
foreground_midground_background_separation
parallax_score
effect_occlusion_ratio
terrain_realism_score
chaos_budget_actual
```

Acceptance:

```text
At thumbnail size, viewer understands:
person standing on cliff edge over abyss/ocean.
```

## Metered Section Choice

Long-video movement must be goal-directed:

```text
probe section
-> build meter grid
-> score target signature
-> rank next sections
-> controller moves only with a meter reason
```

Metered section choice is not browser control by itself. It produces a ranked
plan for a human-owned controller/agent:

```text
section_id
target_signature
score
recommended_action
meter_reasons
rejection_reasons
frame_peak
cell_bounds
```

Current first target:

```text
candidate_lightning
```

Hard laws:

```text
No meter, no section choice.
No graph, no tuning.
No receipt, no success claim.
```

## Future Logger Lanes

Parked for later:

```text
raytracing_alternative_capture_logger
pathtracing_alternative_learn_logger
pathtracing_alternative_transform_logger
arc_learning_transform_logger
```

These are capture/learn/transform logging alternatives, not proof claims.

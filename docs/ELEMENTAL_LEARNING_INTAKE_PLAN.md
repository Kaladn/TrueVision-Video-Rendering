# Elemental Learning Intake Plan

## Purpose

Elemental Learning Intake is the visual-only learning loop for TrueVision
Generation. It lets the lab gather reference state for visual elements such as
fog, smoke, mist, cloud volume, lightning, rain on glass, fire, embers, water,
reflection, and camera motion.

This is not general desktop automation. The goal is not to make a browser bot.
The goal is to let TrueVision build a reusable element memory from observed
video state.

```text
source candidate
-> approved intake task
-> TrueVision capture
-> 6-1-6 temporal map
-> element profile
-> compact learned signature
-> renderer consumes state
-> manifest and receipt
```

## Core Law

```text
TrueVision sees.
TrueVision maps.
TrueVision counts.
TrueVision profiles.
TrueVision renders from learned state.
```

No raw footage is trusted as magic. No generated media becomes evidence. Raw
captures can be temporary teacher material; compact profiles are the durable
learning output.

Retention law:

```text
capture
-> extract creation profile
-> verify profile hash
-> write manifest and receipt
-> purge bulky teacher state
```

The durable artifact is the creation profile, not the observed source.

Open-license prototype law:

```text
approved open-license source
-> temporary teacher cache outside git
-> TrueVision profile extraction
-> attribution receipt
-> purge raw teacher media
```

For prototype learning intake, prefer open-licensed or explicitly approved
sources. If a dataset mixes licenses, each license family must be handled as a
separate source class. Do not treat a mixed dataset as one blanket approval.

## What This Module Learns

The intake lane learns state behavior, not semantic labels.

Examples:

```text
fog:
  density, veil opacity, scatter bloom, edge softness, occlusion pressure

smoke:
  density, curl pressure, rise drift, dissipation, edge softness, turbulence

lightning:
  center line, branch pressure, intensity spike, bloom falloff, flash decay

rain on glass:
  droplet density, streak speed, refraction, merge/split pressure, wetness

clouds:
  volume density, shadow pockets, soft edges, large-mass drift, occlusion
```

## Approved Autonomy Shape

The self-training loop is controlled autonomy, not free mouse control.

Allowed:

```text
element registry
source search terms
candidate source queue
human-approved source URL or local video
capture plan
profile extraction
quality scoring
renderer preset update
receipts and manifests
```

Not allowed in this repo:

```text
arbitrary desktop control
unbounded browser clicking
general web roaming
account actions
comments/uploads/messages
security surveillance
evidence claims
```

Operationally:

```text
TrueVision may recommend candidates.
SecureCore-style policy approves tasks.
The operator or approved harness provides the source.
TrueVision records state.
AnchorWorks-style counts/lexicon can name and compare learned signatures later.
```

## Approved Source Surfaces

Source pages can be described as approved surfaces with explicit display IDs and
button IDs. This lets the intake lane use the page layout safely without
granting broad browser control.

Current source-surface contract:

```text
presets/learning_intake/youtube_source_surface_v1.json
docs/YOUTUBE_SOURCE_SURFACE_SAFE_OPS.md
```

Law:

```text
Display IDs observe.
Button IDs require approval.
Forbidden controls never bind.
Receipts close the loop.
```

YouTube source-surface display IDs:

```text
yt.display.page_url
yt.display.search_query
yt.display.player_region
yt.display.title
yt.display.channel
yt.display.elapsed_time
yt.display.duration
yt.display.fullscreen_state
```

YouTube source-surface button IDs:

```text
yt.button.play_pause
yt.button.seek_to_start
yt.button.fullscreen
yt.button.settings_speed
```

Forbidden controls include account actions, comments, uploads, downloads,
subscriptions, likes/dislikes, recommendations, ads, and arbitrary links.

## Approved Dataset Surfaces

Dataset-backed learning intake is allowed only when the source is explicitly
listed in the third-party notices and a receipt records the exact source family,
license family, sampled actions, profile hash, and purge result.

Current approved prototype dataset candidate:

```text
source_id: hf_faridlab_deepaction_v1
source_url: https://huggingface.co/datasets/faridlab/deepaction_v1
approved_use: human action motion-profile research
prototype_scope: generated folders released under CC BY 4.0
excluded_by_default: Pexels real-video folder until license review
durable_output: compact human-action behavior profiles, manifests, receipts
raw_retention: purge after profile verification
```

This dataset should be used for motion and silhouette behavior, not for copying
visual composition:

```text
walking
running
turning
reaching
dancing
sitting/standing transitions
```

Hard law:

```text
Keep the motion manner.
Do not keep the teacher videos.
```

## System Roles

### TrueVision Generation

Owns:

```text
visual capture
.tvcells state
6-1-6 temporal maps
element profiles
state-pattern templates
render tests
manifest/report output
```

### AnchorWorks Reference Role

AnchorWorks should not run this intake lane. Its role is later symbolic
organization:

```text
element names
state lexicons
lifetime counts
neighborhood/cloud meaning
cross-profile comparison
```

### SecureCore Reference Role

SecureCore should not render or profile. Its role is policy shape:

```text
allowed task registry
source approval
retention law
receipts
health checks
operator/device boundaries
```

Connector:

```text
validated state packets
```

## Completion Law

```text
A completed macro is not a completed capture.
A verified video-state receipt is a completed capture.
```

Approved YouTube sources must use browser address-bar navigation:

```text
approved URL
-> focus browser address bar
-> paste canonical watch URL
-> press Enter
-> wait for video page load
-> verify video element/title/duration
-> only then capture
```

The verified receipt must prove:

```text
resolved_url
video_title
duration_detected
visual_state_records > 0
not_gray_screen
not_error_page
profile_created
teacher_chunks_purged
```

Playlist, search, recommendation, and ephemeral URL noise does not become
source authority.

Coordinate-surface law:

```text
No coordinate map, no run.
No saved map hash, no proof.
No executable map, no clicks.
Template maps can plan only; they cannot execute.
```

Every coordinate intake queue, summary, and receipt must carry the coordinate
map path/hash used for the run. This proves the operation surface existed before
capture started.

Large video rule:

```text
short source -> one compact sample
large source -> four compact samples from different sections
```

For a one-hour source, the default 12-second windows are centered in each
quarter:

```text
00:07:24 - 00:07:36
00:22:24 - 00:22:36
00:37:24 - 00:37:36
00:52:24 - 00:52:36
```

Each sample must independently close out:

```text
verify page
capture state
profile behavior
verify hash
purge teacher chunks
write receipt
```

## Intake Record

Each intake task should be represented as a small JSON record:

```json
{
  "intake_id": "smoke_teacher_001",
  "element_id": "smoke_curl_field",
  "source_kind": "local_video_or_approved_url",
  "source_note": "operator selected",
  "search_terms": ["slow motion smoke plume", "black background smoke"],
  "playback_speed": 0.25,
  "target_duration_seconds": 180,
  "capture": {
    "mode": "full_screen_or_selected_window",
    "fps": 15,
    "resolution": "2560x1440",
    "grid": "640x360",
    "raw_frames_saved": false
  },
  "status": "queued"
}
```

## Creation Profile Closeout

Every approved visual teacher capture must be closed into a creation profile
before moving to the next source.

The profile stores creation-useful fields:

```text
shape_behavior
growth_decay
edge_softness
density_opacity
bloom_intensity
occlusion_behavior
rhythm_pulse
transition_behavior
camera_relation
renderer_binding
```

The AV tool is:

```text
element_creation_profile_from_capture
```

It reads the native `.tvcells` teacher capture, builds compact 6-1-6 windows,
writes a profile/manifest/receipt, verifies the profile hash, and can then
purge the bulky teacher chunks and frame records.

Hard law:

```text
Keep the manner.
Do not keep the movie.
```

Three-source process test:

```text
source 1 capture -> profile -> verify -> purge teacher state
source 2 capture -> profile -> verify -> purge teacher state
source 3 capture -> profile -> verify -> purge teacher state
```

The process is valid only if the final storage contains profiles, manifests,
receipts, and purge reports, with no bulky `.tvcells` teacher chunks left from
the trial.

## Capture Strategy

### Smoke 42s Source

For a 42 second smoke source, use quarter-speed playback:

```text
42s source
0.25x playback
168s observed duration
180s capture window with buffer
15 FPS native capture
640x360 grid
visual only
```

Reason:

```text
smoke detail is in curl, lift, dissipation, and soft edge change.
slowing playback gives more temporal samples without inventing frames.
```

### Lightning Source

Lightning needs sharper temporal samples:

```text
15 FPS minimum for teacher capture
focus on intensity spike, branch center, bloom falloff, flash decay
extract hot-cell signatures with 6 prior / center / 6 future windows
```

### Fog / Mist Source

Fog and mist can tolerate slower motion:

```text
10 FPS is acceptable for large density fields
profile density, edge softness, bloom, occlusion, and motion pressure
```

## Profile Extraction

Every learned element gets:

```text
profile JSON
6-1-6 window list
summary statistics
profile hash
source manifest reference
boundary statement
```

Required profile fields:

```text
schema_version
element_id
source manifest
sampled frame count
state channels
six_one_six radius
window count
summary
profile hash
```

## Retention

Teacher captures can be large. The durable artifact is the learned profile.

Recommended rule:

```text
Keep raw/detail teacher captures until first profile review.
After accepted profile:
  keep profile, manifest, receipt, and small proof preview
  move or expire heavy raw state chunks unless marked useful
```

For rare or excellent captures:

```text
preserve the capture as a gold teacher source
```

## Renderer Consumption

Renderers should consume element profiles as state, not copy the source video.

```text
profile density curve
profile motion pressure
profile edge softness
profile bloom falloff
profile 6-1-6 windows
-> renderer parameter fields
-> deterministic visual behavior
```

Renderer law:

```text
Learned state influences behavior.
It does not clone the source composition.
```

## Today Plan

1. Add smoke to the element registry.
2. Prepare the 42 second smoke source as a visual-only teacher intake.
3. Record at quarter-speed for about 180 seconds.
4. Build a smoke 6-1-6 profile.
5. Compare smoke against the fog/mist profile already captured.
6. Save profile, manifest, receipt, and next-render notes.
7. Do not add sound until the visual lane is stable.

## Next Build Items

```text
element_intake_queue
source_candidate_record
capture_plan_from_intake
profile_from_capture
profile_quality_score
profile_compare
renderer_profile_binding
retention_closeout
```

This becomes the learning intake module. It stays scoped to audio/video state
media and does not become a general automation system.

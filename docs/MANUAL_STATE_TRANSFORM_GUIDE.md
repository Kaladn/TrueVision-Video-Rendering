# Manual State Transform Guide

This guide explains how a human can manually design a TrueVision state
transform before rendering.

The short version:

```text
meaning
-> state layers
-> time arc
-> input signals
-> transform rules
-> render boundary
-> receipt
```

This is not prompt magic. The human chooses what the piece means, then turns
that meaning into controllable state.

## What A State Transform Is

A state transform is a controlled change from one media state to another.

Example:

```text
quiet grief
-> pressure builds
-> memory cracks
-> rage peaks
-> release appears
```

The renderer does not need to understand human emotion. It needs named state
fields it can execute.

```text
quiet grief = low fog, low glow, slow motion
pressure builds = vice closure, bass pressure, red glow
memory cracks = core fracture, white edge bloom
rage peaks = ash storm, high glow, fast pulses
release appears = gold-white fracture, reduced fog, lower red pressure
```

## Step 1: Name The Piece

Give the transform a stable id.

Good:

```text
dead_memory_vice_chamber
edge_audio_river
fade_away_memory_cathedral
```

Avoid names that only describe mood:

```text
cool video
dark thing
song render
```

The id becomes the preset id, scene mode, output folder name, manifest anchor,
and receipt trail.

## Step 2: Write The Boundary

Before designing visuals, lock what the render is and is not.

Template:

```text
This is synthetic state media.
It is not observed evidence.
It uses audio as a timing driver.
It does not use external visual assets.
It must not contain [forbidden literal imagery].
```

Example:

```text
No literal gore.
No cartoon devil.
No monster.
No external source imagery.
The evil is pressure, not a character.
```

This boundary matters because it prevents the transform from drifting into
cheap imagery or false claims.

## Step 3: Split The Song Or Clip Into States

Listen once and divide the track into 5 to 10 major states.

Example:

```text
STATE_01_WHISPER_MEMORY
STATE_02_ROOM_WAKES
STATE_03_VICE_REVEAL
STATE_04_PRESSURE_DROP
STATE_05_FALL_FROM_GRACE
STATE_06_COLLISION_CORE
STATE_07_FINAL_CHORUS_PEAK
STATE_08_OUTRO_RELEASE
```

For each state, write one sentence.

```text
STATE_01_WHISPER_MEMORY:
almost black, low fog, weak red pulse, distant female presence

STATE_06_COLLISION_CORE:
maximum distortion, smoke columns, bass pressure, cracked core
```

Do not start with colors and effects. Start with what changes.

## Step 4: Choose State Layers

A state layer is a named visual behavior the renderer can control.

Example layers:

```text
black_industrial_cathedral_machine
black_iron_vice_jaws
cracked_glowing_memory_core
cold_density_field_fog
chalk_outline_ghosts
burned_photo_fragments
rain_glass_deception_distortion
red_neon_vice_pressure
white_lightning_truth_cuts
ember_ash_memory_bleed
thin_gold_white_survival_fracture
```

Each layer should answer:

```text
What exists?
What can change?
What signal drives it?
What should never happen?
```

Example:

```text
layer: black_iron_vice_jaws
changes: closure distance, edge glow, impact pulse
drivers: bass, beat, pressure stage
never: turn into a monster, face, or body
```

## Step 5: Map Audio To State

Use audio features as control signals, not as decoration.

Common mapping:

```text
rms / loudness      -> general pressure, fog breathing, core bloom
bass / low energy   -> weight, scale, jaw closure, floor pulse
high energy         -> sparks, edge shimmer, rain glass, lightning cuts
beat / transient    -> impact flashes, core cracks, pulse events
vocal presence      -> human veil, phrase pressure, emotional focus
silence             -> release, emptiness, hold frames, reduced motion
```

For a manual transform, write it plainly:

```text
bass closes the vice
highs sharpen the truth cuts
beats crack the memory core
vocals lift fog around the center
silence opens a gold fracture
```

If a signal does not serve the story, do not use it.

## Step 6: Define The Time Arc

Use normalized phase from 0.0 to 1.0.

Example:

```text
0.000 - 0.105  whisper_dead_memory
0.105 - 0.295  room_wakes_bitterness
0.295 - 0.395  vice_reveal
0.395 - 0.585  pressure_drop_truth_cuts
0.585 - 0.720  fall_from_grace
0.720 - 0.895  collision_core
0.895 - 0.962  final_chorus_peak
0.962 - 1.000  outro_release
```

This makes the render deterministic. A given time always belongs to a known
state.

## Step 7: Define Transform Rules

A transform rule says how a layer changes during a state.

Plain format:

```text
when phase is in [state]
and audio signal rises
increase [layer parameter]
within [safe range]
```

Examples:

```text
During pressure_drop_truth_cuts:
if bass rises, close vice jaws up to 74 percent.

During collision_core:
if beat rises, increase core crack brightness and white edge bloom.

During outro_release:
reduce red pressure, thin fog, open a gold-white fracture.
```

Good rules have limits. Bad rules just say "make it intense."

## Step 8: Decide What Gets Logged

The render should leave enough state to explain itself without storing giant
unnecessary data.

Minimum per-frame or interval log:

```text
frame_index
time_seconds
scene
stage
audio rms/bass/high/beat/vocal_presence
phase
key layer measurements
state_layers
render_law
```

For long renders, logging every frame may be wasteful. Logging once per second
can be enough for a presentation receipt.

Example:

```text
state_log_every = 30
```

At 30 fps, this records one state line per second.

## Step 9: Create The Preset

The preset is the human-approved recipe.

It should include:

```text
preset_id
renderer
scene_mode
visual_mode
default_size
default_fps
runtime_defaults
audio_mapping
state_layers
boundary
```

Example summary:

```text
preset_id: dead_memory_vice_chamber
renderer: truevision_weird_occlusion_rs
scene_mode: dead_memory_vice_chamber
default_size: 1080x1920
default_fps: 30
render_threads: 32
encoder: h264_qsv
```

## Step 10: Render A Short Proof

Before a full song, render 5 to 10 seconds.

Check:

```text
Does it open?
Does audio mux?
Does the manifest write?
Does the state log contain the correct scene and layers?
Does the visual obey the boundary?
```

Only then render the full piece.

## Step 11: Verify The Full Render

After rendering, verify:

```text
ffprobe sees video and audio streams
manifest exists
frame-state JSONL exists
frame count matches duration x fps
wall time is recorded
encoder is recorded
memory is recorded
output path is correct
```

Strong receipt fields:

```text
source path
duration
frame count
fps
resolution
encoder
render threads
wall seconds
render speed vs realtime
memory start/end/peak
motifs
boundary
state log path
```

## Manual Worksheet

Copy this for a new transform.

```text
Transform id:

Song/source:

Plain meaning:

Boundary:
- This is synthetic state media.
- This is not evidence.
- Do not include:

Core visual metaphor:

State arc:
1.
2.
3.
4.
5.
6.
7.
8.

State layers:
- layer:
  changes:
  drivers:
  never:

Audio mapping:
- rms:
- bass:
- highs:
- beat:
- vocal_presence:
- silence:

Color law:
- black:
- red:
- white:
- gold:
- fog:
- ash:
- lightning:

Render settings:
- width:
- height:
- fps:
- duration:
- render_threads:
- encoder:
- bitrate:
- state_log_every:

Verification:
- short proof rendered:
- full render rendered:
- manifest present:
- state log present:
- ffprobe passed:
- boundary followed:
```

## Plain English Version

To manually create a state transform, do not begin by saying "make a cool
video." Begin by saying what the video is changing from and into.

Then name the pieces that can change:

```text
fog
pressure
light
motion
cracks
embers
release
```

Tie those pieces to the song:

```text
bass moves heavy things
highs move sharp things
vocals move human things
silence reveals what remains
```

Then render it, prove it, and keep the receipt.


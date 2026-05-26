# Shadow Wolf Storyboard Control Contract

## Purpose

Give the TrueVision render lane clear scene targets for `GREEN MILE __ SHADOW WOL`.

The provided storyboard sheet is visual reference support and order authority. The renderer must create controlled scene states from that authority, not random prison, wolf, or metal-video imagery.

```text
The cell is real.
The wolf is internal.
The storm is earned.
```

## Source Authority

```text
Original sheet:
C:\Users\mydyi\Downloads\Green Mile video proofs May 26, 2026, 05_47_16 AM.png

Ingest packet:
storage/artifacts/shadow_wolf_storyboard_ingest/shadow_wolf_storyboard_authority_v1/shadow_wolf_storyboard_authority_packet.json
```

The authority packet contains `63` ordered panels split from the source sheet.

## Required Arc

```text
isolation
-> counting time
-> star slit / outside world
-> no mirror / no face
-> discipline / control
-> internal fire
-> wolf-shadow as survival instinct
-> void / higher ground
-> call-response flashes
-> final storm / anthem
-> calm still-here ending
```

## Hard Rules

- Use the storyboard panel order as the visual spine.
- Every generated shot must declare `scene_state_id`.
- Every generated shot must declare its emotional function.
- Every generated shot must declare allowed effects and forbidden effects.
- Subject readability must stay above effects.
- Scene changes must move the story forward, not decorate the frame.
- Wolf imagery is shadow, instinct, and discipline, not a literal monster replacement.
- Fire is internal power first; full fire is only earned later.
- Storm language is delayed until the rise/anthem sections.
- Storyboard photos are scene plates, not zoom targets.
- Move the state, not the camera.
- Zoom only when the artwork was made for zoom.

## Camera Law

Default camera rule:

```text
hold the full scene plate
allow only tiny 2-4% drift/parallax
no aggressive zoom
no tight crop
no blur-causing enlargement
```

Allowed camera modes:

```text
LOCKED_PLATE
Full image held. State layers move.

MICRO_DRIFT
Tiny 2-4% motion only. No detail crop.

PARALLAX_PLATE
Foreground/mid/background separated and moved slightly.

DEDICATED_CLOSEUP
Only allowed if the source image was created as a close-up.
```

Forbidden camera modes:

```text
random_zoom
ear_crop
face_mush_zoom
blur_zoom
detail_hunt
ken_burns_slideshow
```

If a close-up is needed, use a dedicated close-up artwork plate. Do not create close-ups by zooming into a wide storyboard image.

## State Motion Layers

Motion should come from state changes inside the image:

```text
fog_drift
breath_haze
light_flicker
shadow_crawl
ember_pulse
orbit_thread
dust_particles
steel_reflection_shift
wolf_shadow_emergence
storm_pressure
star_slit_bloom
glyph_mark_shimmer
```

## Forbidden Output

- literal wolf cosplay
- monster transformation
- random prison wallpaper
- generic metal music-video chaos
- effects covering the emotional subject
- fire everywhere
- storm too early
- wolf before it is earned
- symbol spam without scene function

## Scene States

### SCENE STATE 01 - BLACK CORRIDOR

Function: establish isolation.

Visual: distant black/green steel corridor, no visible face.

Allowed effects: breath haze, slight light flicker.

Forbidden effects: wolf, fire, storm.

### SCENE STATE 02 - CELL GEOMETRY

Function: show confinement.

Visual: narrow cell, steel walls, bunks implied, cold institutional green.

Allowed effects: subtle shadow pressure.

Forbidden effects: dramatic magic effects.

### SCENE STATE 03 - COUNTING MARKS

Function: time becomes structure.

Visual: hand, tally marks, scratched wall.

Allowed effects: tiny pulse on each mark.

Forbidden effects: random glyph spam.

### SCENE STATE 04 - STAR SLIT

Function: first outside-world signal.

Visual: narrow window/slit, one star or white cut.

Allowed effects: dust beam, small bloom.

Forbidden effects: full cosmic sky.

### SCENE STATE 05 - NO MIRROR

Function: identity stripped.

Visual: hidden face, no reflection, wall text or symbol.

Allowed effects: orbit thread begins.

Forbidden effects: clear heroic face reveal.

### SCENE STATE 06 - CONTROL

Function: discipline forms.

Visual: silhouette standing taller, shoulders squared.

Allowed effects: low-intensity chest ember.

Forbidden effects: full flames.

### SCENE STATE 07 - WOLF-SHADOW HINT

Function: instinct appears but stays controlled.

Visual: wolf-shaped shadow behind the man, barely readable.

Allowed effects: rim light, smoke outline.

Forbidden effects: literal wolf body replacing the person.

### SCENE STATE 08 - BONE LANGUAGE

Function: silence becomes language.

Visual: carved marks, glyphs, orbit/gravity lines.

Allowed effects: glyph glow only on beat hits.

Forbidden effects: unreadable symbol soup.

### SCENE STATE 09 - VOID / HIGHER GROUND

Function: breakdown rise.

Visual: dark floor/void, figure rises or stands in light.

Allowed effects: impact pulses, dust/fog displacement.

Forbidden effects: random explosions.

### SCENE STATE 10 - CALL RESPONSE FLASHES

Function: lyric punches.

Visual cuts:

```text
No mirror - face shadow
No need - hands open
No voice - mouth/silence/steel
Still seen - eye/light/rim
```

Allowed effects: hard white/green flashes.

Forbidden effects: long drifting scenes.

### SCENE STATE 11 - SHADOW WOLF ALIVE

Function: transformation statement.

Visual: human silhouette front, wolf-shadow behind, controlled fire core.

Allowed effects: storm behind, rim light.

Forbidden effects: monster attack.

### SCENE STATE 12 - FINAL ANTHEM

Function: power without chaos.

Visual: standing figure, storm/sky behind, distant sparks or others implied.

Allowed effects: gold-white hope fracture, controlled bloom.

Forbidden effects: effect soup.

### SCENE STATE 13 - OUTRO CLEAR

Function: survival, calm.

Visual: quiet cell/corridor, figure still standing.

Allowed effects: one clear light line.

Forbidden effects: big finale effects.

## Required Shot Record

Every generated shot/frame-state sample should be able to report:

```json
{
  "scene_state_id": "SCENE_STATE_01",
  "scene_plate_id": "SW:001",
  "camera_mode": "LOCKED_PLATE",
  "function": "establish isolation",
  "source_panel_ids": ["SW:001"],
  "state_motion_layers": ["breath_haze", "light_flicker"],
  "forbidden_crop_targets": ["ear", "random_face_detail", "random_hand_detail"],
  "allowed_effects": ["breath_haze", "light_flicker"],
  "forbidden_effects": ["wolf", "fire", "storm"],
  "subject_readability": 0.0,
  "state_change_from_previous": "intro_hold",
  "effect_occlusion_ratio": 0.0
}
```

## Acceptance

The next render passes only if it reads as:

```text
isolation -> discipline -> defiance -> survival
```

It fails if it reads as:

```text
random wolf graphics
generic prison wallpaper
monster video
effect soup
```

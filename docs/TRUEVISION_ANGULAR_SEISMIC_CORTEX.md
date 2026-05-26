# TrueVision Angular-Seismic Cortex

## Purpose

The Angular-Seismic Cortex turns visual events into measured directional and
temporal state. It extends the Meter Grid without replacing it.

Core law:

```text
Meters measure state.
Angular fields measure direction.
Seismic traces measure force over time.
Together they prove visual events.
```

This lane exists so TrueVision does not claim:

```text
that looks like lightning
that looks like fog
that looks like ocean
```

It must instead record:

```text
where energy appeared
which direction it moved
how fast it rose
how it spread
how it decayed
whether the whole frame moved
```

## Layer Stack

```text
TrueVision Meter Grid
  base measured visual state

TrueVision Angular Cortex
  Hexadec Direction Field
  Radial Cells
  Director Cells
  Meter Rays
  Angular Signatures
  Radiance Transfer Map

TrueVision Seismic Cortex
  Seismic State Trace
  Event Impulse Profile
  Wavefront Map
  Decay Curve
  Oscillation Bands
  Propagation Vector
  Camera Shake Rejection Score
```

## Hexadec Direction Field

Each candidate source cell receives a 16-sided angular field.

```text
directions = 16
degrees_per_direction = 360 / 16 = 22.5
```

Direction center:

```text
angle_i = i * 22.5 degrees
i = 0..15
```

Direction vector:

```text
dx = cos(angle_i)
dy = sin(angle_i)
```

For a source point:

```text
source = (cx, cy)
radius = r

x = cx + cos(angle_i) * r
y = cy + sin(angle_i) * r
```

## Rings

Start with four rings.

```text
rings = [1, 2, 3, 4]
```

Each ring samples:

```text
16 radial cells
16 director cells
```

So the default directional sampling shape is:

```text
4 rings * 32 directional samples = 128 meter points
```

This is small enough to run often, but rich enough to detect direction,
transfer, bloom spread, and wavefront timing.

## Radial Cells

Radial cells measure where brightness, edge, texture, motion, or color energy
exists along direct angles from the candidate source.

```text
radial_cell_i = sector i at angle i * 22.5 degrees
```

Primary use:

```text
brightness location
edge branch direction
texture direction
motion ray
source-to-environment response
```

## Director Cells

Director cells sit in the pockets between radial cells. They measure where
energy is leaking, transferring, blooming, or drifting between the main rays.

```text
director_angle_i = (i + 0.5) * 22.5 degrees
```

Primary use:

```text
beam transfer
bloom spread
fog drift
wavefront travel
branch interpolation
```

Clean rule:

```text
Radial cells measure where energy is.
Director cells measure where energy is going.
```

## Meter Inputs

The Angular-Seismic Cortex consumes Meter Grid time series and candidate event
windows.

Required meter channels:

```text
luma_mean
luma_peak
luma_delta
saturation
color_temperature
edge_density
edge_orientation
motion_magnitude
motion_direction
texture_energy
flicker_score
bloom_pressure
persistence_frames
softness
occlusion_change
```

## Seismic State Trace

The Seismic Cortex treats visual events as time-series disturbances across a
field.

Per cell or directional sample:

```text
baseline
impulse
rise_time
peak
decay_time
aftershock
oscillation
frequency_band
phase_shift
wavefront_arrival
propagation_direction
neighbor_correlation
field_coherence
global_motion_correlation
```

Impulse:

```text
impulse = I(t) - baseline(t)
```

Rise:

```text
rise_time = t_peak - t_start
```

Decay:

```text
decay_time = t_end - t_peak
```

Neighbor propagation:

```text
arrival_delta = t_neighbor_peak - t_source_peak
```

Direction is inferred from which rings and director cells peak after the
source:

```text
source peaks first
near rings peak next
far rings peak after
the timing pattern gives spread direction
```

## Event Profiles

### Lightning

Lightning is not a line. It is a high-impulse light event plus environmental
response.

Required support:

```text
high impulse
fast rise
high luma peak
short decay
wide bloom wavefront
local-to-regional exposure lift
low persistence
branching angular signature
low global motion correlation
```

Rejection examples:

```text
persistent white reflection
static white line
global camera exposure shift
screen glare
```

### Ocean Surface

Ocean is a persistent oscillating surface, not just blue color.

Required support:

```text
continuous oscillation
repeating wave bands
phase-shifted wave rows
horizontal or diagonal propagation
high texture recurrence
medium persistence
specular shimmer variability
```

### Fog / Mist

Fog is soft occlusion and low-edge drift.

Required support:

```text
low edge density
slow rise
long decay
soft occlusion waves
low-to-medium luma variation
directional drift
layered depth fade
```

### Camera Motion

Camera motion must be separated from object or atmosphere motion.

Required support:

```text
whole-field synchronized displacement
high neighbor correlation
high global motion correlation
low local source confidence
```

This prevents false claims where the whole frame moved and the system mistook
it for element behavior.

## Outputs

Each candidate event should produce:

```text
Seismic State Trace
Event Impulse Profile
Wavefront Map
Decay Curve
Oscillation Bands
Propagation Vector
Camera Shake Rejection Score
Angular Signature
Radiance Transfer Map
```

Minimum receipt fields:

```text
event_type_candidate
frame_start
frame_peak
frame_end
source_cell
ring_count
radial_cell_count
director_cell_count
meter_peaks
rise_time_frames
decay_time_frames
wavefront_arrival_by_ring
propagation_direction
neighbor_correlation
field_coherence
global_motion_correlation
visual_support_reasons
rejection_reasons
```

## Graphs

Graphs are not optional during tuning.

Required graph outputs:

```text
luma_curve
bloom_curve
motion_curve
edge_density_curve
exposure_lift_curve
impulse_curve
decay_curve
wavefront_arrival_curve
oscillation_band_curve
global_motion_correlation_curve
```

Tiny law:

```text
No meter, no claim.
No graph, no tuning.
No profile, no renderer rule.
```

## First Targets

```text
candidate_lightning
candidate_ocean_surface
candidate_fog_mist
candidate_camera_motion
```

## Implementation Contract

Do not replace Meter Grid.

Do not classify from a single still frame.

Do not classify lightning from shape alone.

Do not classify ocean from color alone.

Do not classify fog from softness alone.

Every event claim must include:

```text
meter evidence
temporal trace
directional support
global-motion rejection or admission
receipt
```

## Example Claim Shape

```text
A high-impulse event occurred at frame 1842.
Peak luma rose in 1 frame.
Bloom expanded across rings 1-3.
Director cells show northeast transfer.
Afterglow decayed over 9 frames.
Global motion correlation was low, so this was not camera shake.
Candidate: lightning, visually supported.
```

That is the deterministic movie-watching brain.

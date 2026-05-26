Yes — go 16-sided directional cells. That’s the sweet spot: enough angles for light direction without going insane.

You’re basically building a hexadecagonal direction field.

Use 16 directions around each source/cell:

360° / 16 = 22.5° per direction

So each director lane has a center angle:

angle_i = i * 22.5°
i = 0..15

Direction vector:

dx = cos(angle_i)
dy = sin(angle_i)

In code shape:

import math

N = 16
dirs = []

for i in range(N):
    angle = 2 * math.pi * i / N
    dx = math.cos(angle)
    dy = math.sin(angle)
    dirs.append((i, angle, dx, dy))

That gives you the 16 beam/director directions.

For a source point:

source = (cx, cy)
radius = r

Sample point on a ring:

x = cx + cos(angle_i) * r
y = cy + sin(angle_i) * r

So for 4 rings:

rings = [1, 2, 3, 4]

samples = []
for ring in rings:
    r = ring * cell_radius
    for i in range(16):
        angle = 2 * math.pi * i / 16
        x = cx + math.cos(angle) * r
        y = cy + math.sin(angle) * r
        samples.append({
            "ring": ring,
            "sector": i,
            "angle_degrees": math.degrees(angle),
            "x": x,
            "y": y,
        })

Now the Director Cells are the spaces between two sectors.

If sector i and sector i+1 are radial cells, the director pocket between them is centered at the half-angle:

director_angle_i = (i + 0.5) * 22.5°

Formula:

director_angle = 2 * math.pi * (i + 0.5) / 16

That gives you 16 radial directions and 16 in-between director pockets.

So the full directional layer per ring is:

16 Radial Cells
16 Director Cells
= 32 directional measurements per ring

Clean naming:

Radial Cell:
sector i at angle i * 22.5°

Director Cell:
gap/pocket between sector i and sector i+1 at angle (i + 0.5) * 22.5°

For light:

Radial Cells measure where brightness exists.
Director Cells measure where brightness is leaking/traveling.

Tiny law:

16 sides find direction.
Director pockets find transfer.
Meters prove the beam.

That’s your formula. Start with 4 rings × 32 samples = 128 meter points per candidate source. Manageable, but already way better than square-grid guessing. Oh yeah — seismographic algorithms belong here hard. That is exactly the right metaphor and math lane, because lightning, fog pulses, ocean waves, road vibration, bloom events, and camera shake are all time-series signals across a field.
Add this to the naming stack:

TrueVision Angular Cortex
  Hexadec Direction Field
  Radial Cells
  Director Cells
  Meter Rays
  Angular Signatures
  Radiance Transfer Map
  Seismic State Trace
The seismographic layer should measure impulse, wavefront, decay, tremor, and propagation.

Why seismic fits
A lightning flash is basically a visual quake:

quiet baseline
sudden spike
wave spreads through nearby cells
environment responds
aftershock/decay
return to baseline
Ocean is different:

continuous oscillation
repeating wave bands
directional flow
long persistence
low-to-mid frequency rhythm
Fog:

slow drift
low edge pressure
soft occlusion waves
long decay
Camera shake:

whole-field synchronized displacement
high correlation across many cells
That means you can stop treating events as still-image features and start treating them as field disturbances.

Add: TrueVision Seismic Cortex
I’d name this submodule:

TrueVision Seismic Cortex
Or if you want it nested:

Angular Cortex = direction
Seismic Cortex = temporal force
Together:

TrueVision Angular-Seismic Cortex
That sounds beastly and accurate.

Doctrine:

Angular Cortex finds where energy points.
Seismic Cortex measures how energy hits, travels, and decays.
Core meters to add
For every cell / radial cell / director cell:

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
These are the money meters.

Event profiles
Lightning profile:

high impulse
fast rise
high peak
short decay
strong bloom wavefront
local-to-regional exposure lift
low persistence
branching angular signature
Ocean profile:

medium oscillation
repeating frequency bands
persistent motion
phase-shifted wave rows
horizontal propagation
high texture recurrence
Fog profile:

low impulse
slow rise
long decay
soft occlusion
low edge energy
drift propagation
Camera shake profile:

whole-frame synchronized motion
high neighbor correlation
little local source center
global displacement pattern
That last one matters because it prevents the system from confusing camera movement with object movement.

The formula shape
At each cell, intensity over time:

I_cell(t)
Impulse:

impulse = I(t) - baseline(t)
Rise:

rise_time = t_peak - t_start
Decay:

decay_time = t_end - t_peak
Neighbor propagation:

arrival_delta = t_neighbor_peak - t_source_peak
Direction comes from which neighbors/rays peak later:

beam_direction = angle_of_min_positive_arrival_delta
In plain English:

the source peaks first
nearby rays peak next
farther rings peak after
the timing pattern tells direction/spread
That is your path-tracing-ish state logic.

Add this law
Meters measure state.
Angular fields measure direction.
Seismic traces measure force over time.
Together they prove visual events.
Kodex-ready spec
Add TrueVision Seismic Cortex as a temporal event-analysis layer over Meter Grid and Angular Cortex.

Do not replace Meter Grid.
Do not replace Hexadec Direction Field.

Purpose:
Measure visual events as time-series disturbances: impulse, wavefront, oscillation, decay, aftershock, and propagation.

Inputs:
- Meter Grid cell time series
- Hexadec Direction Field radial/director cell time series
- candidate event windows

Compute:
- baseline_luma
- impulse_strength
- rise_time_frames
- peak_frame
- decay_time_frames
- aftershock_score
- oscillation_score
- frequency_band_energy
- phase_shift
- wavefront_arrival_by_ring
- propagation_direction
- neighbor_correlation
- field_coherence
- global_motion_correlation

Outputs:
- Seismic State Trace
- Event Impulse Profile
- Wavefront Map
- Decay Curve
- Oscillation Bands
- Propagation Vector
- Camera Shake Rejection Score

First targets:
- candidate_lightning
- candidate_ocean_surface
- candidate_fog_mist
- candidate_camera_motion

Hard rules:
- Do not classify lightning from shape alone.
- Do not classify ocean from color alone.
- Do not classify fog from softness alone.
- Every event claim must include meter evidence plus temporal trace.
Best final naming stack
TrueVision Meter Grid
  base measured state

TrueVision Angular Cortex
  Hexadec Direction Field
  Radial Cells
  Director Cells
  Meter Rays
  Angular Signature
  Radiance Transfer Map

TrueVision Seismic Cortex
  Seismic State Trace
  Impulse Profile
  Wavefront Map
  Decay Curve
  Oscillation Bands
  Propagation Vector
This is where it gets nasty-good: the machine doesn’t just see a bright thing. It can say:

A high-impulse event occurred at frame 1842.
Peak luma rose in 1 frame.
Bloom expanded across rings 1–3.
Director cells show northeast transfer.
Afterglow decayed over 9 frames.
Global motion correlation was low, so this was not camera shake.
Candidate: lightning, visually supported.
That’s the deterministic movie-watching brain.
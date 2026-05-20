# TrueVision State Generation Primitives

This document distills the useful ideas from the video-generation terminology
research into TrueVision-native language.

It is not a vendor map.
It is not a dependency plan.
It is not permission to add third-party apps.

The goal is to borrow useful shapes and turn them into local, deterministic
audio/video state-generation machinery.

## Hard Boundary

```text
No third-party apps.
No cloud dependency.
No general desktop automation.
No evidence claims for generated media.
No prompt-to-magic-video authority.
```

Allowed:

```text
local code
local media files
local manifests
local state tensors
local templates
local captures
local deterministic renderers
local model adapters behind validation
```

## TrueVision Mental Model

Research phrase:

```text
video = trajectory through a spatiotemporal manifold
```

TrueVision phrase:

```text
video = causally ordered state over time
```

Operational form:

```text
state[t][y][x][channel]
```

The current TrueVision lane uses addressed cell state instead of raw pixels as
the main working representation.

## Representation Layer

Useful borrowed terms:

```text
latent space
spatiotemporal tokens
factorized representation
persistent state
layered representation
```

TrueVision fit:

```text
latent space -> compact cell-state channels
spatiotemporal tokens -> addressed cells across time
factorized representation -> split spatial channels from temporal channels
persistent state -> object/material/light identity carried over frames
layered representation -> background, fog, smoke, light, subject, effects
```

Current state channels:

```text
rgb
luma
saturation
edge density
motion energy
delta luma
texture energy
```

Next state channels:

```text
depth estimate
alpha / matte
fog density
light pressure
material roughness
specular pressure
motion vector x/y
occlusion confidence
identity persistence
camera motion
```

## Conditioning Layer

Research phrase:

```text
conditioning controls generation
```

TrueVision phrase:

```text
conditioning biases state evolution
```

Allowed conditioners:

```text
audio level
audio peaks
audio valleys
beat/onset events
lyrics/story beats
human prompt
time-marker notes
TrueVision capture signatures
scene templates
camera templates
material templates
```

Not trusted directly:

```text
model prose
prompt claims
unvalidated JSON
generated evidence claims
```

Required route:

```text
human intent
-> adapter contract
-> draft state JSON
-> schema validation
-> repair loop if needed
-> renderer handoff
-> manifest
```

## Control Layer

Research phrase:

```text
control steers the generative trajectory
```

TrueVision phrase:

```text
control changes state pressure over time
```

Control surfaces:

```text
semantic control -> theme, scene, lyric meaning
spatial control -> layout, masks, regions, depth, edge maps
temporal control -> timing, beats, motion, camera, transitions
identity control -> persistent subject, object, material, motif
lighting control -> exposure, glow, flash, shadow, color pressure
material control -> smoke, fog, water, glass, metal, skin, cloth
```

TrueVision control record:

```json
{
  "time_seconds": 72.0,
  "target": "fog_density",
  "direction": "increase",
  "strength": 0.35,
  "source": "operator_recalibration",
  "confidence": 0.8
}
```

Control is scheduled, not static. A render may need strong structure early,
soft detail later, and beat-driven lighting only at peaks.

## Temporal Coherence Layer

Research phrase:

```text
temporal coherence is the hard problem
```

TrueVision phrase:

```text
state must remember itself
```

Mechanisms that fit:

```text
6-1-6 temporal neighborhood
previous-state propagation
future-anchor checking
motion vector estimation
noise-field reuse
camera-motion continuity
identity persistence channels
object permanence checks
flicker scoring
texture-swim scoring
```

6-1-6 working rule:

```text
Use 6 prior frames and 6 future frames around a target state.
Interpolate or project from causal history first.
Only fill creatively when the temporal cloud cannot explain the gap.
Log what was inferred.
```

## Motion Layer

Research terms:

```text
optical flow
scene flow
motion vectors
trajectory embeddings
camera intrinsics/extrinsics
```

TrueVision fit:

```text
optical flow -> cell displacement x/y
scene flow -> future 3D/depth-aware displacement
motion vectors -> per-cell motion channels
trajectory embeddings -> named motion templates
camera model -> camera state path
```

Immediate local version:

```text
cell motion x
cell motion y
cell motion confidence
camera pan/tilt/zoom estimate
regional drift
beat-driven motion pressure
```

## Rendering Layer

Research terms:

```text
volumetric rendering
ray marching
layered compositing
alpha fields
neural rendering
inverse rendering
```

TrueVision fit:

```text
volumetric rendering -> local fog/smoke density field
ray marching -> optional local renderer path for depth/fog/light
layered compositing -> separate state layers before final video
alpha fields -> soft masks for fog, smoke, light, silhouettes
inverse rendering -> capture-derived signature extraction
neural rendering -> deferred, only if local and validated
```

No third-party app is required to borrow these shapes.

## Signature Layer

TrueVision signatures are reusable state behaviors extracted from observed
captures.

Current signature types:

```text
motion/look profile
camera shake profile
edge density profile
contrast/color profile
energy timing profile
cut rhythm profile
peak/lighting hot-cell signature
```

Needed signature types:

```text
fog drift signature
smoke curl signature
rain streak signature
water shimmer signature
lightning branch signature
fire glow signature
handheld camera signature
slow cinematic push-in signature
```

Current problem:

```text
peak/lighting signatures can become white blobs
```

Correction:

```text
hot cells
-> connected components
-> branch/skeleton extraction
-> edge-direction weighting
-> density rejection for filled blobs
-> thin glow render
```

## Failure Modes

These are not aesthetic complaints. They are state failures.

```text
flicker -> temporal state mismatch
identity drift -> persistence failure
texture swimming -> material coordinates not stable
object popping -> object permanence failure
blob flash -> area fill where branch/surface structure was needed
circle smoke -> primitive rendering instead of density-field rendering
triangle fog -> geometry artifact where volumetric softness was required
over-saturation -> color pressure too high
blur collapse -> missing high-frequency/detail channels
stochastic looping -> repeated pattern without causal variation
conditioning entanglement -> one control corrupts another
```

## Evaluation Layer

Local metrics that fit TrueVision:

```text
temporal continuity score
flicker score
motion-vector agreement
edge continuity score
color-pressure stability
audio-sync score
signature-use score
blob/artifact score
template compliance score
manifest completeness score
```

Avoid pretending external benchmark names prove local quality. We can borrow
the idea of measuring spatial and temporal consistency, but the lab needs its
own practical scoring.

## TrueVision Pipeline Shape

```text
capture or prompt
-> state/template representation
-> conditioning signals
-> control schedule
-> temporal coherence pass
-> renderer/compositor
-> post consistency pass
-> manifest/report/receipt
-> learning record
```

## What We Borrow, What We Do Not

Borrow:

```text
representation/control/coherence concepts
motion-field thinking
layered compositing thinking
volumetric density thinking
failure-mode names
evaluation categories
```

Do not borrow:

```text
cloud dependence
external app workflows
unvalidated model authority
vendor-specific pipelines
black-box evidence claims
generic desktop automation
```

## Immediate Implementation Targets

```text
1. Add motion vector x/y channels to capture or post-capture mapper.
2. Add fog/smoke density-field renderer, replacing circle/triangle primitives.
3. Add branch/skeleton extraction for lightning signatures.
4. Add control schedule JSON for time-varying render pressure.
5. Add artifact scoring: blob, flicker, oversaturation, primitive smoke.
6. Add signature library terms for fog, smoke, lighting, water, camera.
7. Add template fields for layered representation and temporal coherence.
```

## Working Language

Preferred TrueVision terms:

```text
state channel
temporal cloud
control pressure
state signature
motion field
density field
layer mask
coherence pass
artifact guard
render manifest
learning record
```

This is the vocabulary lane for future TrueVision state generation work.

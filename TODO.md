# TrueVision Generation Lab TODO

This project is now a standalone generation lab, separate from SecureCore and AnchorWorks.

Core split:

```text
SecureCore = guard, log, approve, preserve.
AnchorWorks = shape/count/routing language.
TrueVision Generation Lab = state-media capture, replay, generation, and visual-language experiments.
```

Hard boundary:

```text
Forward TrueVision witnesses.
Reverse TrueVision replays or demonstrates state.
Generated state media is synthetic, not evidence.
Raw frames are not implied unless explicitly saved.
```

## Base Build Status

Done:

```text
standalone project root at D:\TrueVision_Generation_Lab
static studio HTML copied into ui/
state recorder/replay/generator scripts copied into scripts/
path tracing lane copied into scripts/
still image state capture copied into scripts/
grid mapper and resonance state copied into modules/root
unit tests copied into tests/
region snip tool added as scripts/truevision_region_snip.py
Rust compiled capture lane documented in docs/
outputs, presets, and connected_artifacts isolated/ignored by default
```

Do not let this repo become another all-purpose system. It exists to prove the visual-state language.

## Immediate Verification

- [ ] Run all standalone tests:

```powershell
cd D:\TrueVision_Generation_Lab
$env:PYTHONPATH="D:\TrueVision_Generation_Lab\scripts;D:\TrueVision_Generation_Lab\modules;D:\TrueVision_Generation_Lab"
python -m unittest discover -s tests -v
```

- [ ] Run the snip tool dry path:

```powershell
python scripts\truevision_region_snip.py `
  --region 640,360,1280,720 `
  --preset-id center_video `
  --print-command
```

- [ ] Open the static studio:

```powershell
Invoke-Item D:\TrueVision_Generation_Lab\ui\truevision_state_media_studio.html
```

## Snip Tool Base

Purpose:

```text
Let the operator select a screen/video area once.
Snap that region into TrueVision's 16:9 grid discipline.
Save a preset.
Feed the existing recorder without inventing a new capture system.
```

Rules:

```text
rough selection -> 16:9 snapped region -> clamped bounds -> preset hash -> recorder command
```

Next:

- [ ] Test interactive selector on the real desktop.
- [ ] Add multi-monitor metadata to preset reports.
- [ ] Add visible overlay preview before capture.
- [ ] Add preset list/delete/rename commands.
- [ ] Add "watch this window" mode using top-level window bounds later.
- [ ] Keep raw frame saving off unless explicitly requested.

## Focus Reconstruction

The data-only photo reconstruction proved the shape but came out blurry. That is expected with a coarse cell tensor.

Next focus stack:

```text
Lanczos state upsample
CLAHE/local luma contrast
edge-gated unsharp mask
bilateral cleanup
chroma-safe saturation restore
cell-boundary deblocking
optional edge-directed interpolation
```

Rules:

```text
Input must be stored state only.
Do not read the source photo/video again.
Report every filter, parameter, hash, and output.
Never call it original reconstruction.
```

Target script:

```text
scripts/truevision_focus_reconstruct.py
```

## Prompt-To-State Compiler

This is the bridge from human language to state media.

Contract:

```text
Input: human visual prompt
Output: state transition JSON
Then pass JSON to existing generator/replay lanes.
```

Do not:

```text
build a new renderer
invent a new UI
rewrite TrueVision
let prompt text bypass state contracts
```

First language objects:

```text
scene
camera
entity
material
light
motion
force
constraint
beat
transition
style fingerprint
render budget
artifact manifest
```

## Visual Math Stack

Bring in the math as state language, not as vague effects.

Needed lanes:

```text
geometry: transforms, projection, primitive composition
trigonometry: oscillation, gait, wave, pulse, orbit
linear algebra: matrices, basis vectors, camera transforms
physics: velocity, acceleration, spring, drag, collision hints
electronics/signal: noise, waveform, scanline, pulse, resonance
computer vision: edges, gradients, motion, optical flow, segmentation hints
path tracing: rays, bounces, materials, lighting samples
```

Principle:

```text
Math should produce addressable state, not decorative noise.
```

## Learning Twin

The learning system needs two coordinated halves:

```text
Observer twin:
  studies real captures
  records what channels matter
  logs failures and successes

Renderer twin:
  consumes lessons
  updates state formulas
  improves replay/generation from known contracts
```

Every lesson needs:

```text
input artifact hash
state artifact hash
render output hash
what improved
what failed
which channels mattered
next rule proposal
```

## Rust / Compiled Lane

Rust is worth it, but only where Python is wasting time.

First compiled target:

```text
truevision-capture-worker.exe
```

Responsibilities:

```text
screen/window region capture
frame timing
ring buffer
chunk writing
hashing
minimal IPC status
```

Keep in Python for now:

```text
state language
prompt-to-state compiler
renderer experiments
reports
learning rules
```

Rust comes after:

```text
region preset contract is stable
state tensor schema is stable
chunk manifest schema is stable
writer handoff is stable
```

## Parking Lot

- [ ] Add one-hour capture profile math.
- [ ] Estimate disk use by fps/grid/channel/compression settings.
- [ ] Add sample capture corpus with small fixtures only.
- [ ] Add style/videograph fingerprint schema.
- [ ] Add image/pictorial generation language lane.
- [ ] Add local LLM adapter only after strict JSON schema validation exists.
- [ ] Add import/export contract for SecureCore artifact engine.
- [ ] Add import/export contract for AnchorWorks route/count maps.

Tiny law:

```text
Capture teaches.
State remembers.
Math shapes.
Renderer demonstrates.
SecureCore preserves.
AnchorWorks routes.
```

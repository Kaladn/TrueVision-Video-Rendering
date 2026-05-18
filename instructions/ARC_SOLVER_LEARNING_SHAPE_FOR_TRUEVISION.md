# ARC Solver Learning Shape For TrueVision

## Purpose

This document brings the ARC Solver learning capability into the TrueVision state-media POC without copying ARC workers directly.

We want the shape of the engine:

```text
metadata extraction
channel encoding
resonance compression
example fusion
recognition field
rule vector
candidate generation
success/failure scoring
regeneration
```

We do not want an uncontrolled pile of new workers.

## Core Translation

ARC Solver works because it treats a grid cell as a structured entity, not just a color.

TrueVision needs the same move:

```text
video pixel/cell -> structured visual state entity
frame -> state grid
clip -> temporal state sequence
scene -> persistent entity graph
render -> rule/vector application over time
```

## ARC To TrueVision Mapping

| ARC concept | TrueVision state-media concept |
| --- | --- |
| ARC grid | video frame state grid |
| pixel metadata | cell metadata |
| input/output examples | capture/replay or formula/render examples |
| 6-channel sensory organ | visual cell feature bundle |
| 20-dim resonance state | frame/clip resonance vector |
| multi-example fusion | run/capture/failure fusion |
| recognition field | scene/action/failure classifier |
| rule vector | state transformation vector |
| two-attempt sampler | candidate A/B regeneration |
| submission builder | artifact/manifest builder |

## Required TrueVision Cell Metadata

A cell is not only RGB.

Minimum fields:

```text
cell_id
row
col
norm_x
norm_y
source_frame_index
source_time_sec
mean_r
mean_g
mean_b
luma
contrast
edge_density
motion_delta
dominant_hue
saturation
value
object_id
depth_rank
occlusion_state
material_hint
lighting_hint
confidence
```

Not every field has to be perfect in v1, but the schema needs room for them. The renderer cannot generate detail from channels it never receives.

## 6-Channel Base For TrueVision

The first TrueVision organ can use six broad channels:

```text
CH1 color_luma
CH2 edge_contrast
CH3 motion_delta
CH4 object_entity
CH5 depth_occlusion
CH6 material_lighting
```

Expanded internal fields can still feed those six lanes. The six channels are the stable interface; the subfeatures are implementation detail.

## 20-Dimensional TrueVision Resonance

A clip/frame resonance vector should summarize what matters without flattening the scene into mush.

Suggested dimensions:

```text
D00 active_visual_mass
D01 color_entropy
D02 mean_luma
D03 luma_variance
D04 edge_density_mean
D05 edge_density_variance
D06 motion_mass
D07 motion_direction_consistency
D08 object_count_norm
D09 largest_object_ratio
D10 object_persistence_score
D11 object_fragmentation_score
D12 depth_layer_count
D13 occlusion_change_rate
D14 horizon_or_major_axis_stability
D15 camera_motion_estimate
D16 material_variance
D17 lighting_gradient_strength
D18 temporal_continuity_score
D19 state_schema_completeness
```

This vector is not the media. It is the compressed reasoning layer used to compare attempts.

## Fusion

Fusion is how the engine learns without pretending one prompt is enough.

Inputs:

```text
successful runs
failed runs
captured TrueVision records
operator acceptance notes
objective metric reports
```

Output:

```text
stable invariants
unstable dimensions
missing channels
candidate changes
```

Example:

```text
If three failed walking-person clips all have poor object_persistence_score
and successful captured video has high persistence, the next run must
increase object_id continuity and limb-phase continuity before adding style.
```

## Recognition Field

The recognition field classifies what kind of transformation or failure we are dealing with.

Scene/action families:

```text
STATIC_SCENE
CAMERA_PAN
CAMERA_PUSH
OBJECT_TRANSLATION
OBJECT_ROTATION
ARTICULATED_WALK
FACE_TALKING
GLITCH_EVENT
IMPACT_PULSE
LIGHTING_TRANSITION
ENVIRONMENTAL_MOTION
```

Failure families:

```text
LOW_GRANULARITY
ENTITY_DRIFT
DEPTH_COLLAPSE
OCCLUSION_COLLAPSE
MATERIAL_COLLAPSE
LIGHTING_FLATNESS
MOTION_ALIASING
GEOMETRY_UNDERMODELED
STATE_RENDER_MISMATCH
PROMPT_LANGUAGE_LEAK
```

The LLM may suggest a family, but the local scorer should decide from metrics where possible.

## Rule Vector

A rule vector is the executable shape of an idea.

Example:

```json
{
  "family": "ARTICULATED_WALK",
  "duration_seconds": 5,
  "fps": 9,
  "grid": [90, 160],
  "entities": {
    "person_001": {
      "path": "left_to_right",
      "speed_norm": 0.05,
      "stride_phase_hz": 1.7,
      "limb_visibility": 0.8,
      "object_persistence": 1.0
    }
  },
  "environment": {
    "field_depth_layers": 4,
    "horizon_y": 0.43,
    "wind_motion": 0.15
  },
  "renderer_weights": {
    "geometry": 1.0,
    "edge_density": 0.8,
    "material": 0.6,
    "lighting": 0.7,
    "motion_delta": 1.0
  }
}
```

## Success/Failure Regeneration Loop

Use this loop for every serious improvement run:

```text
1. Build candidate state formula.
2. Render or replay candidate.
3. Extract TrueVision state from output.
4. Compare candidate state to intended state.
5. Score objective metrics.
6. Label success signatures.
7. Label failure signatures.
8. Fuse signatures with prior runs.
9. Adjust rule vector.
10. Generate candidate A and candidate B.
11. Keep both records even when both fail.
```

The goal is not to make the LLM more dramatic. The goal is to make the next state formula more correct.

## Regeneration Pseudocode

```python
def regenerate_from_attempts(intent, prior_runs):
    records = load_attempt_records(prior_runs)
    successes = [r for r in records if r.score.operator_acceptance >= 0.7]
    failures = [r for r in records if r.score.operator_acceptance < 0.7]

    success_invariants = fuse_invariants(successes)
    failure_deltas = classify_failure_deltas(failures)

    base_vector = build_rule_vector(intent, success_invariants)
    next_vector = adjust_rule_vector(base_vector, failure_deltas)

    candidate_a = next_vector
    candidate_b = alternate_interpretation(next_vector, failure_deltas)

    return candidate_a, candidate_b
```

## Candidate A/B Policy

Candidate A should be the best current interpretation.

Candidate B should change one important ambiguity, not everything:

```text
if motion is wrong, vary motion phase
if geometry is wrong, vary entity geometry
if depth is wrong, vary depth/occlusion
if material is wrong, vary material channels
if renderer ignores state, vary renderer weights
```

This preserves learnability. If every knob changes at once, the failure record becomes muddy.

## Ledger Record

Each attempt needs a record:

```json
{
  "attempt_id": "string",
  "parent_attempt_id": "string_or_null",
  "intent_hash": "sha256:...",
  "state_formula_hash": "sha256:...",
  "output_hash": "sha256:...",
  "capture_shape": {
    "fps": 9,
    "grid_rows": 90,
    "grid_cols": 160,
    "feature_count": 16
  },
  "objective_scores": {
    "schema_completeness": 1.0,
    "object_persistence": 0.0,
    "temporal_continuity": 0.0,
    "geometry_consistency": 0.0,
    "operator_acceptance": 0.0
  },
  "success_signatures": [],
  "failure_signatures": [],
  "next_vector_changes": []
}
```

## What To Preserve From ARC

Preserve:

```text
structured entity cells
fixed sensory lanes
compressed resonance state
multi-example fusion
stability/variance reasoning
recognition-field classification
rule vectors
two-attempt ambiguity handling
failure inspection loop
```

Do not blindly preserve:

```text
ARC color-only assumptions
ARC fixed 0-9 palette
ARC task/submission format
ARC workers as runtime authority
ARC competition-specific scoring
```

## Immediate TrueVision Learning Priorities

1. Replace flat RGB-only thinking with cell metadata.
2. Track object identity across frames.
3. Add depth and occlusion sidecars.
4. Add geometry sidecars for people, faces, fields, grids, and camera paths.
5. Score renderer consumption: how much of the available state actually affects output.
6. Keep success/failure ledgers hash-addressed.
7. Generate A/B candidates from one controlled difference.

## Tiny Law

```text
ARC proved the shape.
TrueVision gives the shape motion.
Failure records teach the next rule vector.
Success records tell us what not to break.
```

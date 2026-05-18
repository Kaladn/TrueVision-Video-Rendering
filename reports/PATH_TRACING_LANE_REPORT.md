# TrueVision Path-Tracing Lane Report

## Purpose

The path-tracing lane adds a physically grounded synthetic renderer to the TrueVision state-media POC.

It does not replace TrueVision capture. It gives Reverse TrueVision a stronger renderer that consumes explicit geometry, material, light, camera, and math state before writing the same 16:9 cell-state records used by the rest of the package.

## Boundary

```text
Forward TrueVision witnesses.
Reverse TrueVision replays or demonstrates.
Path tracing improves synthetic rendering state.
Generated media is synthetic, not evidence.
```

## Current Lane

```text
state formula
-> CPU path tracer
-> RGB frame
-> TrueVision cell-state tensor
-> JSONL records + NPZ chunks
-> optional replay video
-> manifest/report
```

The first renderer is intentionally small and deterministic:

```text
pinhole camera
ground plane
animated sphere
sky gradient
directional sun proxy
diffuse/specular material mix
shadow rays
seeded cosine bounce samples
```

## Why This Matters

The earlier renderer could describe motion and color, but it did not speak enough visual physics.

Path tracing gives the state language stronger lanes:

```text
geometry
light transport
shadow
material response
camera rays
surface normal
depth relationship
```

That is the first step toward the richer language needed for believable generated state media.

## Future Full-Power Constraint

The later one-hour full-power capture path must not become loose file spam.

Required future shape:

```text
coordinated temporal writer
append-only causality records
chunk manifests
hash receipts
replay checkpoints
suppression/error receipts
learning ledger
rendering ledger
```

Long captures should write through a coordinated temporal-causality lane so each artifact has:

```text
event_id
sequence
observed_at_utc
writer_id
chunk_id
payload_hash
previous_event_hash
causal_parents
artifact_refs
```

## Twin Learning System

The learning system needs an opposite twin.

```text
Learning twin:
  reads capture/replay/generation outcomes
  scores failures and successes
  extracts reusable state-language corrections
  writes teaching records

Rendering twin:
  consumes teaching records
  updates explicit state formulas
  renders through deterministic engines
  reports which state channels were actually used
```

Tiny law:

```text
The learner teaches.
The renderer proves what it used.
The temporal log proves what happened first.
```

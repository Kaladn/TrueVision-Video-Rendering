# TrueVision Receipt And Manifest Rules

## Purpose

Keep local TrueVision runs reproducible and reviewable.

## Manifest Required When

A manifest is required for:

- capture
- replay
- render
- profile extraction
- meter grid generation
- state focus/lens runs
- driving/high-speed awareness runs
- any run that writes artifacts under `storage/` or `outputs/`

## Receipt Required When

A receipt is required when:

- a tool claims a source was processed
- bulky source/teacher chunks are purged
- a profile is promoted
- a safety or retention decision is made
- a render is used as a proof object

## Timing Rule

```text
Frame index and FPS are the clock.
Wall time is performance, not timeline truth.
```

Full-frame downstream tooling requires:

```text
state_log_every = 1
```

Sampled logs are allowed only when marked as sampled and not promoted as frame-exact.

## Source Boundary

Manifests must state whether source frames, raw audio, or raw video are retained.

Default:

```text
raw frames retained: false
raw video retained: false
compact state retained: true
manifest retained: true
receipt retained: true
```

## Hash Rule

When possible, manifests should record:

```text
source hash
output hash
frame-state hash
manifest hash
receipt hash
```

## Success Claim Rule

No tool should claim success unless:

```text
manifest exists
required output exists
timeline is valid for its declared mode
receipt exists when a write/purge/promotion occurred
known limitations are written
```

# TrueVision Generation Operational Truth Lock

- Created: 2026-05-24
- Authority: code, tests, receipts, and this document outrank older planning docs when they conflict.
- Scope: TrueVision Generation only.

## Core Law

```text
Record state.
Plan state.
Transform state.
Render pixels last.
Prove every run.
```

## Stack Boundary

```text
TrueVision is a local usable stack, not a platform backend.
Generation is an output lane, not the whole purpose.
New ideas enter as bounded workers with contracts, tests, manifests, and receipts.
No worker becomes a platform until local repeatability proves it needs one.
```

The stack exists to make media state usable by later tooling:

```text
capture/log
-> meter
-> profile
-> transform
-> validate
-> render or export
-> receipt
```

Platform-style plans, account systems, remote orchestration, and browser-control
detours stay out unless a proven local worker cannot operate without them.

## Current Live Abilities

```text
Rust native capture emits .tvcells state chunks
TrueFrameGen streaming renderer exists
SegmentField A-to-B transition method exists
state-media rendering lanes exist
audio feature extraction tools exist
TrueAudio analysis state exists
TrueAudio replayable spectral state exists
TrueSpeech speech/background timing exists
lyrics/script alignment candidates exist
AV-only tool registry exists
local studio/server exists
generated media remains out of git
```

Timing lock:

```text
Frame index and FPS are the clock.
Wall time is performance, not timeline truth.
Full-frame downstream tooling requires state_log_every = 1.
Sampled logs may be exact, but they are not full-frame truth.
```

## Parked Or Separate

```text
TrueAudio stays here for now.
AnchorWorks may consume symbols/state later, not raw audio.
SecureCore may gate retention later, not own audio replay.
UI product work is parked.
Generated videos/audio/state artifacts are local outputs, not repo truth.
Platform backend work is parked.
```

## Current Staged Work

```text
voice_state_v1 format
canonical audio feature contract
FFmpeg discovery normalization
script/lyrics manual correction path
audio-to-video sync contract
state-media element learning intake
```

## Boundary With AnchorWorks

TrueVision Generation can teach AnchorWorks method discipline:

```text
state first
render later
manifest every run
receipt every write
do not claim what state does not prove
```

It does not become AnchorWorks intake authority directly.

## Boundary With SecureCore

TrueVision Generation is not a security logger.

```text
SecureCore logs security/ops.
TrueVision Generation records and renders media state.
Cross-system use must pass through validated state packets.
```

## Future Ability Area

```text
product-grade studio UI
long-run chunk rendering
selected-window capture
GPU capture/render acceleration beyond encode
voice timing editor
manual correction UI
cross-system harness proof
```

## Tiny Lock

```text
State is the source.
Pixels are the last mile.
Receipts separate proof from excitement.
```

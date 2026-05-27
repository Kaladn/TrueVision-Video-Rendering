# TrueVision Local Product Map

TrueVision is a local state-machine tool stack.

It is not the main system, not a backend platform, and not the operator brain.

```text
TrueVision records state.
AnchorWorks interprets meaning.
SecureCore proves safety.
```

## System Hierarchy

```text
AnchorWorks
  main face
  language/counts/reasoning
  operator surface
  meaning and review

SecureCore
  guardrail
  policy gate
  receipt layer
  runtime safety wrapper
  verification boundary

TrueVision
  state capture
  media-state tools
  render/replay lab
  measurement workers
  manifests and receipts
```

Worker contracts are locked in:

```text
docs/TRUEVISION_WORKER_RACK_CONTRACT.md
```

## Current Job

TrueVision answers this question:

```text
What was the machine-state shape at this moment?
```

It may render, replay, or generate as a proof surface, but generation is not the center.

## Repository Law

```text
Code repos hold machinery.
Runtime storage holds state.
Outputs are artifacts.
Receipts prove what happened.
```

## Active Local Stack

```text
truevision_runtime/
  core state, storage, AV tools, learning intake, templates, receipts

trueaudio_runtime/
  decoded audio-state logging, replay, speech/lyrics timing helpers

trueframegen/
  derived frame/state interpolation and temporal projection

native/truevision_capture_rs/
  Rust hot paths for capture/render lanes

scripts/
  operator CLI entrypoints

storage/
  ignored local runtime lanes with placeholder directories

outputs/
  ignored generated proof and render artifacts
```

## Dev Preflight

Run:

```powershell
python scripts\truevision_preflight.py
```

For machine-readable output:

```powershell
python scripts\truevision_preflight.py --json
```

Preflight checks prerequisites and repo health. It does not install, patch, open browsers, or mutate runtime state.

## Fresh Runtime Data Shape

A future empty runtime data root should stay outside code repos:

```text
D:\CortexEvolved_RuntimeData\
  AnchorWorks\
    maps\
    counts\
    chat_shadow\
    receipts\

  SecureCore\
    logs\
    forge\
    fusion\
    policy_receipts\
    runtime_receipts\

  TrueVision\
    state_chunks\
    manifests\
    receipts\
    previews\
```

No system should hardcode another repo path.

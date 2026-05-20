# TrueVision Native Power Policy

TrueVision generation work must use native/compiled lanes for heavy compute.

## Rule

```text
Python may orchestrate.
Python may validate schemas.
Python may write manifests, receipts, and reports.
Python may run small tests and prototypes.

Python must not own sustained capture loops.
Python must not own full-length frame generation.
Python must not own high-resolution pixel transforms.
Python must not own final full-quality render loops.
```

## Required Native Lanes

```text
screen capture -> Rust/native
cell-state extraction -> Rust/native
frame generation -> Rust/native
high-resolution transforms -> Rust/native or GPU
encoding -> GPU encoder when available
```

## GPU/Encode Targets

Preferred ffmpeg acceleration families when available:

```text
qsv
amf
d3d11va
d3d12va
opencl
vulkan
```

Preferred generation/encode route:

```text
Rust TFG renderer -> hardware encoder -> MP4
```

Fallback route:

```text
Rust TFG renderer -> ffmpeg libx264 -> MP4
```

## Python Quarantine

Python files in this project are allowed only as:

```text
CLI wrappers
schema validators
tool policy gates
manifest/report writers
small previews
unit tests
local LLM adapter glue
```

Any Python path that processes sustained frame streams must be treated as
temporary and replaced with Rust/native before full-length use.

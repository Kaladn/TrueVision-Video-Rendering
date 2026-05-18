# Rust / Compiled Capture Lane

Rust is the right direction for the hot parts, but not before the contracts settle.

## Use Rust For

```text
screen/window region selection
high-rate frame acquisition
fixed-size ring buffers
background NPZ or binary chunk writing
hash-chain receipts
retention spool control
native tray/CLI shell
```

## Keep In Python For Now

```text
state language experiments
renderer experiments
focus reconstruction experiments
prompt-to-state compiler prototypes
test fixtures
reports
```

## First Rust Boundary

The first compiled component should be a capture worker, not a generator:

```text
truevision-capture-worker.exe
  input: region preset JSON
  output: frame/state chunks + receipts
  no prompt handling
  no evidence claims
  no policy authority
```

## Contract

```text
Python defines state schema.
Rust accelerates capture and writing.
Renderer consumes the same files either way.
```

Tiny law:

```text
Rust makes the pipeline fast.
It does not make the pipeline true.
The manifest and state contract make it true.
```

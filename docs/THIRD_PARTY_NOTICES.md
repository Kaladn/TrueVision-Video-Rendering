# Third-Party Notices And Credits

This project contains in-house TrueVision code and uses several third-party
tools/libraries from the local development environment.

This file is a practical attribution list for the repository. It is not legal
advice. Before any public binary release, package distribution, or commercial
release, re-check every dependency license and include the exact upstream
license text required by that distribution path.

## In-House / Project Code

The following repository areas are TrueVision Generation Lab work unless a file
states otherwise:

```text
README.md
TODO.md
docs/
instructions/
modules/
native/truevision_capture_rs/
presets/
scripts/
templates/
tests/
trueframegen/
truevision_runtime/
ui/
screen_resonance_state.py
```

Current project license status:

```text
No repository license has been selected yet.
```

Until a license is added, do not assume the in-house code is open source.

## Direct Python Dependencies

Declared in:

```text
pyproject.toml
```

### NumPy

Use in this repo:

```text
array math
cell-state tensors
audio feature arrays
image/frame buffers
temporal reconstruction data
```

Credit:

```text
NumPy project and contributors
https://numpy.org/
```

Observed local package:

```text
numpy 2.4.4
```

License note:

NumPy is commonly distributed under the BSD 3-Clause license. Confirm the exact
license text from the installed package or upstream before redistribution.

### OpenCV / opencv-python

Use in this repo:

```text
image loading/saving
color conversion
resize/interpolation
blur/filter operations
edge operations
prototype rendering helpers
```

Credit:

```text
OpenCV project and opencv-python maintainers
https://opencv.org/
https://github.com/opencv/opencv-python
```

Observed local package:

```text
opencv-python 4.13.0.92
license metadata: Apache 2.0
```

License note:

The OpenCV library and opencv-python package have their own license terms. Keep
their notices if packaging or redistributing binaries.

### MSS

Use in this repo:

```text
Python screen capture prototypes
screen region capture support
```

Credit:

```text
MSS by Mickael Schoentgen / Tiger-222
https://github.com/BoboTiG/python-mss
```

Observed local package:

```text
mss 10.1.0
license metadata: MIT License
copyright: 2013-2025, Mickael 'Tiger-222' Schoentgen
```

License note:

The MIT notice must be retained in substantial copies of the software.

## Build-Time Python Packaging

### setuptools

Use in this repo:

```text
Python editable install and packaging backend
```

Credit:

```text
Python Packaging Authority / setuptools maintainers
https://github.com/pypa/setuptools
```

Observed local package:

```text
setuptools 65.5.0
```

### wheel

Use in this repo:

```text
declared build requirement
```

Credit:

```text
Python Packaging Authority / wheel maintainers
https://github.com/pypa/wheel
```

Observed local note:

```text
wheel was declared in pyproject.toml but was not found by pip show in the active environment.
```

## External Command-Line Tools

These are called by the repo but are not vendored in this repository.

### FFmpeg

Use in this repo:

```text
audio decoding
audio duration probing
raw frame encoding to MP4
audio/video muxing
hardware encoder access
image probing/decoding for some render lanes
```

Credit:

```text
FFmpeg project and contributors
https://ffmpeg.org/
```

Observed local binary:

```text
ffmpeg 8.1.1 full build from gyan.dev
configuration includes --enable-gpl
```

License note:

FFmpeg license obligations depend on the exact build configuration and linked
libraries. The observed local build enables GPL components, so treat the local
binary as GPL-affected for redistribution purposes. This repository calls the
installed executable and does not redistribute FFmpeg binaries.

### FFprobe

Use in this repo:

```text
audio/video/image metadata probing
duration checks
render verification
```

Credit:

```text
FFmpeg project and contributors
https://ffmpeg.org/ffprobe.html
```

License note:

Same family as FFmpeg. Do not redistribute without checking the exact binary
license obligations.

## Rust Toolchain

Use in this repo:

```text
native capture
native TrueFrameGen
full-song render hot paths
parallel CPU render loops
Windows API calls
FFmpeg process piping
```

Credit:

```text
Rust project
https://www.rust-lang.org/
```

Observed local versions:

```text
rustc 1.95.0
cargo 1.95.0
```

License note:

The Rust toolchain is commonly distributed under MIT/Apache-2.0 terms. This
repo currently uses the Rust standard library and no third-party Rust crates in
`native/truevision_capture_rs/Cargo.toml`.

## Python Runtime

Use in this repo:

```text
research scripts
tests
tool bus
local studio server
template rendering
document-state reader
manifest/report generation
```

Credit:

```text
Python Software Foundation
https://www.python.org/
```

Observed local version:

```text
Python 3.11.0
```

License note:

Python is distributed under the Python Software Foundation License. Check the
runtime distribution terms before bundling Python itself.

## Windows APIs

Use in this repo:

```text
native process/memory statistics
native capture lanes
screen/system interactions needed for local capture tests
```

Examples:

```text
kernel32
psapi
Windows GDI/user32 style capture APIs in native lanes
```

Credit:

```text
Microsoft Windows platform APIs
https://learn.microsoft.com/windows/win32/
```

License note:

The repo calls platform APIs available on the user's Windows system. It does
not redistribute Windows SDK/runtime components.

## Optional Local Model Stack

The repo can talk to a local LLM through loopback endpoints. The model is a
planner/controller only; it does not execute directly.

### Ollama / OpenAI-Compatible Local Endpoint

Use in this repo:

```text
local Qwen chat/proxy route
prompt-to-state draft generation
operator planning
```

Credit:

```text
Ollama project if used locally
https://ollama.com/
```

License note:

The repo does not vendor Ollama. If Ollama is used, follow Ollama's license and
the license for each model pulled into the local environment.

### Qwen Models

Use in this repo:

```text
local model planning/controller role
draft AV state requests
chat about audio/video projects
```

Credit:

```text
Qwen model family by Alibaba Cloud / Qwen team
https://qwenlm.github.io/
```

License note:

Model licenses vary by model and release. The repository does not redistribute
Qwen model weights. Check the exact local model license before any release or
commercial use.

## Optional / Environment-Only Packages

The current Python environment contains additional packages that are not direct
project dependencies in `pyproject.toml`, including packages such as OpenVINO,
Torch, Ultralytics, SciPy, pandas, matplotlib, and torchvision.

They may have been used in adjacent experiments or available on the machine, but
they are not declared as required dependencies for this repository at this time.
Do not credit them as project requirements unless they become direct imports or
declared dependencies.

## Generated Inputs And User Media

User-provided audio files, images, screenshots, gameplay recordings, generated
videos, manifests, and frame-state logs are runtime artifacts. They are not
third-party code dependencies and are intentionally ignored by git.

Examples of ignored lanes:

```text
outputs/
storage/
E:\TruEVision Generation
```

Do not commit personal media, downloaded songs, generated MP4s, or capture
chunks unless a small fixture is deliberately created and cleared for inclusion.

## Redistribution Checklist

Before publishing a packaged release:

```text
1. Pick and add a license for in-house TrueVision code.
2. Freeze dependency versions.
3. Generate exact dependency license inventory from the release environment.
4. Include required license texts for all redistributed packages/binaries.
5. Confirm whether FFmpeg binaries are redistributed or only invoked externally.
6. Confirm whether model weights are redistributed or only used locally.
7. Keep generated media and personal source artifacts out of the source release.
```

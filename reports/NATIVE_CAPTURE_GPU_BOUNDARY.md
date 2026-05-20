# Native Capture GPU Boundary

## Verified Capture Loop

The Rust lane is capture-only.

```text
screen blit
-> BGRA buffer
-> RGB/luma/saturation/delta cell features
-> compact frame record
-> native cell chunk
```

It does not run:

```text
6-1-6 temporal mapping
fingerprint building
signature extraction
TrueFrameGen gap fill
replay generation
render scoring
AI/template work
```

Those belong to post-capture workers.

## What This Captures

Current native capture uses Windows GDI desktop capture. It sees the final composited screen image after the desktop has been presented.

That is useful for:

```text
visible pixels
visible motion
visible color/luma change
visible fog/smoke appearance
visible camera motion
visible UI/video playback as displayed
```

## What GPU Output We Are Not Capturing

We are not capturing GPU internals:

```text
raw swapchain/backbuffer before desktop composition
depth buffers
normal buffers
motion vectors
object IDs
material IDs
lighting buffers
shadow maps
shader parameters
particle simulation buffers
geometry/mesh data
HDR values before SDR tonemapping
offscreen render targets
occluded window content
protected hardware overlays
audio/video decode internals
GPU timing counters
```

Plain truth:

```text
GDI capture sees what the desktop presents, not what the GPU knows.
```

## Next Better Capture Paths

```text
DirectX Desktop Duplication
  better desktop-frame capture path

Graphics API hook / capture layer
  can see swapchain frames and possibly timing, but is more invasive

Game/render-engine integration
  can export depth, motion vectors, object IDs, and materials

GPU compute feature extraction
  keeps the TrueVision state path but moves resize/color/cell math off CPU
```

## Playback Clarity Test

Prepared harness:

```powershell
python scripts\truevision_native_clarity_test.py
```

It prints a capture/replay plan without recording.

Execute only when ready:

```powershell
python scripts\truevision_native_clarity_test.py --execute
```

Default test:

```text
2560x1440 capture
640x360 cells
4x4 pixels per cell
9 fps target
3 seconds
```

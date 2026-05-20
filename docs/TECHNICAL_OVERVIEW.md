# Technical Overview

TrueVision Video Rendering is a local state-media research project.

## Representation

Captured video is converted into addressed visual state:

```text
state[t][y][x][channel]
```

Current channels include RGB, luminance, saturation, edge density, motion energy, delta luminance, and texture energy. This representation is smaller than raw video and easier to analyze causally, but it does not preserve detail that was never captured.

## Capture

The native recorder captures the display, extracts cell-level features, writes compact state chunks, and returns to the capture loop. It does not run temporal mapping, signature analysis, or frame reconstruction inside the hot path.

## Frame Reconstruction

TrueFrameGen reconstructs higher-rate timelines from known captured states. The preferred current method is SegmentField:

```text
A, B = adjacent observed states
field = estimate_transition(A, B)
for each target time between A and B:
    render step along field
```

This prevents each generated frame from solving independently. It also keeps the grid as an internal lattice rather than a visible output structure.

## Trust Boundary

Generated output is synthetic. Manifests describe source state, timing, algorithm, output path, and known limits. A render may demonstrate continuity, but it is not observational evidence.

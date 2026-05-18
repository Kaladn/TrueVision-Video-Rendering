# TrueVision FPS Optimization Smoke Report

## Purpose

Quickly test whether the 9 fps ceiling was a concept limit, GPU limit, or ordinary Python/capture pipeline drag.

## Result

The 9 fps ceiling is not conceptual.

It was mostly:

```text
Python grid loop
full-screen 2560x1440 capture before resize
cell-state extraction cost
foreground NPZ chunk writes
```

## Test Setup

Recorder command shape:

```powershell
cd D:\arc_solver_clean\cod_616

python truevision_resonance_recorder.py `
  --duration 10 `
  --fps 15 `
  --resolution 960x540 `
  --grid 160x90 `
  --blocks 16x9 `
  --start-delay 0
```

Baseline run:

```text
run_id: fps_baseline_10s_pre_vector
capture source: full 2560x1440 screen
```

Vectorized grid run:

```text
run_id: fps_vector_grid_10s
capture source: full 2560x1440 screen
change: frame_to_grid uses cv2.INTER_AREA resize instead of nested Python loops
```

Vectorized grid + crop run:

```text
run_id: fps_vector_grid_center_crop_10s
capture source: center crop 640,360,1280,720
analyzed frame: 960x540
```

## Results

```text
baseline:
  frames: 85
  duration: 9.896942s
  effective fps: 8.59
  summary fps mean: 9.01
  output size: 17.68 MiB

vectorized grid:
  frames: 110
  duration: 9.975439s
  effective fps: 11.03
  summary fps mean: 12.47
  output size: 24.02 MiB

vectorized grid + center crop:
  frames: 131
  duration: 9.999721s
  effective fps: 13.10
  summary fps mean: 14.44
  output size: 21.05 MiB
```

## Speedup

Effective FPS:

```text
baseline -> vectorized grid:
  8.59 fps -> 11.03 fps
  about 28.4 percent improvement

baseline -> vectorized grid + crop:
  8.59 fps -> 13.10 fps
  about 52.5 percent improvement
```

## Stage Timing After Vectorization

No-write timing probe after vectorized grid:

```text
capture + resize:   mean 41.69 ms
frame_to_grid:      mean 0.24 ms
delta:              mean 0.018 ms
blocks:             mean 0.99 ms
resonance:          mean 0.23 ms
cell_state:         mean 32.93 ms
total no-write:     mean 76.10 ms
derived fps:        13.14
```

Before vectorization, `frame_to_grid` was roughly:

```text
about 48 ms
```

After vectorization:

```text
about 0.24 ms
```

That was the clean first win.

## Code Change

`modules/screen_grid_mapper.py` changed `frame_to_grid` from a nested Python loop over every cell to:

```python
return cv2.resize(
    gray,
    (self.grid_cols, self.grid_rows),
    interpolation=cv2.INTER_AREA,
).astype(np.float32)
```

This is both faster and more correct for fractional cell boundaries.

## New Test

`test_screen_grid_mapper_dimensions.py` now includes:

```text
test_frame_to_grid_uses_area_weighting_for_fractional_cells
```

That test failed before the patch and passed after the patch.

## Remaining Bottlenecks

After the first fix, the main costs are:

```text
screen capture + resize: about 42 ms
cell-state extraction: about 33 ms
foreground compressed writes: occasional run-level drag
```

Next optimization order:

```text
1. Capture only the target video/window region.
2. Avoid duplicated gray/luma work between grid and cell-state extraction.
3. Background-write NPZ chunks.
4. Consider DXGI / Windows Graphics Capture instead of MSS.
5. Consider Rust/C++/OpenCL/OpenVINO only after algorithmic cleanup.
```

## Bottom Line

```text
9 fps was pipeline drag, not a hard limit.
CPU-only 15 fps is within reach.
30 fps likely needs better capture backend and background writing.
```


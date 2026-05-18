# TrueVision Data Usage And Recording Math

## Baseline Capture

Measured run:

```text
D:\arc_solver_clean\cod_616\data\truevision_full\music_video_30s_cell_vectors_20260518
```

Observed capture:

```text
duration: 29.952428 seconds
frames: 243
frame shape: 960x540
grid shape: 90x160
features per cell: 16
cell dtype: float32
```

Per-frame raw cell tensor:

```text
90 * 160 * 16 * 4 bytes = 921,600 bytes
921,600 bytes = 0.87890625 MiB per frame
```

## Actual Disk Usage From The 30s Run

```text
.npz cell chunks: 48,645,687 bytes = 46.392 MiB
records JSONL: 2,227,637 bytes = 2.124 MiB
JSON manifests/reports: 13,937 bytes = 0.013 MiB
lossless FFV1 replay MKV: 4,002,493 bytes = 3.817 MiB
MP4 preview: 1,612,232 bytes = 1.538 MiB
total run folder: 56,501,986 bytes = 53.884 MiB
```

The observed compressed cell-state chunks used:

```text
200,188 bytes per frame
1,624,098 bytes per second
about 5.445 GiB per hour for compressed NPZ cell chunks only
```

Capture-only estimate with JSONL:

```text
cell chunks: about 5.445 GiB/hour
JSONL records: about 0.249 GiB/hour
capture-only total: about 5.7 GiB/hour
```

Capture plus replay files, using this run as the ratio:

```text
total folder rate: about 6.32 GiB/hour
```

This includes compressed cells, records, summary/manifest/report JSON, FFV1 replay, and MP4 preview.

## Raw Equivalent

Uncompressed cell tensor at 90x160x16 float32:

```text
9 fps: 27.81 GiB/hour
15 fps: 46.35 GiB/hour
30 fps: 92.70 GiB/hour
```

Uncompressed RGB frames at 960x540:

```text
one RGB frame: 960 * 540 * 3 = 1,555,200 bytes
9 fps: 46.93 GiB/hour
15 fps: 78.21 GiB/hour
30 fps: 156.43 GiB/hour
```

The current compressed cell-state lane is therefore much smaller than raw RGB and much smaller than uncompressed float32 cells.

## If We Go Full-Fledged

Full-fledged can mean different recording tiers.

### Tier 1: State Audit

```text
960x540 input
90x160 grid
16 float32 features
9 fps
compressed NPZ chunks
JSONL records
no raw frames
```

Expected disk:

```text
about 5.7 GiB/hour capture-only
about 6.3 GiB/hour with replay artifacts retained
```

Best for:

```text
security telemetry
temporal visual state
motion/flash/activity review
replayable but not raw-pixel evidence
```

### Tier 2: Rich State

If grid density doubles each axis:

```text
320x180 grid = 4x more cells than 160x90
```

Expected compressed storage if compression behaves similarly:

```text
about 21.8 GiB/hour for compressed cell chunks
about 23 GiB/hour including JSONL and small reports
```

If features double from 16 to 32 at 320x180:

```text
about 43.6 GiB/hour for compressed chunks, rough estimate
```

Raw uncompressed math for 320x180x32 float32 at 9 fps:

```text
320 * 180 * 32 * 4 * 9 * 3600 = about 222.47 GiB/hour
```

Best for:

```text
better reverse rendering
object/material/depth sidecars
research-grade visual state
```

### Tier 3: Evidence Plus State

If raw video is also preserved:

```text
raw RGB 960x540 at 30 fps alone is about 156.43 GiB/hour uncompressed
```

Real codec size depends heavily on content and codec:

```text
lossless video can vary wildly
compressed H.264/H.265 is smaller but not a pure forensic raw store
state capture still adds its own storage
```

Best for:

```text
evidence preservation
forensic review
training state/replay models against raw source
```

## Write Bandwidth

Measured compressed cell chunk write rate:

```text
about 1.62 MB/s for cell chunks
about 1.89 MB/s including all run files at observed settings
```

Raw cell float32 bandwidth:

```text
90x160x16 at 9 fps: about 8.29 MB/s
90x160x16 at 15 fps: about 13.82 MB/s
90x160x16 at 30 fps: about 27.65 MB/s
```

This is not extreme for an SSD, but retention matters. A system running continuously at the current compressed tier would consume about:

```text
6.3 GiB/hour
151 GiB/day
about 1 TiB/week
```

That is why retention policy has to be designed early.

## Recommended Retention Shape

```text
normal windows:
  keep summaries and event receipts
  compact or delete cell chunks after short TTL

interesting windows:
  keep compressed cell chunks
  keep replay reports
  keep hash manifests

critical windows:
  preserve state chunks
  preserve raw source if available
  preserve lossless replay
  preserve event correlations
```

Suggested starting retention:

```text
normal state chunks: 6 to 24 hours
interesting state chunks: 7 to 30 days
critical state chunks: case-controlled retention
raw evidence: explicit policy only
```

## Bottom Line

The current state lane is cheap enough for short security windows and focused sessions. It is not cheap enough to run forever without retention.

Good working number:

```text
about 6 GiB/hour for the current 960x540, 90x160, 16-feature, 9 fps compressed state lane with replay artifacts.
```


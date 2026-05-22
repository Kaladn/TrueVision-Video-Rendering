# TrueVision Video Rendering TODO

## 1. Repository Discipline

- [x] Keep generated media and run artifacts out of git.
- [x] Keep runtime storage directories as placeholders only.
- [x] Keep the project scoped to audio/video state media.
- [ ] Add a `CONTRIBUTING.md`.
- [ ] Add a license after ownership and release intent are decided.
- [ ] Add CI for Python tests and Rust build.

## 2. Native Capture

- [x] Rust native capture emits `.tvcells` state chunks.
- [x] Capture loop keeps mapping/fingerprinting out of the hot path.
- [x] Capture manifests state that raw frames are not saved by default.
- [ ] Add a stop/abort control for long captures.
- [ ] Add selected-window capture.
- [ ] Add GPU capture research branch only after CPU/native capture is stable.

## 3. TrueFrameGen

- [x] Python 6-1-6 temporal map exists.
- [x] Rust streaming renderer exists.
- [x] SegmentField mode exists: one A-to-B transition field, many generated frames.
- [ ] Promote SegmentField as the default proof path after clean source validation.
- [ ] Add stronger cell-boundary deblocking for final presentation.
- [ ] Add transition confidence maps for review.
- [ ] Add long-run chunked rendering.
- [ ] Validate `libx264`, `h264_qsv`, `hevc_qsv`, and `h264_amf` output quality.

## 4. Rendering Language

- [x] Template renderer exists.
- [x] Audio feature extraction tools exist.
- [x] Initial state-pattern library exists.
- [x] Document-state reader exists for page-frame and glyph-cell recall.
- [x] Render Law is documented: state/grid/primitive first, pixels last.
- [ ] Formalize the AV state template schema.
- [ ] Add material channels for fog, smoke, water, glass, glow, and lighting pressure.
- [ ] Add camera-motion primitives.
- [ ] Add time-marker recalibration patches.
- [ ] Connect document-state packets to shape/language rules for tutorials, charts, and graph generation.

## 5. Local Studio

- [x] Local HTML studio exists.
- [x] Local server exists.
- [x] AV-only tool registry and policy layer exist.
- [x] Local model adapter shape exists.
- [x] Reusable Studio tool contracts exist for source snap, existing-state animation, glow intensity, spectrum city, frame diff, manifests, presets, and Qwen control.
- [x] Proven render lanes are represented as reusable presets, including the House Remix visual preset.
- [ ] Remove any remaining placeholder UI language.
- [ ] Add native capture controls to the studio server.
- [ ] Add template comparison view.
- [ ] Add render status polling.
- [ ] Add true FFT/Goertzel frequency-bin analyzer bars to replace the current low/mid/high facsimile.
- [ ] Wire Studio preset launch into the Rust renderer with preview/full render job execution receipts.

## 6. Documentation

- [x] Public README explains the project boundary.
- [x] Tool inventory exists.
- [x] State generation primitive notes exist.
- [ ] Add architecture diagram.
- [ ] Add capture format specification for `.tvcells`.
- [ ] Add TrueFrameGen algorithm notes with pseudocode.
- [ ] Add plain-language walkthrough.

## 7. Validation

- [x] Unit tests cover core Python modules.
- [x] Rust build has been proven locally.
- [x] Full-song QSV/32-thread render produced per-frame deterministic state records.
- [ ] Add small fixture capture for reproducible tests.
- [ ] Add benchmark command for capture FPS, frame time, CPU, and RAM.
- [ ] Add generated-video quality metrics that do not require external services.

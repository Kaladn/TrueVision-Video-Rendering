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

### TrueAudio Pre-Output Lane

- [x] Define TrueAudio as a sibling audio-state system used alongside TrueVision, not inside it.
- [x] Add `trueaudio_log_pre_sound` for decoded PCM state logging before playback/output.
- [x] Add `trueaudio_log_machine_pre_sound` for local machine output-mix logging before speakers.
- [x] Add Windows WASAPI loopback backend for machine pre-sound logging.
- [x] Add `trueaudio_replay_state` for deterministic log sonification.
- [x] Add replayable TrueAudio spectral-state capture for close audio replay without raw PCM storage.
- [x] Add replayable TrueAudio state replay to WAV.
- [x] Write derived audio state JSONL without saving raw audio or PCM.
- [x] Write TrueAudio manifest and receipt with source hash, decoder path, frame count, duration, and boundary claims.
- [x] Expose TrueAudio logging through the AV-only tool bus.
- [x] Prove a 10-second live machine-output log can be sonified from state.
- [x] Prove a 10-second live machine-output replayable state can reconstruct close audio.
- [ ] Run a live machine-output capture with user-started playback and review the first receipt.
- [ ] Promote the TrueAudio FFmpeg discovery helper into the canonical repo-wide media executable resolver.
- [ ] Add a TrueAudio-to-TrueVision sync contract for audio-driven video renders.

### Voice State Lane

- [x] Add `truespeech_detect_segments` for speech/background timing from replayable TrueAudio state.
- [x] Write TrueSpeech frame/segment outputs with no transcript or ASR claim.
- [x] Prove replayable-state speech detection can run faster than realtime on a 10-second capture.
- [x] Add `trueaudio_log_file_replayable` for fast file ingestion into replayable TrueAudio state.
- [x] Add `truespeech_align_lyrics_candidate` for provided-lyrics timing candidates.
- [x] Prove a 60-second Rescue Me source-file joint test can produce replayable state, speech segments, and candidate lyric timing.
- [ ] Define `voice_state_v1` as a deterministic audio-state format.
- [ ] Add `voice_extract_timeline` tool using FFmpeg PCM decode.
- [ ] Add `voice_align_script` for script/lyrics/narration line timing.
- [ ] Add vocal-isolation or voice-vs-music calibration before treating music vocals as clean speech.
- [ ] Add phoneme/word candidate lane after speech/background segmentation is stable.
- [ ] Add editable voice timing JSON for manual correction.
- [ ] Add voice channels: `voice_rms`, `voice_presence`, `voice_attack`, `voice_decay`, `breath_noise`, `sibilance`, `low_voice_weight`, `spoken_phrase_pressure`, `line_start`, `line_end`, `silence_gap`.
- [ ] Add voice manifest fields showing source audio, timing source, line count, duration, hash, and whether timing was estimated or manually corrected.
- [ ] Add `voice_mix_bed` plan for voice/music balancing and muxing.

### Cleanup Before State Vocal Services

- [ ] Pick one canonical audio feature contract; audio extraction is currently duplicated across Python tools and Rust render lanes.
- [ ] Mark current Rust `vocal_presence()` as a heuristic, not a real voice detector.
- [ ] Normalize FFmpeg discovery so every tool can find it without shell-specific PATH hacks.
- [ ] Formalize voice artifact storage: `storage/artifacts/voice`, `storage/manifests`, and `storage/receipts`.
- [ ] Keep voice lane audio/video only: no ASR claims, no cloud TTS requirement, no assistant-service sprawl.
- [ ] Use `voice_state` naming unless a long-running service is intentionally introduced.
- [ ] Preserve the default design: FFmpeg decodes and muxes, TrueAudio derives audio/voice state, TrueVision consumes validated sync state when rendering, and manifests prove what audio drove what.

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
- [ ] Add voice/narration timing view.
- [ ] Add line timing editor for voice-state JSON.
- [ ] Add voiceover preview lane for presentation videos.

## 6. Documentation

- [x] Public README explains the project boundary.
- [x] Tool inventory exists.
- [x] State generation primitive notes exist.
- [x] Repo system guide documents what the project is, why it exists, and who talks to who.
- [x] Third-party notices document direct dependencies, external tools, and credit guidance.
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
- [x] Test TrueAudio pre-output logging is deterministic.
- [x] Test TrueAudio manifest records source hash, timing, schema, and no-raw-audio boundary.
- [x] Test AV tool bus can execute `trueaudio_log_pre_sound` and route `trueaudio_log_machine_pre_sound`.
- [x] Test TrueAudio state replay is bounded sonification and does not claim source-audio recovery.
- [x] Test TrueSpeech detection reads replayable TrueAudio state without transcript claims.
- [x] Test AV tool bus routes `truespeech_detect_segments`.
- [x] Test source-file replayable state logging uses FFmpeg decode without raw PCM storage.
- [x] Test provided-lyrics alignment is candidate-only and survives manifest/receipt write.
- [x] Test AV tool bus routes source-file replayable logging and candidate lyric alignment.
- [ ] Add tiny voice WAV fixture.
- [ ] Test voice extraction is deterministic.
- [ ] Test script-line timing survives save/load.
- [ ] Test voice manifest records source hash and timing method.

## 8. Cross-System Harness Pickup

- [ ] Open the other workspace in SecureCore first.
- [ ] Inventory what is actually on the SecureCore / AnchorWorks / TrueVision workbench.
- [ ] Lock the cross-system harness shape before adding new features.
- [ ] Prove AnchorWorks, SecureCore, and TrueVision health checks separately.
- [ ] Bring TrueVision Generation in only as an optional state-media lane.

Current handoff law:

```text
AnchorWorks = face and language/count brain
SecureCore = safety, agents, logging, policy, system substrate
TrueVision Generation = state-media render lab
Harness = proof that they can cooperate without becoming one tangled thing
Connector point = validated state packets
```

# TrueVision Video Rendering TODO

## 1. Repository Discipline

- [x] Keep generated media and run artifacts out of git.
- [x] Keep runtime storage directories as placeholders only.
- [x] Keep the project scoped to audio/video state media.
- [x] Park platform-style backend plans; TrueVision is a local usable stack first.
- [x] Treat new ideas as bounded workers until contracts, tests, manifests, and receipts prove them.
- [x] Lock generation POC as an output lane, not the product center.
- [x] Add local product map, active tool surface, parked experiment map, and AW/SC boundary contracts.
- [x] Add `scripts/truevision_preflight.py` for non-mutating local prerequisite checks.
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
- [x] Add first material channels for fog, mist, clouds, and rain-on-glass state tools.
- [ ] Add remaining material channels for smoke, water, glow, and lighting pressure.
- [ ] Add camera-motion primitives.
- [ ] Add time-marker recalibration patches.
- [ ] Connect document-state packets to shape/language rules for tutorials, charts, and graph generation.

### Atmosphere / Weather State Lane

- [x] Add reusable atmosphere element contracts for `fog_density_field`, `mist_veil_field`, `cloud_volume_field`, and `rain_glass_field`.
- [x] Add state channels: `density`, `veil_opacity`, `scatter_bloom`, `edge_softness`, `motion_pressure`, `curl_pressure`, `occlusion_pressure`, `droplet_density`, `droplet_streak`, `refraction`, `surface_wetness`.
- [x] Add native `.tvcells` capture profiler that samples state and builds 6-1-6 windows.
- [x] Add `atmosphere_toolset_create` to write reusable templates and manifests.
- [x] Add `atmosphere_profile_from_capture` to the AV-only tool bus.
- [ ] Run the new profiler on the full fog/mist teacher capture and review density windows.
- [ ] Add renderer hooks that consume atmosphere profiles without hardcoding one script.
- [ ] Add rain-on-glass reference capture and compare droplet/streak/refraction channels.
- [ ] Add cloud-volume reference capture and compare curl/occlusion channels.

### Elemental Learning Intake

- [x] Add `docs/ELEMENTAL_LEARNING_INTAKE_PLAN.md`.
- [x] Add YouTube source-surface safe-ops contract with declared display IDs, button IDs, and forbidden controls.
- [x] Add `source_surface_capture_plan` for deterministic non-fullscreen source trials: capture starts before play and stops from source video time.
- [x] Add `source_surface_multi_sample_plan` for large videos: four section samples instead of one long capture.
- [x] Require browser address-bar navigation for approved YouTube URLs; strip playlist/search noise before capture.
- [x] Add `source_surface_video_state_receipt` so a completed macro cannot count unless URL/title/duration/state/profile/purge checks pass.
- [x] Require a saved coordinate map before every coordinate intake run; no map, no run, and map hash must appear in queue, summary, and receipt.
- [ ] Add `smoke_curl_field` and `smoke_dissipation_field` element contracts.
- [ ] Add an `element_intake_queue` JSONL format for approved visual-only teacher tasks.
- [ ] Add source candidate records with `element_id`, search terms, source URL/path, approval state, capture settings, and retention intent.
- [ ] Add capture-plan builder for intake tasks without free desktop/browser control.
- [x] Add `element_creation_profile_from_capture` to convert teacher state into compact creation signatures.
- [x] Add purge-after-profile receipt path so bulky observed teacher chunks expire after profile verification.
- [ ] Add 42s smoke source plan: `0.25x` playback, about `180s` capture, visual-only, 15 FPS, `640x360` grid.
- [x] Add 3-source process regression test proving profile/receipt/report survive while teacher chunks expire.
- [ ] Run the 3-video process test: capture, profile, verify, purge teacher state, then move to next source.
- [ ] Add profile quality scoring for element captures.
- [ ] Add profile comparison for fog vs smoke vs mist vs clouds.
- [ ] Add renderer-profile binding so renderers consume learned element profiles instead of hardcoded script parameters.
- [ ] Add retention closeout: keep profile/manifest/receipt/proof preview, then expire heavy teacher chunks unless marked gold.
- [x] Add open-license dataset intake policy for prototype learning sources.
- [x] Attribute `faridlab/deepaction_v1` in third-party notices as a human-action motion-profile candidate.
- [ ] Add Hugging Face DeepAction intake queue that selects only CC BY 4.0 generated folders by default.
- [ ] Add DeepAction cache guard so downloaded clips live outside the repo and are purged after profile verification.
- [ ] Add DeepAction motion-profile receipt fields: source folder, action class, clip count, license family, profile hash, purge proof.
- [ ] Run a tiny DeepAction proof: 5 actions, 2 generated source families, 3 clips each, profile then purge raw clips.

### Terrain Realism Teacher

- [x] Add `docs/TERRAIN_REALISM_TEACHER_PLAN.md`.
- [x] Add bounded `terrain_teacher` workspace contract for real-world structure before cinematography.
- [x] Seed source classes for oceans/cliffs, canyons, volcanoes, storm oceans, and mountain fog layers.
- [x] Add terrain candidate ranking that prefers 30-90 minute real geography sources with transcripts.
- [x] Add disk guard and cleanup receipt shape so raw teacher media/cache flushes after each job.
- [x] Add terrain human-review packets and block auto-promotion of learned rules.
- [ ] Initialize the morning terrain workspace outside git before the first source run.
- [ ] Process one ocean-cliff source first and extract horizon, edge, scale, texture, depth, and occlusion rules.
- [ ] Render only a 12-second `edge_nightmare_world --shot-type wide_edge_intro` proof after the first promoted terrain rule.
- [ ] Add terrain realism QA comparison against the current wide-edge proof before any full-song render.
- [ ] Add future raytracing/pathtracing alternative capture/learn/transform logger contracts after terrain rules prove useful.

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
- [x] External API adapter shape exists.
- [x] Reusable Studio tool contracts exist for source snap, existing-state animation, glow intensity, spectrum city, frame diff, manifests, presets, and external API draft assistance.
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
- [x] Add saved timeline audit for frame/FPS timing integrity across state logs.
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

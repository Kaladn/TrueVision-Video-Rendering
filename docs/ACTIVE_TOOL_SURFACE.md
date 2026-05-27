# Active Tool Surface

This file lists the currently active local TrueVision tool surface.

Active means:

```text
local-first
repeatable
covered by a contract or test
no surprise install
no browser control unless explicitly operator-approved
writes manifests/receipts when it changes runtime state
```

## Preflight

```powershell
python scripts\truevision_preflight.py
```

Checks local prerequisites and repo health.

## Timing Audit

```powershell
python scripts\truevision_timing_audit.py <manifest.json>
```

Verifies saved timeline logs against frame index, FPS, manifest duration, and state-log cadence.

Law:

```text
Frame index and FPS are the clock.
Wall time is performance, not timeline truth.
```

## TrueAudio / TrueSpeech

```powershell
python scripts\trueaudio_log_file_replayable.py ...
python scripts\trueaudio_log_machine_replayable.py ...
python scripts\truespeech_detect_segments.py ...
python scripts\truespeech_align_lyrics_candidate.py ...
```

These lanes derive audio state from decoded PCM and produce replay/timing artifacts.

## TrueFrameGen

```powershell
python scripts\trueframegen_upsample.py ...
python scripts\trueframegen_live_upsample.py ...
python scripts\trueframegen_fill.py ...
```

These lanes derive intermediate frame/state transitions. They do not replace source truth frames.

## Learning / Measurement Workers

```powershell
python scripts\truevision_worker_forge.py inventory ...
python scripts\truevision_worker_forge.py forge ...
python scripts\truevision_worker_forge.py choose ...
python scripts\truevision_meter_grid.py ...
python scripts\truevision_angular_seismic_video.py ...
python scripts\truevision_driving_school.py ...
python scripts\truevision_state_focus_lens.py ...
```

The worker forge is the local mini-SecureCore-style selector for TrueVision tools
and workers. It writes manifests, append-only logs, and receipts; it does not
execute selected workers.

The measurement workers produce compact profiles, candidates, and receipts.
Candidate-first output is intentional.

Worker law:

```text
One worker.
One job.
One artifact.
One receipt.
No hidden work.
```

Full worker rack contract:

```text
docs/TRUEVISION_WORKER_RACK_CONTRACT.md
```

## Render / Proof Lanes

```powershell
python scripts\truevision_render_template.py ...
python scripts\render_truedepth_fog_reveal_samples.py ...
python scripts\render_trudepth_rave_laser_sample.py ...
```

Render lanes are proof and presentation surfaces. They are not the center of the product.

## Studio / UI

```powershell
python scripts\truevision_studio_server.py
```

Local studio tooling is a development surface. It is not a remote backend.

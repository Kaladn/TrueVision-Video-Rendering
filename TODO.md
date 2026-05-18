# TrueVision Generation Lab TODO

Hard boundary:

```text
Forward TrueVision witnesses.
Reverse TrueVision replays or demonstrates state.
Generated state media is synthetic, not evidence.
SecureCore guards later.
AnchorWorks routes later.
The lab proves the visual-state language first.
```

## Layer 0: In-Repo Storage

- [ ] Use `storage/` as the local runtime root.
- [ ] Keep generated runtime data ignored by default.
- [ ] Keep only directory structure in git.
- [ ] Store local runtime data by lane:

```text
storage/
  inbox/
  outbox/
  events/
  state_chunks/
  artifacts/
  manifests/
  reports/
  receipts/
  presets/
  tmp/
```

## Layer 1: Storage Contracts

- [ ] Define `storage_manifest_v1`.
- [ ] Define `state_chunk_manifest_v1`.
- [ ] Define `artifact_manifest_v1`.
- [ ] Define `writer_receipt_v1`.
- [ ] Define `tip_event_v1`.
- [ ] Hash every stored payload.
- [ ] Keep raw frames disabled unless explicitly requested.

## Layer 2: SecureCore-Shaped Backend Ports

- [ ] Add writer port.
- [ ] Add artifact port.
- [ ] Add policy port.
- [ ] Add notification port.
- [ ] Require all writes to pass through ports.
- [ ] Forbid services from writing directly to final storage.

Target shape:

```text
truevision_runtime/
  ports/
    writer_port.py
    artifact_port.py
    policy_port.py
    notification_port.py
```

## Layer 3: Local Adapters

- [ ] Add local writer adapter.
- [ ] Add local artifact adapter.
- [ ] Add local policy adapter.
- [ ] Add local notification adapter.
- [ ] Write local events to `storage/outbox/`.
- [ ] Write receipts to `storage/receipts/`.
- [ ] Keep this mode independent of SecureCore.

Target shape:

```text
truevision_runtime/
  adapters/
    local_writer_adapter.py
    local_artifact_adapter.py
    local_policy_adapter.py
    local_notification_adapter.py
```

## Layer 4: Capture And Snip

- [ ] Finish region snip/watch workflow.
- [ ] Add interactive selector verification.
- [ ] Add multi-monitor metadata.
- [ ] Add overlay preview before capture.
- [ ] Add preset list/delete/rename.
- [ ] Add window-bound watch mode later.
- [ ] Store presets under `storage/presets/`.
- [ ] Store state chunks under `storage/state_chunks/`.

## Layer 5: State Interpreter

- [ ] Add state interpreter service.
- [ ] Detect idle loops.
- [ ] Detect repeated failures.
- [ ] Detect covered/active window mismatch.
- [ ] Detect motion spikes.
- [ ] Detect visual flash spikes.
- [ ] Detect stable stuck-screen state.

Target shape:

```text
truevision_runtime/
  services/
    capture_service.py
    state_interpreter.py
```

## Layer 6: Tip Engine

- [ ] Add rule-first tip engine.
- [ ] Add local LLM wording adapter only after schemas validate.
- [ ] Add ClearSpeak-style wording layer.
- [ ] Add confidence thresholds.
- [ ] Add user feedback: useful, wrong, silence.
- [ ] Store tips as events, not direct UI commands.

Target shape:

```text
truevision_runtime/
  services/
    tip_engine.py
    avatar_event_router.py
```

## Layer 7: Avatar Overlay

- [ ] Build overlay shell only after tip events exist.
- [ ] Consume approved tip events only.
- [ ] Move away from cursor/text-entry areas.
- [ ] Add visible pause/on/off state.
- [ ] Add local-only mode indicator.
- [ ] No autonomous clicking.
- [ ] No autonomous typing.

## Layer 8: Prompt-To-State Compiler

- [ ] Define strict prompt-to-state JSON schema.
- [ ] Add compiler service.
- [ ] Validate generated state before rendering.
- [ ] Reject prompt output that bypasses state contracts.
- [ ] Route valid state into existing generators.

Target shape:

```text
truevision_runtime/
  services/
    prompt_to_state_compiler.py
```

## Layer 9: Focus Reconstruction

- [ ] Add `scripts/truevision_focus_reconstruct.py`.
- [ ] Use stored state only.
- [ ] Do not reread original photo/video.
- [ ] Apply deterministic focus math:

```text
Lanczos upsample
CLAHE/local contrast
edge-gated unsharp mask
bilateral cleanup
chroma-safe saturation restore
cell-boundary deblocking
```

## Layer 10: Learning Twin

- [ ] Add observer twin records.
- [ ] Add renderer twin records.
- [ ] Log failures.
- [ ] Log successes.
- [ ] Log which channels improved output.
- [ ] Propose next render rules from evidence.
- [ ] Store learning records under `storage/events/`.

## Layer 11: SecureCore Adapter

- [ ] Add SecureCore writer adapter only after local contracts are stable.
- [ ] Map local event envelopes to SecureCore Central Writer envelopes.
- [ ] Map artifact manifests to SecureCore Artifact Engine manifests.
- [ ] Map local policy decisions to SecureCore Policy Gate later.
- [ ] Keep SecureCore import optional.

Target shape:

```text
truevision_runtime/
  adapters/
    securecore_writer_adapter.py
    securecore_artifact_adapter.py
```

## Layer 12: Rust / Compiled Capture Worker

- [ ] Build Rust only after Python contracts are stable.
- [ ] First target: `truevision-capture-worker.exe`.
- [ ] Rust handles capture speed, timing, ring buffer, chunk writing, hashing.
- [ ] Python keeps state language, reports, learning rules, compiler prototypes.

## Existing Base To Preserve

- [x] Standalone project root at `D:\TrueVision_Generation_Lab`.
- [x] Static studio HTML copied into `ui/`.
- [x] Recorder/replay/generator scripts copied into `scripts/`.
- [x] Path tracing lane copied into `scripts/`.
- [x] Still image state capture copied into `scripts/`.
- [x] Grid mapper and resonance state copied into modules/root.
- [x] Region snip tool added.
- [x] Rust compiled capture lane documented.

Tiny law:

```text
Storage first.
Ports second.
Local adapters third.
SecureCore later.
No rewiring.
```

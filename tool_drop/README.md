# TrueVision Tool Drop

This directory is the categorized TrueVision tool inventory.

It is a catalog surface only. It does not own implementations, move code, rename
scripts, or imply authority.

## Law

```text
Catalog first.
Wrappers second.
Copies third.
Authority never implied.
```

## Boundaries

```text
scripts/ = existing user-facing launchers
truevision_runtime/ = runtime/tool implementations
trueaudio_runtime/ = audio-state implementations
native/ = Rust/native power lanes
storage/ = outputs, manifests, receipts, reports
tool_drop/ = catalog and contracts only
```

Every `.tool.json` manifest points to existing implementation paths and declares
what the tool reads, writes, calls, and must not claim.

## State Direction

```text
Watch direction:
media or playback surface -> state rows/cells -> behavior profile

Generate direction:
behavior profile + desired scene -> state path -> optional media surface
```

The tool drop must declare both directions without confusing either one with
raw media truth. Rendered media is a surface. Learned behavior lives in state.

Canonical TrueVision terms:

```text
Witness -> Profile -> Plan -> Replay -> Surface
```

```text
TrueVision does not record video to make video.
TrueVision witnesses state to learn behavior.
TrueVision does not render effects from copies.
TrueVision surfaces planned state as media.
```

## Copy Policy

```text
Do not move working tools.
Do not break existing script paths.
When a tool needs to enter the drop as code, copy it first.
The copied version must earn promotion before any caller changes.
```

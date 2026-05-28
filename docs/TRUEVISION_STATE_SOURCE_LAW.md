# TrueVision State Source Law

Status: locked contract.

This law exists to stop raw media, generated clips, screenshots, cartoons, or recovered video payloads from being treated as TrueVision source truth.

## Law

```text
If it is raw pixels, it is not the TrueVision source.
If it is state, it can be replayed.
If it is replayed, it is derived.
If it is generated/cartoon, it is visualization.
If it is not state-backed, it does not count.
```

## Source Truth

Allowed as TrueVision source truth:

```text
.tvcells native cell state
records_jsonl / frame-state JSONL
state/profile JSON
cell_state NPZ
manifests
receipts
temporal pulse bridge rows
```

Not allowed as TrueVision source truth:

```text
MP4 / MKV / MOV / WEBM / AVI
H.264 / HEVC payloads
PNG / JPG / screenshots
WAV / MP3 / FLAC / AAC
rendered videos
cartoon overlays
salvaged media
desktop witness captures
generated music videos
OpenAI images
YouTube recordings
```

Those media files may be inputs, temporary bridges, or visualizations only when explicitly marked that way.

## Required Boundary

Every replay/render/cartoon/overlay proof must say:

```text
derived_from_state
visualization_only
source_truth_allowed: false
generated_media_is_evidence: false
state_refs
```

## Runtime Helper

The executable law lives in:

```text
truevision_runtime/state_source_law.py
```

Focused tests:

```powershell
python -m unittest discover -s tests -p test_state_source_law.py -v
```

## Chain

Correct chain:

```text
capture state
-> write manifest/receipt
-> replay from state
-> mark replay as derived
-> audit timing/provenance
```

Wrong chain:

```text
record raw video
-> treat video as source truth
-> salvage broken media
-> call it TrueVision evidence
```

# TrueVision To AnchorWorks Packet Contract

## Purpose

Define how TrueVision state leaves the media/tooling stack and becomes consumable by AnchorWorks.

## Boundary

```text
TrueVision owns observed/renderable state.
AnchorWorks owns language, counts, reasoning, and operator meaning.
The packet owns alignment.
```

TrueVision packets must not claim final semantic truth.

## Packet Shape

Minimum packet:

```json
{
  "schema": "truevision_anchorworks_packet.v1",
  "packet_id": "tv_aw_...",
  "source_manifest": "storage/manifests/...",
  "source_receipt": "storage/receipts/...",
  "state_kind": "meter_grid|trudepth|trueaudio|driving_school|render_proof",
  "timebase": {
    "clock": "frame_index_fps",
    "fps": 30.0,
    "frame_count": 300,
    "duration_seconds": 10.0
  },
  "candidate_records": [],
  "meter_summary": {},
  "known_limits": [],
  "truth_status": "candidate_only",
  "anchorworks_review_required": true
}
```

## Allowed Truth Status

```text
candidate_only
meter_supported
operator_reviewed
rejected
```

Forbidden in TrueVision output:

```text
final_truth
meaning_promoted
operator_intent_claimed
security_decision
```

## Adapter Law

```text
Words name.
State records measure.
AnchorWorks interprets.
```

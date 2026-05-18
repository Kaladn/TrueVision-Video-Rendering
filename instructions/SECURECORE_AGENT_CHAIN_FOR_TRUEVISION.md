# SecureCore Agent Chain For TrueVision State Media

## Purpose

This package must use existing SecureCore agents first.

The user already created agents that can chain. The correct move is to bind the TrueVision state-media POC to those agents before building more runtime machinery.

## Current SecureCore Agents

### temporal_causality_log_checker

Role:

```text
read-only temporal, causal, replay, hash-chain, and lead-up verifier
```

Use for:

```text
TrueVision JSONL or Forge-style event logs
capture/replay sequence validation
success/failure ledger ordering
causal parent checks
breach/error lead-up checks
```

Authority:

```text
read-only
no approval required
no system mutation
```

### reverse_state_motion_demo

Role:

```text
contained reverse-state media demonstration
```

Use for:

```text
checking deterministic media artifact generation shape
checking manifest/artifact output pattern
small contained synthetic sanity demos
```

Authority:

```text
contained media write only
no evidence authority
no recognition authority
no policy authority
```

### recovered_security_snapshot

Role:

```text
report-only security snapshot
```

Use for:

```text
host state snapshot
network/process/service/task/DNS report
security context report before or after a sensitive run
```

Authority:

```text
writes report/evidence files
requires exact human approval phrase
does not enforce firewall blocks unless block flags are used elsewhere
```

### recovered_security_firewall_enforcer

Role:

```text
approval-gated firewall enforcement
```

Use for:

```text
explicit user-approved firewall blocks
security incident enforcement
```

Authority:

```text
system mutation
requires exact human approval phrase
restricted lane
never part of media generation
```

## Chain Law

```text
Use existing agents first.
Do not create a new agent until the missing capability is documented.
No prompt-only agent becomes authority.
No media-generation path receives enforcement authority.
No generated media becomes evidence.
```

## Recommended Chains

### Chain: Observed Capture Replay

```text
TrueVision capture manifest
-> temporal_causality_log_checker if log stream exists
-> state replay script
-> replay manifest
-> state-media report
-> Central Writer later
```

Purpose:

```text
prove that captured state can replay deterministically
```

Human approval:

```text
not required unless a host snapshot is requested
```

### Chain: Synthetic State Generation

```text
operator intent
-> LLM state formula draft
-> ARC learning shape validation
-> local scene generator
-> TrueVision state capture of output
-> objective scoring
-> success/failure ledger
```

Purpose:

```text
learn the state language and improve generation through measured attempts
```

Human approval:

```text
not required for contained local synthetic artifacts
```

### Chain: Failure Regeneration

```text
failed output
-> extract TrueVision state
-> compare intended state vs generated state
-> classify failure family
-> fuse with prior success/failure records
-> emit candidate A/B state vectors
-> rerun contained generator
```

Purpose:

```text
use failures and successes as learning records
```

Human approval:

```text
not required unless accessing sensitive evidence or system state
```

### Chain: Security Context Snapshot

```text
operator approval
-> recovered_security_snapshot
-> generated report files
-> temporal_causality_log_checker if reports are converted to event logs
-> Central Writer later
```

Purpose:

```text
document the host/security context around a run
```

Human approval:

```text
required
```

### Chain: Firewall Enforcement

```text
explicit operator request
-> policy/approval gate
-> recovered_security_firewall_enforcer
-> snapshot/report
-> temporal verification where applicable
```

Purpose:

```text
security enforcement only
```

Human approval:

```text
required
```

Never route media generation through this chain.

## Central Writer Boundary

For full SecureCore integration later:

```text
agents produce structured records
Central Writer produces operator-facing reports
Policy Gate approves mutation
adapters act only after approval
```

This POC package may contain reports and handoff docs, but runtime SecureCore should eventually route all operator-facing output through Central Writer.

## Agent Request Contract

Any future runner that invokes these agents should use this minimum request shape:

```json
{
  "request_id": "string",
  "created_at_utc": "ISO-8601-Z",
  "requested_by": "truevision_state_media_poc",
  "agent_id": "string",
  "mode": "dry_run | read_only | approved",
  "approval": {
    "required": false,
    "phrase": "",
    "provided": false
  },
  "input_refs": [],
  "params": {},
  "expected_outputs": [],
  "risk_boundary": {
    "reads": [],
    "writes": [],
    "mutates_system": false
  }
}
```

## Rejection Rules

Reject a chain request if:

```text
agent_id is unknown
approval is missing for a gated agent
synthetic media is labeled as evidence
firewall enforcement is requested from a media chain
raw command execution is requested
facts lack evidence references
requested writes are outside the package/runtime lane
```

## Where New Agents May Be Needed Later

Document first, build later:

```text
truevision_capture_agent
truevision_replay_agent
truevision_state_scorer_agent
truevision_arc_fusion_agent
truevision_artifact_manifest_agent
central_writer_request_agent
```

These are not required for the current package. They are future candidates only after the existing agents are exhausted.

## Tiny Law

```text
Temporal checker verifies trails.
Reverse demo proves contained generation shape.
Snapshot reports host context with approval.
Firewall enforcer acts only with approval.
TrueVision learns inside the guardrails.
```

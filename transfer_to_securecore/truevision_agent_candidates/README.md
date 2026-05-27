# TrueVision Agent Candidates For SecureCore

This package stages SecureCore-bound agents only.

Workers stay in the organ repos. TrueVision workers stay with TrueVision.
SecureCore receives coordinating, reasoning, and zero-tolerance gate agents that
look across worker outputs.

```text
TrueVision logs.
TrueVision workers inspect.
SecureCore agents coordinate/gate.
Operators approve.
Receipts prove.
```

## Package Shape

```text
AGENTS/
  agents/
    *.agent.json
  catalog/
    agent_catalog.csv
    securecore_agents.csv
    securecore_export.json
  truevision-agents/
    *.py
HANDOFF_REPORT.md
```

The `*.agent.json` files mirror SecureCore's live-agent manifest contract:

```text
D:\SecureCore_Workspace\SecureCore\securecore\live_agents\manifest.py
```

## Boundary

These agents are staged candidates. Directory presence does not activate them.
SecureCore still owns promotion, registration, policy gates, and runtime tests.

## Zero-Tolerance Rule

```text
Only agents go to SecureCore.
Workers remain close to the organs.
No prompt-only agents.
No fake hashes.
No mutation without SecureCore approval.
```

